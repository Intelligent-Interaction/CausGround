"""Shared bounded causal closure + independent retrieval union. No execution planner."""
from copy import deepcopy
from dataclasses import dataclass
import math
import re

from .builder import _state
from .goals import CommandGoalParser, validate_spec
from .knowledge import TRUE, adapt, bind, capabilities, document, ontology, predicate_shape, read_json, legacy_vocabulary, legacy_context_conditions
from .predicates import evaluate, index_state, token, SYMMETRIC
from .semantic import LexicalRetriever
from .contracts import invalidates, opposite, validate_initial_facts, canonicalize_facts


def _unify(pattern, target, bindings):
    if isinstance(pattern, str) and pattern.startswith("$"):
        if pattern in bindings: return bindings if bindings[pattern] == target else None
        if not isinstance(target, str) or target == "*": return None
        return {**bindings, pattern: target}
    if isinstance(pattern, dict) and isinstance(target, dict):
        if set(pattern) != set(target): return None
        result = dict(bindings)
        for k in pattern:
            result = _unify(pattern[k], target[k], result)
            if result is None: return None
        return result
    if isinstance(pattern, list) and isinstance(target, list) and len(pattern) == len(target):
        result = dict(bindings)
        for a,b in zip(pattern,target):
            result = _unify(a,b,result)
            if result is None: return None
        return result
    return dict(bindings) if type(pattern) is type(target) and pattern == target else None


def _matches(provided, needed, policy="standard"):
    provided, needed = predicate_shape(provided), predicate_shape(needed)
    direct = _unify(provided, needed, {})
    if direct is not None: return [(direct, False)]
    if provided.get("type") == needed.get("type") == "edge" and provided["relation"] in SYMMETRIC and policy!="virtualhome_legacy":
        reverse = {**needed, "source": needed["target"], "target": needed["source"]}
        direct = _unify(provided, reverse, {})
        if direct is not None: return [(direct, False)]
    if provided.get('type')==needed.get('type')=='node_attribute' and provided.get('operator')=='equals':
        actual=provided.get('value'); op=needed.get('operator')
        proves=(op=='not_equals' and type(actual) is type(needed.get('value')) and actual!=needed.get('value')) or (op=='in' and actual in needed.get('values',[])) or (op=='not_in' and actual not in needed.get('values',[]))
        if policy=='virtualhome_legacy':proves=proves and op=='not_equals' and isinstance(actual,bool) and isinstance(needed.get('value'),bool)
        if proves:
            b=_unify({'node':provided['node'],'attribute':provided['attribute']},{'node':needed['node'],'attribute':needed['attribute']},{})
            if b is not None:return [(b,False)]
    if needed.get("type") == "any_of":
        return [item for p in needed["predicates"] for item in _matches(provided, p, policy)]
    if provided.get("type") in {"any_of", "all_of"}:
        return [(b, uncertain or provided["type"] == "any_of") for p in provided["predicates"] for b,uncertain in _matches(p, needed, policy)]
    return []


def _complete_bindings(rule, bindings, vocabulary):
    bindings = dict(bindings)
    actor = rule["action"]["actor"]
    if actor.startswith("$"): bindings.setdefault(actor, vocabulary["actor"])
    aliases = rule.get("role_aliases", {})
    def propagate():
        for _ in range(len(aliases) + 1):
            changed = False
            for a,b in aliases.items():
                if a in bindings and b in bindings and bindings[a] != bindings[b]: return False
                if a in bindings and b not in bindings: bindings[b] = bindings[a]; changed = True
                if b in bindings and a not in bindings: bindings[a] = bindings[b]; changed = True
            if not changed: return True
        return True
    if not propagate(): return None
    for arg in rule["action"]["arguments"]:
        var = arg.get("node")
        if var and var.startswith("$") and var not in bindings:
            typ = rule["type_constraints"].get(arg["role"])
            if typ: bindings[var] = typ
    if not propagate(): return None
    for role, typ in rule["type_constraints"].items():
        var = "$" + role
        value = bindings.get(rule.get("role_aliases", {}).get(var, var))
        if value is None: continue
        if role == "actor" and value == vocabulary["actor"] and value not in vocabulary["object_types"]: continue
        actual = vocabulary["object_types"].get(value, value)
        # A schema-level actor token is deliberately independent of simulator IDs.
        if role == "actor" and value == "agent" and actual == "agent": continue
        if actual != typ: return None
    if rule["source_format"] == "builder" and rule["scope"] == "generic":
        for role, types in rule["observed_argument_types"].items():
            if role == "actor": continue
            value = bindings.get("$"+role)
            if value is not None and vocabulary["object_types"].get(value, value) not in types: return None
    for role, constraint in rule.get("role_type_constraints", {}).items():
        value = bindings.get(role)
        if value is None: continue
        typ = vocabulary["object_types"].get(value, value)
        declared = vocabulary.get("entity_types", {}).get(typ)
        if declared is None:
            # A constrained template cannot be rebound to an unknown type.
            if any(k in constraint for k in ("kind", "class", "group")): return None
            continue
        if constraint.get("kind") and constraint["kind"] != declared.get("kind"): return None
        if constraint.get("class") and constraint["class"] != declared.get("class"): return None
        if constraint.get("group") and constraint["group"] not in declared.get("groups", []): return None
    if rule["source_format"] == "legacy_robocasa" and rule["scope"] == "generic":
        for arg in rule["action"]["arguments"]:
            if arg["role"] in {"object", "tool", "container"}:
                value = bindings.get(arg["node"], arg["node"])
                if value.startswith("$"): continue
                typ = vocabulary["object_types"].get(value, value)
                if vocabulary.get("entity_types", {}).get(typ, {}).get("kind") != "movable": return None
    return bindings


def _binding_key(rule, bindings, phase=0):
    def variables(value):
        if isinstance(value,str):return {value} if value.startswith('$') else set()
        if isinstance(value,dict):return set().union(*(variables(v) for v in value.values())) if value else set()
        if isinstance(value,list):return set().union(*(variables(v) for v in value)) if value else set()
        return set()
    relevant=variables([rule[k] for k in ('action','conditions','condition_branches','postconditions','effect_variants','effect_branches')])
    effective={k:v for k,v in bindings.items() if k in relevant}
    return rule["id"], token(effective), phase


def _mentions(task, vocabulary):
    text = task.lower()
    words = {}
    for obj in vocabulary["objects"]:
        if re.search(r"(?<!\w)" + re.escape(obj.replace("_", " ").lower()) + r"(?!\w)", text): words[obj] = obj
    for alias, obj in vocabulary["aliases"].items():
        if re.search(r"(?<!\w)" + re.escape(alias.replace("_", " ").lower()) + r"(?!\w)", text): words[obj] = obj
    return sorted(words)


@dataclass
class RetrievalResult:
    subgraph: dict
    context: str


class CausGround:
    def __init__(self, graph, *, vocabulary=None, profile='auto'):
        if graph.get('schema_version')=='virtualhome-legacy-catalog-v1':
            from .adapters.virtualhome import catalog_to_graph
            graph=catalog_to_graph(graph)
        self.graph = deepcopy(graph)
        self.profile = (graph.get('profile') or ('robocasa' if graph.get('domain')=='RoboCasa' and graph.get('schema_version')=='causwm-object-graph-v2-multitrajectory' else 'standard')) if profile=='auto' else profile
        self.rules = adapt(graph)
        self.vocabulary = {**legacy_vocabulary(graph), **deepcopy(graph.get("vocabulary",{})), **deepcopy(vocabulary or {})} if "specific_mechanisms" in graph else {**deepcopy(graph.get("vocabulary",{})), **deepcopy(vocabulary or {})}
        self.domain = graph.get("domain", self.vocabulary.get("domain", "custom"))
        self.documents = [document(r) for r in self.rules]
        self._capabilities = [(r, c) for r in self.rules for c in capabilities(r)]

    @classmethod
    def load(cls, path, *, vocabulary=None, profile='auto'):
        if vocabulary is not None and not isinstance(vocabulary, dict): vocabulary = read_json(vocabulary)
        return cls(read_json(path), vocabulary=vocabulary, profile=profile)

    def retrieve(self, task, *, goals=None, mode="hybrid", state=None, initial_state=None, goal_parser=None,
                 retriever=None, top_k=5, max_nodes=32, max_depth=6, max_producers=2):
        if not isinstance(task, str) or not task.strip(): raise ValueError("task must be nonempty text")
        facts=validate_initial_facts(initial_state)
        if mode not in {"causal", "semantic", "hybrid"}: raise ValueError("mode must be causal, semantic or hybrid")
        for name, value in [("top_k", top_k), ("max_nodes", max_nodes), ("max_depth", max_depth), ("max_producers", max_producers)]:
            if type(value) is not int or value < 1: raise ValueError(name + " must be a positive integer")
        rules, documents, library_caps = self.rules, self.documents, self._capabilities
        frontend_metadata = {}
        query_vocabulary = self.vocabulary
        if self.profile == 'robocasa':
            from .adapters.robocasa import prepare
            prepared=prepare(self.graph,task,self.vocabulary)
            rules=adapt(prepared['graph'])
            documents=[document(r) for r in rules]
            library_caps=[(r,c) for r in rules for c in capabilities(r)]
            query_vocabulary={**legacy_vocabulary(prepared['graph']),**self.vocabulary}
            if goals is None and goal_parser is None and mode!='semantic': goals=prepared['goalspec']
            frontend_metadata=prepared.get('metadata',{})
        if state is not None:
            _state(state, "state")
            state = index_state(deepcopy(state))
        vocabulary = ontology(rules, query_vocabulary, state)
        if isinstance(goals,dict) and ('terminal_predicates' in goals or 'seed' in goals):
            from .adapters.virtualhome import seed_to_spec
            goals=seed_to_spec(goals,self.graph,aliases=vocabulary.get('aliases',{}))
        if mode == "semantic" and goals is None:
            spec = {"goals": [], "events": [], "ordering": [], "unresolved": [], "basis": "not_requested"}
        else:
            spec = validate_spec(goals if goals is not None else (goal_parser or CommandGoalParser())(task, vocabulary), vocabulary)
        if self.profile=='virtualhome_legacy':
            from .adapters.virtualhome import extend_catalog_for_events
            prepared_graph=extend_catalog_for_events(self.graph,spec)
            rules=adapt(prepared_graph)
            documents=[document(r) for r in rules]
            library_caps=[(r,c) for r in rules if r.get('provenance',{}).get('selection')!='dominant_role_schema_rebound' for c in capabilities(r)]
            frontend_metadata['role_schema_transfers']=prepared_graph.get('source_metadata',{}).get('role_schema_transfers',[])
        facts=canonicalize_facts(facts,vocabulary.get('aliases',{}))
        excluded={x if isinstance(x,str) else x.get('mechanism_ref') for x in spec.get('excluded_mechanisms',[])}
        scene_facts=validate_initial_facts(spec.get('scene_facts',[]))
        facts=([p for p in scene_facts if not any(opposite(p,q) or predicate_shape(p)==predicate_shape(q) for q in facts)]+facts)
        nodes, dependencies, unresolved, grounding, rankings, truncated = {}, [], [], [], [], []
        expanded = {}
        active_phase = 0
        active_meta = {}
        roots = {}
        event_links = []
        occurrences, choices = {}, {}
        def identity(rule, bindings, phase):
            key = _binding_key(rule, bindings, phase)
            return occurrences.setdefault(key, f"use:{len(occurrences) + 1:04d}")
        def allowed(rule, bindings):
            if rule['source_id'] in excluded:return False
            if self.profile=='robocasa':
                from .adapters.robocasa import allows_bound_action
                return allows_bound_action(bind(rule,bindings),spec.get('prohibitions',[]),vocabulary['objects'])
            return True

        def status(p):
            kind = p.get("type", p.get("kind"))
            if kind in {"all_of", "any_of", "derived_relation_condition"}:
                values = [status(x) for x in p["predicates"]]
                if kind == "any_of": return True if True in values else None if None in values else False
                conjunction = False if False in values else None if None in values else True
                if kind == "derived_relation_condition": return None if conjunction is None else not conjunction
                return conjunction
            if kind == "not":
                value = status(p["predicate"])
                return None if value is None else not value
            def free(v, bound=frozenset()):
                if isinstance(v, str): return v.startswith("$") and v not in bound
                if isinstance(v, dict):
                    if v.get("type") == "not_exists": bound = bound | {"$"+v["variable"]}
                    return any(free(x,bound) for x in v.values())
                if isinstance(v, list): return any(free(x,bound) for x in v)
                return False
            if free(p): return None
            if any(n['phase']<active_phase and invalidates(n,p) for n in nodes.values()):return None
            for fact in facts:
                if _matches(fact,p):return True
                if opposite(fact,p):return False
            return evaluate(p, state) if state is not None else (True if p == TRUE else None)

        def record_status(p, consumer, guard=TRUE):
            value = status(p)
            item = {"consumer": consumer, "predicate": deepcopy(p), "when": deepcopy(guard),
                    "status": "satisfied" if value is True else "violated" if value is False else "unknown"}
            if item not in grounding: grounding.append(item)
            return value

        def add(rule, bindings, source):
            id = identity(rule, bindings,active_phase)
            if id not in nodes:
                if len(nodes) >= max_nodes:
                    truncated.append({"source": source, "mechanism_ref": rule["id"], "bindings": bindings, "reason": "max_nodes"})
                    return None
                node = bind(deepcopy(rule), bindings)
                for scoped in node.get('query_phase_conditions',[]):
                    if scoped['from_phase']<=active_phase<=scoped['through_phase']:
                        if scoped['condition'] not in node['conditions']: node['conditions'].append(deepcopy(scoped['condition']))
                for condition in ([] if self.profile=='robocasa' else legacy_context_conditions(node, task, vocabulary)):
                    if condition not in node["conditions"]: node["conditions"].append(condition)
                # IDs and source references are never instantiated from task input.
                node.update(id=id, mechanism_ref=rule["source_id"], bindings=dict(bindings), sources=[], phase=active_phase,
                            binding_level="instance" if state and all(a.get("node") in {n["id"] for n in state["nodes"]} for a in node["action"]["arguments"]) else "type_or_unresolved",
                            execution_verified=False)
                nodes[id] = node
            if source not in nodes[id]["sources"]: nodes[id]["sources"].append(source)
            return id

        def candidates(p, requested_action=None, preferred=None):
            found = []
            for rule, cap in library_caps:
                if requested_action and rule['action']['name']!=requested_action: continue
                if rule['source_id'] in excluded:continue
                for bindings, alternative_only in _matches(cap["predicate"], p, self.profile):
                    bindings = _complete_bindings(rule, bindings, vocabulary)
                    if bindings is None or not allowed(rule,bindings): continue
                    grounded = bind(cap, bindings)
                    anchors = []
                    invalid_context = False
                    for context in cap.get("context_bindings", []):
                        value = bindings.get(context["variable"])
                        actual = vocabulary["object_types"].get(value, value)
                        if value is not None and context.get("type") and actual != context["type"]:
                            invalid_context = True; break
                        anchors.extend(bind({"type": "edge", **rel}, bindings) for rel in context.get("before_relations", []))
                    if invalid_context: continue
                    if anchors: grounded["when"] = {"type": "all_of", "predicates": [grounded["when"], *anchors]}
                    if status(grounded["when"]) is False: continue
                    found.append((rule, bindings, grounded, alternative_only))
            if self.profile=='virtualhome_legacy':
                # Preserve the old catalogue's exact/bool predicate matching and
                # support-first producer ranking as data-selection policy.
                found=[x for x in found if p.get('type')!='node_attribute' or p.get('operator')!='not_equals' or isinstance(p.get('value'),bool)]
                found.sort(key=lambda x:(-x[2].get('support',0),x[0]['action']['name'],tuple(a.get('node','') for a in x[0]['action']['arguments'])))
            else: found.sort(key=lambda x: (preferred is not None and x[0]['source_id']!=preferred, x[3] or x[2]["conditional_outcome"], x[0]["scope"] != "specific",
                        len(x[0]["conditions"]), -x[2].get("support", 0), x[0]["id"], token(x[1])))
            # Specific records take precedence over generic fallbacks for the same grounded action.
            specific_calls = {token(bind(r["action"], b)) for r,b,c,u in found if r["scope"] == "specific"}
            return [x for x in found if x[0]["scope"] == "specific" or token(bind(x[0]["action"], x[1])) not in specific_calls]

        def requirements(id, rule, bindings, depth, ancestors):
            if id in expanded and expanded[id]<=depth:return
            expanded[id]=depth
            unresolved[:]=[u for u in unresolved if not (u.get('consumer')==id and u.get('reason')=='max_depth')]
            for item in nodes[id]["conditions"]:
                satisfy(item["predicate"], id, depth+1, ancestors | {id}, TRUE, item.get("basis", "required_condition"))
            for branch in rule["condition_branches"]:
                guard = bind(branch.get("when", TRUE), bindings)
                # A false condition guard invalidates this conclusion, not the action.
                if status(guard) is False: continue
                formula = branch.get("success_formula") or {}
                predicates = ([formula["predicate"]] if formula.get("controlled_truth_table_complete") and "predicate" in formula else
                              [x["predicate"] for x in branch.get("required_conditions", [])])
                for p in predicates:
                    satisfy(bind(p, bindings), id, depth+1, ancestors | {id}, guard, branch.get("basis", "scoped_condition"))

        def satisfy(p, consumer, depth=0, ancestors=frozenset(), guard=TRUE, condition_basis="goal", choice=None):
            value = record_status(p, consumer, guard)
            if value is True: return
            if condition_basis != "goal" and p.get("type") in {"attribute", "node_attribute"}:
                resource = p.get("attribute") in vocabulary.get("resource_attributes", [])
                if resource:
                    unresolved.append({"consumer": consumer, "predicate": p, "when": guard, "reason": "requires_current_state_or_resource_resolution"})
                    return
            if depth > max_depth:
                unresolved.append({"consumer": consumer, "predicate": p, "reason": "max_depth", "when": guard}); return
            if p.get("type") == "all_of":
                for child in p["predicates"]: satisfy(child, consumer, depth, ancestors, guard, condition_basis, choice)
                return
            # Reuse an earlier occurrence only while its outcome has not been
            # invalidated by an intervening declared effect or location move.
            previous=[]
            for n in nodes.values():
                if n['phase']>=active_phase or n['id'] in ancestors or 'causal' not in n['sources']:continue
                if condition_basis=='goal' and active_meta.get('requested_action') and n['action']['name']!=active_meta['requested_action']:continue
                if any(other['id']!=n['id'] and n['phase']<=other['phase']<active_phase and invalidates(other,p) for other in nodes.values()):continue
                for cap in capabilities(n):
                    if predicate_shape(cap['predicate'])==predicate_shape(p) and status(cap['when']) is not False:previous.append((n,cap))
            if previous:
                n,cap=max(previous,key=lambda item:item[0]['phase'])
                dependencies.append({'producer':n['id'],'consumer':consumer,'predicate':deepcopy(p),'produced_predicate':cap['predicate'],'when':cap['when'],
                    'condition_when':deepcopy(guard),'condition_basis':condition_basis,'basis':cap['basis'],'conditional_outcome':cap['conditional_outcome'],
                    'variant_id':cap.get('variant_id'),'choice_group':choice,'support':cap.get('support',0),'execution_verified':False,'reused_prior_phase':True})
                if consumer.startswith(('goal:','event:')):roots.setdefault(consumer,[]).append(n['id'])
                return
            if self.profile=='robocasa' and condition_basis!='goal' and p.get('type')=='edge' and p.get('source')=='agent' and p.get('relation') in {'NEAR','AT_WORK_AREA'} and value is not False:
                unresolved.append({'consumer':consumer,'predicate':deepcopy(p),'reason':'initial_scene_state_unspecified','when':guard})
                return
            matches = candidates(p,active_meta.get('requested_action') if condition_basis=='goal' else None,
                                 active_meta.get('candidate_mechanism_ref') if condition_basis=='goal' else None)
            if not matches and p.get("type") == "any_of":
                group = choices.setdefault((consumer, token(p)), f"choice:{len(choices) + 1}")
                for child in p["predicates"]: satisfy(child, consumer, depth, ancestors, guard, condition_basis, group)
                return
            if self.profile=='virtualhome_legacy':
                # The historical bound limits ranked catalogue producers before
                # cycle pruning; a skipped cycle must not pull in a third action.
                selected=[]
                for r,b,c,u in matches:
                    key=identity(r,b,active_phase)
                    if key not in selected:selected.append(key)
                selected=set(selected[:max_producers])
                matches=[x for x in matches if identity(x[0],x[1],active_phase) in selected]
            chosen, seen = 0, set()
            for rule, bindings, capability, alternative_only in matches:
                key = identity(rule, bindings,active_phase)
                if key in ancestors:
                    unresolved.append({"consumer": consumer, "predicate": p, "reason": "dependency_cycle", "candidate": key})
                    # A cyclic contract remains useful knowledge, but is not a
                    # plan step and must never trigger another recursive expansion.
                    reference={"producer":key,"consumer":consumer,"predicate":deepcopy(p),
                        "produced_predicate":capability["predicate"],"when":capability["when"],
                        "condition_when":deepcopy(guard),"condition_basis":condition_basis,"basis":capability["basis"],
                        "conditional_outcome":alternative_only or capability["conditional_outcome"],"variant_id":capability.get("variant_id"),
                        "choice_group":choice,"support":capability.get("support",0),"execution_verified":False,"cycle_reference":True}
                    if reference not in dependencies:dependencies.append(reference)
                    continue
                if key not in seen:
                    if chosen >= max_producers:
                        truncated.append({"consumer": consumer, "predicate": p, "mechanism_ref": rule["id"], "reason": "max_producers"}); continue
                    chosen += 1; seen.add(key)
                id = add(rule, bindings, "causal")
                if id is None:
                    unresolved.append({"consumer": consumer, "predicate": p, "reason": "max_nodes"})
                    continue
                dependency = {"producer": id, "consumer": consumer, "predicate": deepcopy(p),
                    "produced_predicate": capability["predicate"], "when": capability["when"],
                    "condition_when": deepcopy(guard), "condition_basis": condition_basis,
                    "basis": capability["basis"], "conditional_outcome": alternative_only or capability["conditional_outcome"],
                    "variant_id": capability.get("variant_id"), "choice_group": choice,
                    "support": capability.get("support", 0), "execution_verified": False}
                if dependency not in dependencies: dependencies.append(dependency)
                if consumer.startswith(('goal:','event:')):roots.setdefault(consumer,[]).append(id)
                if dependency["conditional_outcome"]:
                    note = {"consumer": consumer, "predicate": p, "reason": "possible_outcome_only", "candidate": id}
                    if note not in unresolved: unresolved.append(note)
                requirements(id, rule, bindings, depth, ancestors)
            if not matches:
                unresolved.append({"consumer": consumer, "predicate": deepcopy(p), "when": guard,
                                   "reason": "no_supported_producer", "choice_group": choice})

        if mode in {"causal", "hybrid"}:
            requests=[(m.get('phase',0),'goal',i,p,m) for i,(p,m) in enumerate(zip(spec['goals'],spec.get('goal_metadata',[{}]*len(spec['goals']))))]
            requests += [(e.get('phase',0),'event',i,e,{}) for i,e in enumerate(spec['events'])]
            for active_phase,kind,i,item,active_meta in sorted(requests,key=lambda x:(x[0],x[1]=='event',x[2])):
                if kind=='goal':
                    satisfy(item,'goal:'+str(i));continue
                event=item
                matches = []
                for rule in rules:
                    if rule["action"]["name"] != event["action"] or rule["source_id"] in excluded: continue
                    b = _complete_bindings(rule, event.get("bindings", {}), vocabulary)
                    if b is not None and allowed(rule,b):
                        # Legacy fixed bindings must match the requested event too.
                        action = bind(rule["action"], b)
                        fixed = {"$"+a["role"]: a.get("node") for a in action["arguments"]}
                        if all(fixed.get(k, v) == v for k,v in event.get("bindings", {}).items() if k != "$actor"): matches.append((rule,b))
                if not matches: unresolved.append({"consumer": "event:"+str(i), "event": event, "reason": "no_event_mechanism"})
                for rule,b in matches[:max_producers]:
                    id = add(rule,b,"causal")
                    if id:
                        roots.setdefault('event:'+str(i),[]).append(id)
                        event_links.append({'event':'event:'+str(i),'mechanism':id,'phase':active_phase})
                        requirements(id,rule,b,-1,set())

        ranker_name = None
        if mode in {"semantic", "hybrid"}:
            retriever = retriever or LexicalRetriever()
            ranker_name = getattr(retriever, "name", "injected_ranker")
            scores = list(retriever(task, documents))
            if len(scores) != len(rules) or any(not math.isfinite(float(s)) for s in scores): raise ValueError("Retriever must return one finite score per mechanism")
            ordered = sorted(zip(rules, scores), key=lambda x: (-float(x[1]), x[0]["scope"] != "specific", x[0]["id"]))
            for rule,score in [x for x in ordered if x[1] > 0 and x[0]["source_id"] not in excluded][:top_k]:
                existing = [n for n in nodes.values() if n["mechanism_ref"] == rule["source_id"]]
                active_phase=existing[0]['phase'] if existing else 0
                bindings = existing[0]["bindings"] if existing else _complete_bindings(rule, {}, vocabulary)
                if bindings is None or not allowed(rule,bindings): continue
                id = add(rule, bindings, "semantic")
                rankings.append({"mechanism_ref": rule["id"], "node_id": id, "score": float(score), "retained": id is not None})
        for node in nodes.values():
            active_phase=node['phase']
            node["candidate_only"] = node["sources"] == ["semantic"]
            for item in node["conditions"]: record_status(item["predicate"], node["id"])
        phase_order=[]
        for order in spec.get('phase_ordering',[]):
            before={id for ids in roots.values() for id in ids if nodes[id]['phase']==order['before']}
            later={id for id,n in nodes.items() if n['phase']==order['after'] and 'causal' in n['sources']}
            # Requested phase order constrains the entry prerequisites too.
            after=later-{d['consumer'] for d in dependencies if d['producer'] in later and d['consumer'] in later}
            if not after: after=later
            phase_order.extend({'source':a,'target':b,'basis':order.get('basis','requested_text_order'),'before_phase':order['before'],'after_phase':order['after']} for a in sorted(before) for b in sorted(after) if a!=b)
        conditional=[{'node':n['id'],'bindings':{a['role']:a.get('node') for a in n['action']['arguments']},'constraints':spec.get('binding_constraints',{})}
                     for n in nodes.values() if any(a.get('node','').startswith('$') for a in n['action']['arguments'])]
        output = {"schema_version": "causground-task-subgraph-v1", "domain": self.domain, "task": task, "mode": mode,
                  "goal_spec": spec, "mechanisms": list(nodes.values()), "dependencies": dependencies,
                  "semantic_rankings": rankings, "retriever": ranker_name,
                  "grounding": grounding, "unresolved": spec["unresolved"] + unresolved,
                  "truncated": truncated, "text_order_constraints": spec["ordering"], "complete_execution_plan": False,
                  "phase_order_constraints":phase_order,"event_links":event_links,"goal_links":roots,"conditional_bindings":conditional,
                  "frontend_metadata":frontend_metadata,"profile":self.profile,
                  "initial_facts":facts, "context_policy":vocabulary.get("context_policy","full"),
                  "knowledge_scope": "Retrieved evidence and declared mechanisms; alternatives are not a required action sequence"}
        from .diagnostics import analyze
        output.update(analyze(output, vocabulary))
        from .context import render_context
        return RetrievalResult(output, render_context(output))
