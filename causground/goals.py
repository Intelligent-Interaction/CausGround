"""Goal predicates/events from explicit JSON, a small grammar, or an injected LLM."""
import json
from pathlib import Path
import re
from copy import deepcopy

from .knowledge import normalize_predicate
from .pipeline import strict_loads


def entity(text, vocabulary):
    text = re.sub(r"^(?:the|a|an|my)\s+", "", text.strip().lower()).replace(" ", "_")
    aliases = {k.lower().replace(" ", "_"): v for k,v in vocabulary.get("aliases", {}).items()}
    value = aliases.get(text, text)
    return value if value in vocabulary["objects"] else None


def validate_spec(spec, vocabulary, *, strict_entities=False):
    spec = deepcopy(spec)
    if isinstance(spec, list): spec = {"goals": spec}
    if not isinstance(spec, dict): raise ValueError("Goal specification must be an object or predicate list")
    extra_lists = {"goal_metadata", "phase_ordering", "prohibitions", "excluded_mechanisms", "scene_facts", "ambiguities"}
    if set(spec) - {"goals", "events", "ordering", "unresolved", "basis", "binding_constraints", *extra_lists}: raise ValueError("Unknown goal specification fields")
    constraints = spec.get("binding_constraints", {})
    if not isinstance(constraints, dict) or any(not isinstance(k,str) or not k.startswith('$') or not isinstance(v,dict) for k,v in constraints.items()):
        raise ValueError("binding_constraints must declare placeholder objects")
    aliases = {k.lower().replace(' ','_'):v for k,v in vocabulary.get('aliases',{}).items()}
    def canonical(value):
        if not isinstance(value,str): return value
        seen=set()
        while value.lower().replace(' ','_') in aliases:
            if value in seen: raise ValueError('Cyclic entity aliases')
            seen.add(value)
            target=aliases[value.lower().replace(' ','_')]
            if target == value: break
            value=target
        return value
    result = {k: spec.get(k, []) for k in ("goals", "events", "ordering", "unresolved")}
    if any(not isinstance(x, list) for x in result.values()): raise ValueError("Goal specification fields must be lists")
    def check(p, depth=0):
        if depth > 20 or not isinstance(p, dict): raise ValueError("Invalid goal predicate")
        p = normalize_predicate(p)
        kind = p.get("type")
        if kind in {"all_of", "any_of"}:
            if not isinstance(p.get("predicates"), list) or not p["predicates"]: raise ValueError("Empty goal conjunction/disjunction")
            p['predicates']=[check(child,depth+1) for child in p['predicates']]
        elif kind == "not": p['predicate']=check(p.get('predicate'),depth+1)
        elif kind in {"edge", "node_attribute", "node_exists", "node_type", "node_category"}:
            for k in (("source", "target") if kind == "edge" else ("node",)):
                p[k] = canonical(p.get(k))
                if not isinstance(p.get(k), str) or not p[k] or (p[k].startswith("$") and p[k] not in constraints) or p[k] == "*": raise ValueError("Goal objects must be explicit entities or declared constrained placeholders")
                if strict_entities and p[k] not in vocabulary["objects"] and p[k] not in constraints: raise ValueError("Unknown goal entity: " + p[k])
            if kind == "edge" and p.get("relation") not in vocabulary["relations"]: raise ValueError("Unknown goal relation: " + str(p.get("relation")))
            if kind == "node_attribute":
                if p.get("attribute") not in vocabulary["attributes"]: raise ValueError("Unknown goal attribute")
                op = p.get("operator", "equals")
                if op not in {"equals", "not_equals", "in", "not_in"}: raise ValueError("Unsupported goal attribute operator")
                if op in {"equals", "not_equals"} and "value" not in p: raise ValueError("Goal attribute requires value")
                if op in {"in", "not_in"} and not isinstance(p.get("values"), list): raise ValueError("Goal attribute requires values")
        else: raise ValueError("Unsupported goal predicate: " + str(kind))
        return p
    result["goals"] = [check(p) for p in result["goals"]]
    for event in result["events"]:
        if not isinstance(event, dict) or event.get("action") not in vocabulary["actions"]: raise ValueError("Unknown required action")
        bindings = event.get("bindings", {})
        bindings = {k:canonical(v) for k,v in bindings.items()} if isinstance(bindings,dict) else bindings
        event['bindings'] = bindings
        roles = {"$" + r for r in vocabulary["actions"][event["action"]]} | {"$actor"}
        if not isinstance(bindings, dict) or set(bindings) - roles: raise ValueError("Invalid required-event bindings")
        if any(not isinstance(v, str) for v in bindings.values()): raise ValueError("Event binding values must be strings")
        if strict_entities and any(v not in vocabulary["objects"] and v not in constraints for v in bindings.values()): raise ValueError("Unknown required-event entity")
        if 'phase' in event and (type(event['phase']) is not int or event['phase'] < 0): raise ValueError('Event phase must be a nonnegative integer')
    ids = {"goal:"+str(i) for i in range(len(result["goals"]))} | {"event:"+str(i) for i in range(len(result["events"]))}
    for order in result["ordering"]:
        if not isinstance(order, dict) or order.get("before") not in ids or order.get("after") not in ids or order["before"] == order["after"]: raise ValueError("Ordering must reference different goal:N or event:N IDs")
    result["basis"] = spec.get("basis", "provided_goal_spec")
    for field in extra_lists:
        if field in spec:
            if not isinstance(spec[field],list): raise ValueError(field+' must be a list')
            result[field]=spec[field]
    result['binding_constraints']=constraints
    metadata=result.get('goal_metadata', [{} for _ in result['goals']])
    if len(metadata)!=len(result['goals']) or any(not isinstance(m,dict) for m in metadata): raise ValueError('goal_metadata must align with goals')
    for m in metadata:
        if type(m.get('phase',0)) is not int or m.get('phase',0)<0: raise ValueError('Goal phase must be a nonnegative integer')
    result['goal_metadata']=metadata
    for order in result.get('phase_ordering',[]):
        if not isinstance(order,dict) or any(type(order.get(k)) is not int or order[k]<0 for k in ('before','after')) or order['before']==order['after']:raise ValueError('Invalid phase ordering')
    return result


class CommandGoalParser:
    """Restricted positive English command grammar. Unsupported text is retained."""
    def __call__(self, task, vocabulary):
        result = {"goals": [], "events": [], "ordering": [], "unresolved": [], "basis": "restricted_command_grammar"}
        # Do not silently discard negation, temporal/quantified requirements.
        if re.search(r"\b(?:not|never|don't|do\s+not|before|after|then|all|every|each|without|until)\b", task, re.I):
            result["unresolved"].append({"text": task, "reason": "Use goal JSON or LLMGoalParser for negation, ordering or quantifiers"})
            return result
        clauses = [c.strip() for c in re.split(r"\s+and\s+|[.;]", task.strip()) if c.strip()]
        if len(clauses) > 1:
            result["unresolved"].append({"text": task, "reason": "Compound commands need event/final-state interpretation; use goal JSON or LLMGoalParser"})
            return result
        for clause in clauses:
            text = clause.strip().lower()
            if not text: continue
            p = None
            match = re.fullmatch(r"(?:please\s+)?(open|close)\s+(.+)", text)
            if match and (obj := entity(match[2], vocabulary)):
                p = {"type": "node_attribute", "node": obj, "attribute": "is_open", "value": match[1] == "open"}
            match = re.fullmatch(r"(?:please\s+)?(?:turn|switch)\s+(on|off)\s+(.+)", text)
            if match and (obj := entity(match[2], vocabulary)):
                p = {"type": "node_attribute", "node": obj, "attribute": "is_on", "value": match[1] == "on"}
            match = re.fullmatch(r"(?:please\s+)?(?:pick up|grab|take|hold)\s+(.+)", text)
            if match and (obj := entity(match[1], vocabulary)):
                hands = [{"type": "edge", "source": vocabulary["actor"], "relation": rel, "target": obj} for rel in vocabulary["holding_relations"]]
                p = hands[0] if len(hands) == 1 else {"type": "any_of", "predicates": hands} if hands else None
            match = re.fullmatch(r"(?:please\s+)?(?:put|place)\s+(.+?)\s+(in|into|inside|on|onto)\s+(.+)", text)
            if match and (obj := entity(match[1], vocabulary)) and (target := entity(match[3], vocabulary)):
                relation = vocabulary["inside_relation"] if match[2] in {"in", "into", "inside"} else vocabulary["on_relation"]
                p = {"type": "edge", "source": obj, "relation": relation, "target": target}
            if p:
                try: result["goals"].extend(validate_spec([p], vocabulary, strict_entities=True)["goals"])
                except ValueError: p = None
            if not p: result["unresolved"].append({"text": clause, "reason": "Outside restricted grammar or environment vocabulary"})
        return result


class LLMGoalParser:
    def __init__(self, model, prompt=None):
        self.model = model
        self.prompt = prompt or (Path(__file__).resolve().parents[1] / "prompts/task_goals.txt").read_text(encoding="utf-8")

    def __call__(self, task, vocabulary):
        reply = self.model([{"role": "system", "content": self.prompt},
                            {"role": "user", "content": json.dumps({"task": task, "vocabulary": vocabulary}, ensure_ascii=False)}])
        spec = strict_loads(reply) if isinstance(reply, str) else reply
        result = validate_spec(spec, vocabulary, strict_entities=True)
        result["basis"] = "llm_interpreted_task_not_execution_evidence"
        return result
