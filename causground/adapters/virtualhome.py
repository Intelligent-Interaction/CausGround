"""Conversions for the historical VirtualHome catalogue and planner protocol.

The catalogue is offline source material. These functions convert records and
serialize the common retriever's result; they do not select producer chains.
"""
from copy import deepcopy
import itertools
import json
import re

from ..knowledge import TRUE, effect_predicate, normalize_predicate, predicate_shape


PRIVATE_PROFILES = {"planning75", "virtualhome_legacy", "mechanisms_only"}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _aliases(aliases):
    aliases = aliases or {}
    if "aliases" in aliases: aliases = aliases["aliases"]
    elif "original_to_planning_class" in aliases: aliases = aliases["original_to_planning_class"]
    if not isinstance(aliases, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in aliases.items()):
        raise ValueError("Aliases must map strings to canonical class strings")
    return {_normalize(k): _normalize(v) for k, v in aliases.items()}


def _normalize(value):
    if not isinstance(value, str) or value.startswith("$") or value == "*": return value
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def _canonical(value, aliases):
    """Legacy records contain class tokens, not runtime instance handles."""
    if isinstance(value, str) and not value.startswith("$"):
        value = re.sub(r"_\d+$", "", value)
    value = _normalize(value)
    seen = set()
    while value in aliases and aliases[value] != value:
        if value in seen: raise ValueError("Cyclic object aliases")
        seen.add(value)
        value = aliases[value]
    return value


def _objects(value, aliases):
    if isinstance(value, list): return [_objects(x, aliases) for x in value]
    if not isinstance(value, dict): return deepcopy(value)
    return {k: _canonical(v, aliases) if k in {"node", "source", "target", "object"} and isinstance(v, str)
            else _objects(v, aliases) for k, v in value.items()}


def _native_rules(catalog):
    if catalog is None: return []
    if isinstance(catalog, list): rows = catalog
    elif not isinstance(catalog, dict): raise ValueError("Expected a native catalogue, rule list or exported catalogue object")
    elif "rules" in catalog: rows = catalog["rules"]
    elif "specific_rules" in catalog:
        rows = [r.get("native") or {"action": r["action"]["name"],
                  "arguments": [{"role": a["role"], "object": a["node"]} for a in r["action"].get("arguments", [])]}
                for r in catalog.get("specific_rules", []) + catalog.get("generic_rules", [])]
    else: rows = list(catalog.values())
    if not isinstance(rows, list) or any(not isinstance(r, dict) or not isinstance(r.get("action"), str) for r in rows):
        raise ValueError("Native catalogue rules require action strings")
    return rows


def catalog_to_graph(catalog, aliases=None, metadata=None, *, id_offset=0):
    """Wrap native selected-rule records in the common v3 input schema.

    A recorded dominant schema is a legacy declaration, not a newly verified
    universal postcondition. Original records and source provenance stay intact.
    """
    native_aliases = catalog.get("aliases", {}) if isinstance(catalog, dict) else {}
    aliases = {**_aliases(native_aliases), **_aliases(aliases)}
    aliases.setdefault("character", "agent")
    rules, identifiers = [], {}
    for native in _native_rules(catalog):
        row = _objects(native, aliases)
        arguments = [{"role": a["role"], "node": a.get("object", a.get("node"))} for a in row.get("arguments", [])]
        if not arguments and row.get("target") is not None:
            arguments = [{"role": "target", "node": row["target"]}]
        signature = {"action": row["action"], "arguments": arguments,
                     "conditions": row.get("required_conditions", []), "effects": row.get("essential_effects", [])}
        identity = identifiers.setdefault(_json(signature), f"legacy-vh:{id_offset + len(identifiers) + 1:04d}")
        conditions = []
        for item in row.get("required_conditions", []):
            condition = deepcopy(item)
            condition["predicate"] = normalize_predicate(condition["predicate"])
            condition.setdefault("basis", "legacy_catalog_condition")
            conditions.append(condition)
        effects = deepcopy(row.get("essential_effects", []))
        transferred = row.get("provenance", {}).get("selection") == "dominant_role_schema_rebound"
        support = 0 if transferred else row.get("provenance", {}).get("selected_schema_support", 0)
        outcome_basis = "legacy_role_schema_transfer" if transferred else "legacy_catalog"
        posts = []
        for effect in effects:
            predicate = effect_predicate(effect)
            if predicate is not None:
                posts.append({"predicate": predicate, "basis": outcome_basis,
                              "when": deepcopy(effect.get("applicability_condition", TRUE)),
                              "conditional_outcome": True, "evidence_kind": "declared_or_intended",
                              "support": support, "necessity_verified": False})
        rules.append({"id": identity, "scope": "specific", "source_format": "legacy_virtualhome",
                      "action": {"name": row["action"], "actor": "agent", "arguments": arguments},
                      "type_constraints": {}, "observed_argument_types": {},
                      "tail": {"required_conditions": conditions, "observational_conditions": [], "condition_branches": [],
                               "status": "legacy_catalog_conditions", "condition_schema_status": row.get("condition_schema_status", "legacy_declared")},
                      "head": {"postconditions": posts, "effect_variants": [{"id": identity + ":declared",
                               "effects": effects, "context_bindings": [], "basis": outcome_basis, "support": support}],
                               "effect_branches": [], "successful_executions": support},
                      "generalization_status": "legacy_role_schema_transfer_not_target_observed" if transferred else "legacy_selected_class_schema_not_newly_verified",
                      "native": deepcopy(native), "provenance": deepcopy(native.get("provenance", {}))})
    identities = [r["id"] for r in rules]
    if len(set(identities)) != len(identities):
        # Aliases may collapse identical source schemas; retain their provenance
        # instead of counting the same normalized mechanism twice.
        merged = {}
        for rule in rules:
            if rule["id"] in merged:
                merged[rule["id"]].setdefault("native_alternatives", []).append(rule["native"])
            else: merged[rule["id"]] = rule
        rules = list(merged.values())
    source_metadata = deepcopy(catalog.get("provenance", {})) if isinstance(catalog, dict) else {}
    source_metadata.update(deepcopy(metadata or {}))
    return {"schema_version": "causground.offline.v3", "domain": "VirtualHome", "profile": "virtualhome_legacy",
            "source_format": "legacy_virtualhome", "source_metadata": source_metadata,
            "vocabulary": {"domain": "VirtualHome", "actor": "agent", "aliases": aliases,
                           "context_policy": "mechanisms_only", "holding_relations": ["HOLDS_LH", "HOLDS_RH"]},
            "specific_rules": sorted(rules, key=lambda r: r["id"]), "generic_rules": [],
            "representation": "native_selected_catalog_wrapped_for_shared_retrieval",
            "condition_semantics": "Historical selected schemas, with original audits retained; not re-induced necessary conditions.",
            "effect_semantics": "Declared legacy outcomes remain conditional; source selection is not universal outcome verification."}


def _replace_objects(value, bindings):
    # Match the historical transfer operation exactly: whole string values,
    # recursively, never textual substring replacement or rule inference.
    if isinstance(value, dict): return {k: _replace_objects(v, bindings) for k, v in value.items()}
    if isinstance(value, list): return [_replace_objects(v, bindings) for v in value]
    return bindings.get(value, value) if isinstance(value, str) else value


def extend_catalog_for_events(catalog_or_graph, spec, aliases=None):
    """Apply the old dominant role-schema transfer to explicit missing events.

    This is query preparation, not producer search. Only action name, argument
    arity and the original support/roles/arguments ordering choose the source.
    Transfers never become source candidates for later transfers. No goal,
    reference plan, task ID, scene inventory or evaluator fact is consulted.
    """
    is_graph = isinstance(catalog_or_graph, dict) and "specific_rules" in catalog_or_graph
    graph = deepcopy(catalog_or_graph) if is_graph else catalog_to_graph(catalog_or_graph, aliases)
    aliases = {**_aliases(graph.get("vocabulary", {}).get("aliases", {})), **_aliases(aliases)}
    native = [_objects(r, aliases) for r in _native_rules(catalog_or_graph)]
    source = {}
    for row in native:
        args = tuple(a.get("object", a.get("node")) for a in row.get("arguments", []))
        if not args and row.get("target") is not None: args = (row["target"],)
        source[(row["action"], args)] = row
    existing = set(source)
    # A prior query's transferred records are not empirical source schemas.
    source = {key: row for key, row in source.items() if row.get("provenance", {}).get("selection") != "dominant_role_schema_rebound"}
    seed = spec.get("seed", spec)
    events = seed.get("required_events", seed.get("events", []))
    transfers = []
    for event in events:
        action = event.get("action")
        if "arguments" in event or "target" in event:
            args = event.get("arguments", []) or ([event["target"]] if event.get("target") is not None else [])
        elif "native" in event:
            original = event["native"]
            args = original.get("arguments", []) or ([original["target"]] if original.get("target") is not None else [])
        else:
            bindings = event.get("bindings", {})
            role_options = {tuple(a["role"] for a in row.get("arguments", [])) for (name, _), row in source.items()
                            if name == action and {"$" + a["role"] for a in row.get("arguments", [])} == set(bindings) - {"$actor"}}
            if len(role_options) != 1: continue
            args = [bindings["$" + role] for role in next(iter(role_options))]
        args = tuple(_canonical(v, aliases) for v in args)
        if (action, args) in existing or any(not isinstance(v, str) or v.startswith("$") for v in args): continue
        candidates = []
        for (name, source_args), row in source.items():
            if name != action or len(source_args) != len(args): continue
            roles = tuple(a.get("role") for a in row.get("arguments", []))
            support = row.get("provenance", {}).get("selected_schema_support", 0)
            candidates.append((support, roles, source_args, row))
        if not candidates: continue
        _, roles, source_args, source_rule = sorted(candidates, key=lambda x: (-x[0], x[1], x[2]))[0]
        bindings = dict(zip(source_args, args))
        row = _replace_objects(deepcopy(source_rule), bindings)
        row["provenance"] = {**row.get("provenance", {}), "selection": "dominant_role_schema_rebound",
            "schema_source_action": action, "schema_source_arguments": list(source_args), "schema_target_arguments": list(args),
            "argument_roles": list(roles), "object_bindings": bindings, "target_observation_count": 0,
            "support_scope": "source_class_schema_only_not_transferred_target", "transfer_algorithm": "original_virtualhome_dominant_role_schema_v3"}
        transfers.append(row)
        existing.add((action, args))
    if transfers:
        occupied = [r['id'] for r in graph['specific_rules'] + graph.get('generic_rules', [])]
        offset = max((int(value.split(':')[1]) for value in occupied if re.fullmatch(r'legacy-vh:\d+', value)), default=0)
        additions = catalog_to_graph(transfers, aliases, id_offset=offset)["specific_rules"]
        graph["specific_rules"] = sorted(graph["specific_rules"] + additions, key=lambda rule: rule["id"])
        graph.setdefault("source_metadata", {}).setdefault("role_schema_transfers", []).extend(
            {"rule_id": rule["id"], **deepcopy(rule["provenance"])} for rule in additions)
    return graph


def _roles(catalog):
    result = {}
    for rule in _native_rules(catalog):
        roles = tuple(a["role"] for a in rule.get("arguments", []))
        if not roles and rule.get("target") is not None: roles = ("target",)
        result.setdefault(rule["action"], set()).add(roles)
    return result


def _semantic_expansion(predicate, bindings):
    """Keep a semantic candidate set disjunctive while retaining nested logic."""
    predicate = deepcopy(predicate)
    if "predicates" in predicate:
        predicate["predicates"] = [_semantic_expansion(p, bindings) for p in predicate["predicates"]]
    if "predicate" in predicate:
        predicate["predicate"] = _semantic_expansion(predicate["predicate"], bindings)
    fields = [k for k in ("node", "source", "target") if isinstance(predicate.get(k), str) and predicate[k] in bindings]
    if not fields: return predicate
    choices = [bindings[predicate[k]]["candidate_classes"] for k in fields]
    if any(not values for values in choices): return predicate
    variants = []
    for values in itertools.product(*choices):
        variants.append({**deepcopy(predicate), **dict(zip(fields, values))})
    return variants[0] if len(variants) == 1 else {"type": "any_of", "predicates": variants,
                                                  "meaning": "legacy_global_ontology_candidates_not_scene_inventory"}


def seed_to_spec(seed, catalog_or_graph=None, aliases=None, profile="planning75"):
    """Translate a supplied historical compiler seed; never look up a task ID."""
    wrapper = seed
    seed = seed.get("seed", seed)
    if not isinstance(seed, dict): raise ValueError("Expected a legacy compiler seed")
    inherited_aliases = _aliases(catalog_or_graph.get("aliases", {})) if isinstance(catalog_or_graph, dict) else {}
    aliases = {**inherited_aliases, **_aliases(aliases)}
    if isinstance(catalog_or_graph, dict):
        aliases = {**_aliases(catalog_or_graph.get("vocabulary", {}).get("aliases", {})), **aliases}
    aliases.setdefault("character", "agent")
    semantic = {}
    for name, entry in seed.get("semantic_bindings", {}).items():
        if entry.get("scene_inventory_exposed") is not False:
            raise ValueError("semantic_bindings require scene_inventory_exposed=false")
        semantic[_canonical(name, aliases)] = {**deepcopy(entry), "candidate_classes": sorted({_canonical(x, aliases) for x in entry.get("candidate_classes", [])})}
    spec = {"goals": [], "goal_metadata": [], "events": [], "ordering": [], "unresolved": [],
            "binding_constraints": {name if name.startswith("$") else "$" + name: deepcopy(entry) for name, entry in semantic.items()},
            "basis": "replayed_legacy_compiler_seed_not_live_inference"}
    references = {}
    for index, predicate in enumerate(seed.get("terminal_predicates", [])):
        converted = _semantic_expansion(_objects(predicate, aliases), semantic)
        spec["goals"].append(normalize_predicate(converted))
        source_id = predicate.get("id", "goal:" + str(index))
        spec["goal_metadata"].append({"id": source_id, "phase": predicate.get("phase", 0), "scope": predicate.get("scope", "final"),
                                      "native_predicate": deepcopy(predicate)})
        references[source_id] = "goal:" + str(index)
        references["goal:" + str(index)] = "goal:" + str(index)
    role_catalog = _roles(catalog_or_graph)
    for event in seed.get("required_events", []):
        action = event.get("action")
        values = event.get("arguments", []) or ([event["target"]] if event.get("target") is not None else [])
        values = [_canonical(x, aliases) for x in values]
        roles = sorted(r for r in role_catalog.get(action, set()) if len(r) == len(values))
        if len(roles) != 1 or any(v in semantic for v in values):
            spec["unresolved"].append({"native_event": deepcopy(event), "reason": "unknown_or_ambiguous_event_roles" if len(roles) != 1 else "semantic_event_candidates_require_explicit_binding"})
            continue
        index = len(spec["events"])
        converted = {"action": action, "bindings": {"$" + role: value for role, value in zip(roles[0], values)},
                     "id": event.get("id", "event:" + str(index)), "native": deepcopy(event)}
        if "phase" in event: converted["phase"] = event["phase"]
        spec["events"].append(converted)
        references[converted["id"]] = "event:" + str(index)
        references["event:" + str(index)] = "event:" + str(index)
    if profile not in PRIVATE_PROFILES:
        for order in seed.get("ordering_constraints", []):
            before, after = references.get(order.get("before")), references.get(order.get("after"))
            if before is not None and after is not None and before != after:
                spec["ordering"].append({"before": before, "after": after})
            else: spec["unresolved"].append({"native_ordering": deepcopy(order), "reason": "unresolved_legacy_ordering_reference"})
    for item in seed.get("unresolved_terms", []): spec["unresolved"].append(deepcopy(item))
    for item in wrapper.get("goal_to_cwm_projection", {}).get("unresolved_goal_facts", []):
        spec["unresolved"].append({"native_goal_fact": deepcopy(item), "reason": "legacy_goal_projection_unresolved"})
    return spec


def _project_fact(fact, vocabulary):
    """Project the old public compiler schema, without consulting task answers."""
    if not isinstance(fact, dict): raise ValueError("Goal facts must be objects")
    result = deepcopy(fact)
    kind = fact.get("type")
    if kind in {"all_of", "any_of"}:
        if not fact.get("predicates"): raise ValueError("Empty goal fact group")
        result["predicates"] = [_project_fact(p, vocabulary) for p in fact["predicates"]]
    elif kind == "not":
        result["predicate"] = _project_fact(fact["predicate"], vocabulary)
    elif kind == "relation":
        result.update(type="edge", source=result.pop("subject"), target=result.pop("object"))
        result.setdefault("present", True)
    elif kind == "node_attribute":
        if "object" in result: result["node"] = result.pop("object")
        attribute, value = fact.get("attribute"), fact.get("value")
        enum_projection = {"power": ("is_on", {"on": True, "off": False}),
                           "openness": ("is_open", {"open": True, "closed": False})}
        if attribute in enum_projection:
            name, values = enum_projection[attribute]
            if not isinstance(value, str) or value not in values: raise ValueError("Unknown " + attribute + " value")
            result.update(attribute=name, value=values[value])
        elif attribute not in vocabulary.get("attributes", []):
            raise ValueError("Unsupported old goal attribute: " + str(attribute))
        result.setdefault("operator", "equals")
    elif kind not in {"edge", "node_exists", "node_type", "node_category"}:
        raise ValueError("Unsupported old goal fact type: " + str(kind))
    return result


def _goal_spec_seed(goal_spec, vocabulary):
    seed = {"terminal_predicates": [], "required_events": deepcopy(goal_spec.get("required_events", [])),
            "ordering_constraints": deepcopy(goal_spec.get("ordering_constraints", [])),
            "semantic_bindings": deepcopy(goal_spec.get("semantic_bindings", {})),
            "unresolved_terms": deepcopy(goal_spec.get("unresolved_terms", []))}
    for fact in goal_spec.get("terminal_facts", []):
        try:
            projected = _project_fact(fact, vocabulary)
            projected["native_goal_fact"] = deepcopy(fact)
            seed["terminal_predicates"].append(projected)
        except (ValueError, KeyError, TypeError) as error:
            seed["unresolved_terms"].append({"native_goal_fact": deepcopy(fact), "reason": "unsupported_legacy_goal_projection", "detail": str(error)})
    return seed


class LegacyVirtualHomeGoalParser:
    """Inject a model callback returning a native seed or old ``terminal_facts``.

    The callback sees only task text and the global rule vocabulary. Supplying a
    saved compiled seed directly to retrieval remains the separate replay path;
    this class does not load benchmark task IDs or provide a hosted model.
    """
    def __init__(self, model, catalog_or_graph=None, aliases=None, profile="planning75"):
        self.model, self.catalog = model, catalog_or_graph
        self.aliases, self.profile = aliases, profile

    def __call__(self, task, vocabulary):
        from ..goals import validate_spec
        from ..pipeline import strict_loads
        prompt = ("Return one JSON object containing terminal_predicates, required_events, ordering_constraints and optional semantic_bindings; "
                  "or the historical terminal_facts schema (node_attribute with object, attribute, value; relation with subject, relation, object). "
                  "Use only the supplied global vocabulary. Required event arguments are ordered object class strings. "
                  "Do not infer current scene state, instances, evaluator goals or reference plans. Record unsupported interpretations in unresolved_terms. "
                  "Every semantic binding must explicitly set scene_inventory_exposed=false.")
        reply = self.model([{"role": "system", "content": prompt},
                            {"role": "user", "content": json.dumps({"task": task, "vocabulary": vocabulary}, ensure_ascii=False)}])
        payload = strict_loads(reply) if isinstance(reply, str) else reply
        if not isinstance(payload, dict): raise ValueError("Expected a legacy seed or terminal_facts object")
        if "seed" in payload:
            seed = payload
        elif "terminal_facts" in payload or "terminal_facts" in payload.get("goal_spec", {}):
            seed = _goal_spec_seed(payload.get("goal_spec", payload), vocabulary)
        elif set(payload) & {"terminal_predicates", "required_events"}:
            seed = payload
        else: raise ValueError("Expected a legacy seed or terminal_facts object")
        catalog = self.catalog
        if catalog is None:
            catalog = [{"action": action, "arguments": [{"role": role, "object": "$" + role} for role in roles]}
                       for action, roles in vocabulary.get("actions", {}).items()]
        aliases = {**_aliases(vocabulary.get("aliases", {})), **_aliases(self.aliases)}
        spec = seed_to_spec(seed, catalog, aliases, self.profile)
        spec["basis"] = "model_callback_legacy_schema_interpretation_not_execution_evidence"
        spec = validate_spec(spec, vocabulary)
        def missing_entities(predicate):
            kind = predicate.get("type")
            if kind in {"all_of", "any_of"}:
                return set().union(*(missing_entities(p) for p in predicate["predicates"]))
            if kind == "not": return missing_entities(predicate["predicate"])
            return {predicate[k] for k in (("source", "target") if kind == "edge" else ("node",))
                    if predicate.get(k) not in vocabulary["objects"] and predicate.get(k) not in spec["binding_constraints"]}
        for index, predicate in enumerate(spec["goals"]):
            for entity in sorted(missing_entities(predicate)):
                spec["unresolved"].append({"consumer": "goal:" + str(index), "object": entity,
                                           "reason": "object_class_absent_from_rule_library"})
        for index, event in enumerate(spec["events"]):
            for entity in sorted(set(event["bindings"].values()) - set(vocabulary["objects"]) - set(spec["binding_constraints"])):
                spec["unresolved"].append({"consumer": "event:" + str(index), "object": entity,
                                           "reason": "object_class_absent_from_rule_library"})
        return spec


def _legacy_predicate(predicate):
    p = deepcopy(predicate)
    if p.get("type") == "not":
        inner = _legacy_predicate(p["predicate"])
        if inner.get("type") == "edge": return {**inner, "present": not inner.get("present", True)}
        if inner.get("type") == "node_attribute" and inner.get("operator", "equals") == "equals":
            return {**inner, "operator": "not_equals"}
    if "predicates" in p: p["predicates"] = [_legacy_predicate(x) for x in p["predicates"]]
    return p


def _predicate_label(p):
    if p.get("type") == "edge": return "{} {} {}".format(p.get("source"), p.get("relation"), p.get("target"))
    if p.get("type") == "node_attribute":
        return "{}.{} {} {}".format(p.get("node"), p.get("attribute"), p.get("operator", "equals"), json.dumps(p.get("value")))
    return _json(p)


def _action_label(rule):
    return rule["action"] + "(" + ", ".join(str(a["object"]) for a in rule.get("arguments", [])) + ")"


def to_legacy_subgraph(result, task=None, seed=None, profile="planning75"):
    """Serialize shared retrieval into the old planner's input shape.

    ``planning75`` hides compiler goals/events/order. It retains conditional
    rule knowledge and actual producer links, never manufactures event order.
    """
    graph = result.subgraph if hasattr(result, "subgraph") else result
    task = task or {}
    if isinstance(task, str): task = {"id": task}
    private = profile in PRIVATE_PROFILES
    rules, labels, relations, objects = [], {}, {}, {"agent"}
    for mechanism in graph.get("mechanisms", []):
        action = mechanism["action"]
        arguments = [{"role": a["role"], "object": a.get("node", a.get("value"))} for a in action.get("arguments", [])]
        native = mechanism.get("native", {})
        variants = mechanism.get("effect_variants", [])
        # Only the single selected native schema is the old essential-effects
        # contract. Multiple empirical variants must never be flattened to AND.
        effects = deepcopy(variants[0].get("effects", [])) if len(variants) == 1 and mechanism.get("source_format") == "legacy_virtualhome" else []
        rule = {"action": action["name"], "target": next((a["object"] for a in arguments if a["role"] == "target"), arguments[0]["object"] if arguments else None),
                "arguments": arguments, "required_conditions": [{**deepcopy(c), "predicate": _legacy_predicate(c["predicate"])} for c in mechanism.get("conditions", [])],
                "essential_effects": effects, "condition_schema_status": native.get("condition_schema_status", "legacy_declared"),
                "provenance": {**deepcopy(native.get("provenance", {})), "shared_mechanism_ref": mechanism.get("mechanism_ref"),
                               "outcome_semantics": "conditional_legacy_declaration_not_universal_verification"},
                "shared_occurrence_id": mechanism["id"], "sources": deepcopy(mechanism.get("sources", [])),
                "binding_level": mechanism.get("binding_level", "type_or_unresolved"),
                "candidate_only": mechanism.get("candidate_only", False), "execution_verified": False,
                "condition_branches": deepcopy(mechanism.get("condition_branches", [])),
                "effect_branches": deepcopy(mechanism.get("effect_branches", [])),
                "effect_variants": deepcopy(variants)}
        rules.append(rule)
        labels[mechanism["id"]] = _action_label(rule)
        objects.update(a["object"] for a in arguments if isinstance(a["object"], str))
        for p in [c["predicate"] for c in rule["required_conditions"]] + [e["edge"] for e in effects if "edge" in e]:
            if all(k in p for k in ("source", "relation", "target")):
                key = (p["source"], p["relation"], p["target"])
                relations[key] = {"source": key[0], "relation": key[1], "target": key[2], "status": "causal_rule_relation_not_current_state_assertion"}
    dependencies = []
    choices = {}
    def add_edge(item):
        if item not in dependencies: dependencies.append(item)
    for dependency in graph.get("dependencies", []):
        producer, consumer = dependency["producer"], dependency["consumer"]
        if producer not in labels or (private and consumer not in labels): continue
        requirement = predicate_shape(dependency["predicate"]) if private else deepcopy(dependency["predicate"])
        supported = requirement
        disjunctive = requirement.get("type") == "any_of"
        if disjunctive and dependency.get("produced_predicate") is not None:
            supported = predicate_shape(dependency["produced_predicate"]) if private else deepcopy(dependency["produced_predicate"])
        predicate = _predicate_label(_legacy_predicate(supported))
        details = {"source": dependency.get("basis", "shared_core"), "possible_outcome_only": dependency.get("conditional_outcome", True),
                   "when": predicate_shape(dependency.get("when", TRUE)) if private else deepcopy(dependency.get("when", TRUE)),
                   "requirement_when": predicate_shape(dependency.get("condition_when", TRUE)) if private else deepcopy(dependency.get("condition_when", TRUE))}
        if disjunctive:
            details["requirement_formula"] = _legacy_predicate(requirement)
            details["choice_group"] = dependency.get("choice_group") or choices.setdefault(
                (consumer, _json(predicate_shape(requirement))), f"requirement:{len(choices) + 1}")
        elif dependency.get("choice_group"):
            details["choice_group"] = dependency["choice_group"]
        for key in ("cycle_reference", "variant_id"):
            if key in dependency: details[key] = deepcopy(dependency[key])
        add_edge({"from": labels[producer], "relation": "produces", "to": predicate, **details})
        add_edge({"from": predicate, "relation": "required_by", "to": labels.get(consumer, "TASK_TERMINAL"), **details})
    native_seed = deepcopy((seed or {}).get("seed", seed or {}))
    if private:
        native_seed = {"terminal_predicates": [], "required_events": [], "ordering_constraints": []}
    elif not native_seed:
        spec = graph.get("goal_spec", {})
        native_seed = {"terminal_predicates": deepcopy(spec.get("goals", [])),
                       "required_events": [{"action": e["action"], "arguments": list(e.get("bindings", {}).values())} for e in spec.get("events", [])],
                       "ordering_constraints": deepcopy(spec.get("ordering", []))}
    unresolved = [deepcopy(x) for x in graph.get("unresolved", []) if not private or x.get("consumer") in labels]
    if private:
        unresolved = [{k: predicate_shape(v) if k in {"predicate", "when"} else deepcopy(v)
                       for k, v in x.items() if k in {"consumer", "predicate", "when", "reason", "candidate", "choice_group"}}
                      for x in unresolved]
    truncated = graph.get("truncated", [])
    truncation_counts = {}
    for item in truncated:
        reason = item.get("reason", "other")
        if reason not in {"max_nodes", "max_depth", "max_producers"}: reason = "other"
        truncation_counts[reason] = truncation_counts.get(reason, 0) + 1
    result = {"schema_version": "task-conditioned-causal-subgraph-v3", "task_id": task.get("id", graph.get("task_id", "query")),
              "task_instruction": task.get("instruction", graph.get("task", "")),
              "retrieval_policy": "shared_core_retrieval_serialized_for_legacy_virtualhome", "retrieved_once_per_episode": True,
              "seed_spec": native_seed, "semantic_bindings": deepcopy(native_seed.get("semantic_bindings", {})),
              "task_compiler": {"input_policy": "compiler_private_retrieval_only" if private else "explicit_goal_spec",
                                "inference_replayed": "replayed" in graph.get("goal_spec", {}).get("basis", ""),
                                "evaluator_goal_exposed": False, "reference_plan_exposed": False},
              "compiler_dependency_edges_removed": private,
              "nodes": [{"id": name, "type": "character" if name == "agent" else name} for name in sorted(objects)],
              "relations": [relations[k] for k in sorted(relations)], "action_hyperedges": rules,
              "dependency_edges": dependencies, "frontier_conditions": unresolved,
              "unresolved_required_events": [] if private else [x for x in unresolved if x.get("consumer", "").startswith("event:")],
              "cycle_or_shared_dependencies": [x for x in unresolved if x.get("reason") == "dependency_cycle"],
              "candidate_actions": list(dict.fromkeys(_action_label(r) for r in rules)),
              "retrieval_audit": {"shared_core": True, "profile": profile, "truncated": [] if private else deepcopy(truncated),
                                  "truncation_counts": truncation_counts},
              "complete_execution_plan": False,
              "usage_note": "Bounded reference knowledge, not a plan or current state. Legacy outcomes and semantic-only mechanisms are candidates; preserve current RGB/history grounding."}
    return result
