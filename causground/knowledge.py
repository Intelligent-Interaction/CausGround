"""Read-only views of builder graphs and the legacy RoboCasa mechanism library."""
from copy import deepcopy
import gzip
import json
import re

from .predicates import token


TRUE = {"type": "all_of", "predicates": []}


def read_json(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def bind(value, bindings):
    if isinstance(value, dict): return {k: bind(v, bindings) for k, v in value.items()}
    if isinstance(value, list): return [bind(v, bindings) for v in value]
    return bindings.get(value, value) if isinstance(value, str) else value


def normalize_predicate(p):
    p = deepcopy(p)
    if p.get("type", p.get("kind")) in {"attribute", "node_attribute"}:
        p.pop("kind", None)
        p["type"] = "node_attribute"
        if 'object' in p and 'node' not in p:p['node']=p.pop('object')
        p.setdefault("operator", "equals")
    if p.get("kind") == "edge": p["type"] = p.pop("kind")
    if "predicates" in p: p["predicates"] = [normalize_predicate(x) for x in p["predicates"]]
    if "predicate" in p: p["predicate"] = normalize_predicate(p["predicate"])
    if p.get("type") == "edge" and "present" in p:
        present = p.pop("present")
        if not isinstance(present, bool): raise ValueError("edge.present must be boolean")
        if not present: return {"type": "not", "predicate": p}
    return p


def predicate_shape(p):
    """Metadata is not part of predicate identity; boolean structure is."""
    p = normalize_predicate(p)
    kind = p.get("type")
    fields = {"edge": ("source", "relation", "target"),
              "node_attribute": ("node", "attribute", "operator", "value", "values"),
              "node_exists": ("node",), "node_type": ("node", "value"),
              "node_category": ("node", "value")}.get(kind)
    if fields: return {"type": kind, **{k: p[k] for k in fields if k in p}}
    if kind in {"any_of", "all_of", "derived_relation_condition", "not_exists"}:
        return {"type": kind, **{k: p[k] for k in ("operator", "variable") if k in p},
                "predicates": [predicate_shape(x) for x in p["predicates"]]}
    if kind == "not": return {"type": "not", "predicate": predicate_shape(p["predicate"])}
    return p


def effect_predicate(effect):
    op = effect.get("operation")
    if op in {"set_attribute", "update_attribute"} and effect.get("after") is not None:
        return {"type": "node_attribute", "node": effect["node"], "attribute": effect["attribute"],
                "operator": "equals", "value": effect["after"]}
    if op in {"add_edge", "remove_edge", "delete_edge"}:
        p = {"type": "edge", **effect["edge"]}
        return p if op == "add_edge" else {"type": "not", "predicate": p}
    if op in {"add_node", "remove_node"}:
        node = effect.get("node")
        if isinstance(node, dict): node = node.get("id")
        if node:
            p = {"type": "node_exists", "node": node}
            return p if op == "add_node" else {"type": "not", "predicate": p}
    return None


def _v3(rule):
    tail, head = rule.get("tail", {}), rule.get("head", {})
    result = {"id": rule["id"], "source_id": rule["id"], "source_format": rule.get('source_format','builder'),
              "scope": rule.get("scope", "specific"), "action": deepcopy(rule["action"]),
              "type_constraints": deepcopy(rule.get("type_constraints", {})),
              "observed_argument_types": deepcopy(rule.get("observed_argument_types", {})),
              "role_aliases": deepcopy(rule.get("role_aliases", rule["action"].get("role_aliases", {}))),
              "conditions": deepcopy(tail.get("required_conditions", [])),
              "observational_conditions": deepcopy(tail.get("observational_conditions", [])),
              "condition_branches": deepcopy(tail.get("condition_branches", [])),
              "condition_status": tail.get("status", "unknown"),
              "postconditions": deepcopy(head.get("postconditions", [])),
              "effect_variants": deepcopy(head.get("effect_variants", [])),
              "effect_branches": deepcopy(head.get("effect_branches", [])),
              "generalization_status": rule.get("generalization_status", "unknown"),
              "support": head.get("successful_executions", 0),
              "provenance": {"evidence": deepcopy(rule.get("evidence", {})), **{k: deepcopy(tail[k]) for k in ("intervention_rule_ids", "contextual_intervention_rule_ids") if k in tail}},
              "rejected_candidates": deepcopy(tail.get("rejected_candidates", [])),
              "pending_conditions": deepcopy(tail.get("pending_conditions", [])),
              "pending_postconditions": deepcopy(head.get("pending_postconditions", [])),
              "rejected_postconditions": deepcopy(head.get("rejected_postconditions", []))}
    for key in ('native', 'provenance', 'description', 'role_types', 'query_phase_conditions'):
        if key in rule: result[key]=deepcopy(rule[key])
    return result


def _legacy(rule, scope):
    """Keep declared contracts/intended outcomes distinct from measured support."""
    rule = deepcopy(rule)
    bindings = rule.get("bindings", {})
    # Explicitly preserve guards previously added by RoboCasa cg_contracts.py.
    for fx in rule.get("effects", []):
        if rule["action"] == "TILT" and fx.get("operation") == "add_edge" and fx.get("edge", {}).get("source") != "agent":
            fx.setdefault("applicability_condition", {"type": "edge", "source": fx["edge"]["source"], "relation": "INSIDE", "target": bindings.get("$container", "$container")})
            fx.setdefault("applicability_basis", "legacy_recorded_content_membership_guard")
    if any(fx.get("attribute") == "washed_locations" for fx in rule.get("effects", [])):
        rule.setdefault("conditions", []).append({"predicate": {"type": "node_attribute", "node": "sink", "attribute": "is_on", "value": True}, "basis": "legacy_process_definition_guard_water_running", "necessity_verified": False})
    arguments = [{"role": k.lstrip("$"), "node": v} for k, v in bindings.items()]
    if not arguments:
        arguments = [{"role": p, "node": "$" + p} for p in rule.get("parameters", [])]
    effects = deepcopy(rule.get("effects", []))
    posts = []
    for fx in effects:
        p = effect_predicate(fx)
        if p:
            posts.append({"predicate": p, "basis": fx.get("basis", "legacy_declared_effect"),
                          "when": fx.get("applicability_condition", TRUE), "necessity_verified": False})
    return {"id": rule["id"], "source_id": rule["id"], "source_format": "legacy_robocasa",
            "native": deepcopy(rule), "query_phase_conditions":deepcopy(rule.get('query_phase_conditions',[])),
            "scope": scope, "action": {"name": rule["action"], "actor": "agent", "arguments": arguments},
            "conditions": [{**deepcopy(c), "predicate": normalize_predicate(c["predicate"])} for c in rule.get("conditions", [])], "observational_conditions": [],
            "condition_branches": [], "postconditions": posts,
            "effect_variants": [{"id": rule["id"] + ":declared", "effects": effects,
                                 "context_bindings": [], "basis": "legacy_declared_effects"}],
            "effect_branches": [], "type_constraints": {}, "observed_argument_types": {}, "role_aliases": {},
            "role_type_constraints": deepcopy(rule.get("role_type_constraints", {})),
            "evidence_statistics": deepcopy(rule.get("evidence_statistics", {})),
            "multitrajectory_observation_groups": deepcopy(rule.get("multitrajectory_observation_groups", [])),
            "provenance": {"source_id": rule["id"], "evidence_ids": deepcopy(rule.get("evidence_ids", []))},
            "support": len(rule.get("evidence_ids", [])),
            "generalization_status": "declared_contract" if scope == "generic" else "legacy_recorded_binding"}


def _legacy_variants(rule, scope):
    parameterized = next((fx for fx in rule.get("effects", []) if fx.get("operation") in {"add_edge_from_argument", "set_attribute_from_argument"}), None)
    if not parameterized: return [_legacy(rule, scope)]
    edge_operation = parameterized["operation"] == "add_edge_from_argument"
    variable = parameterized["edge"]["relation_argument"] if edge_operation else parameterized["argument"]
    mapping = parameterized["relation_value_map"] if edge_operation else parameterized["argument_value_map"]
    values = rule.get("bindings", {})
    variants = []
    for value, outcome in mapping.items():
        if variable in values and values[variable] != value: continue
        copied = bind(deepcopy(rule), {variable: value})
        for original, fx in zip(rule["effects"], copied["effects"]):
            if fx.get("operation") == "add_edge_from_argument" and original["edge"]["relation_argument"] == variable:
                fx["operation"] = "add_edge"
                fx["edge"].pop("relation_argument")
                fx["edge"]["relation"] = fx["relation_value_map"][value]
            if fx.get("operation") == "set_attribute_from_argument" and original["argument"] == variable:
                fx["operation"] = "set_attribute"
                fx["after"] = fx["argument_value_map"][value]
        for result in _legacy_variants(copied, scope):
            for arg in result["action"]["arguments"]:
                if arg["role"] == variable.lstrip("$"): arg["node"] = value
            result["id"] += ":" + value
            variants.append(result)
    return variants


def adapt(graph):
    version = graph.get("schema_version")
    if version is not None and version not in {"causground.offline.v3", "causground-v3", "causwm-object-graph-v2-multitrajectory"}:
        raise ValueError("Unsupported graph schema_version: " + str(version))
    if "specific_rules" in graph:
        rules = [_v3(x) for x in graph.get("specific_rules", []) + graph.get("generic_rules", [])]
    elif "specific_mechanisms" in graph:
        rules = [r for key, scope in [("specific_mechanisms", "specific"),
                  ("role_templates", "template"), ("generic_mechanisms", "generic")] for x in graph.get(key, []) for r in _legacy_variants(x, scope)]
    else:
        raise ValueError("Expected a builder graph or legacy RoboCasa mechanism library")
    if len({r["id"] for r in rules}) != len(rules): raise ValueError("Mechanism IDs must be unique")
    return rules


def capabilities(rule):
    """Each candidate outcome carries its guard and epistemic status."""
    result = []
    for post in rule["postconditions"]:
        result.append({"predicate": normalize_predicate(post["predicate"]), "when": deepcopy(post.get("when", TRUE)),
                       "basis": post.get("basis", "observed_postcondition"), "conditional_outcome": rule["source_format"] != "builder" or post.get('conditional_outcome',False),
                       "evidence_kind": "declared_or_intended" if rule["source_format"] != "builder" else "observed_stable_postcondition",
                       "support": post.get("support", rule.get("support", 0)), "kind": "postcondition"})
    # A stable postcondition is stronger than a particular observed delta.
    stable = {token(predicate_shape(x["predicate"])) for x in result if x["when"] == TRUE}
    if rule["source_format"] == "builder":
        for variant in rule["effect_variants"]:
            branches = [b for b in rule["effect_branches"] if any(a["variant_id"] == variant["id"] for a in b["alternatives"])]
            for index, effect in enumerate(variant.get("effects", [])):
                if index in variant.get("unconfirmed_effects", []): continue
                p = effect_predicate(effect)
                if not p or token(predicate_shape(p)) in stable: continue
                for branch in branches or [{"when": TRUE, "alternatives": []}]:
                    result.append({"predicate": p, "when": deepcopy(branch["when"]),
                        "basis": "observed_effect_variant", "variant_id": variant["id"],
                        "conditional_outcome": True, "kind": "variant",
                        "change_confirmed": index not in variant.get("unconfirmed_effects", []),
                        "support": variant.get("support", 0),
                        "context_bindings": deepcopy(variant.get("context_bindings", [])),
                        "generalization_verified": False})
    return result


def ontology(rules, vocabulary=None, state=None):
    vocabulary = deepcopy(vocabulary or {})
    objects, relations, attributes, actions = set(), set(), set(), {}
    def walk(v):
        if isinstance(v, dict):
            if "relation" in v and isinstance(v["relation"], str): relations.add(v["relation"])
            if "attribute" in v: attributes.add(v["attribute"])
            for key in ("node", "source", "target"):
                x = v.get(key)
                if isinstance(x, str) and not x.startswith("$") and x != "*": objects.add(x)
            for x in v.values(): walk(x)
        elif isinstance(v, list):
            for x in v: walk(x)
    for r in rules:
        actions.setdefault(r["action"]["name"], sorted(a["role"] for a in r["action"]["arguments"]))
        objects.update(r["type_constraints"].values())
        for types in r["observed_argument_types"].values(): objects.update(types)
        walk({k: r[k] for k in ("action", "conditions", "postconditions", "effect_variants", "condition_branches")})
    object_types = {o: o for o in objects}
    object_types.update(vocabulary.get("object_types", {}))
    if state:
        object_types.update({n["id"]: n["type"] for n in state["nodes"]})
    relations.discard("$relation")
    actor = vocabulary.get("actor", "agent")
    objects.add(actor)
    return {**vocabulary, "objects": sorted(set(object_types) | {actor}), "object_types": object_types,
            "relations": sorted(relations), "attributes": sorted(attributes), "actions": actions,
            "aliases": vocabulary.get("aliases", {}), "actor": actor,
            "holding_relations": vocabulary.get("holding_relations", [r for r in ("HOLDS", "HOLDS_LH", "HOLDS_RH") if r in relations]),
            "inside_relation": vocabulary.get("inside_relation", "INSIDE"),
            "on_relation": vocabulary.get("on_relation", "ON"),
            "entity_types": vocabulary.get("entity_types", {}),
            "resource_attributes": vocabulary.get("resource_attributes", [])}


def legacy_vocabulary(graph):
    """Type declarations from the legacy library, not inferred simulator facts."""
    entities = {name: {"kind": "movable", "class": name, "groups": groups}
                for name,groups in graph.get("public_object_type_hierarchy", {}).items()}
    for rule in graph.get("specific_mechanisms", []):
        for role, value in rule.get("bindings", {}).items():
            if value.startswith("$"): continue
            definitions = rule.get("role_types", {}).get(role, [])
            if isinstance(definitions, dict): definitions = [definitions]
            if definitions: entities[value] = deepcopy(definitions[0])
    return {"entity_types": entities, "resource_attributes": ["gripper_empty"]}


def legacy_context_conditions(mechanism, task, vocabulary):
    """Old query enclosure access assumptions, separate from learned conditions."""
    if mechanism["source_format"] != "legacy_robocasa": return []
    args = {a["role"]: a.get("node") for a in mechanism["action"]["arguments"]}
    enclosures = {"fridge", "freezer", "drawer", "cabinet", "microwave", "toaster_oven", "dishwasher"}
    targets = set()
    if mechanism["action"]["name"] == "PLACE" and args.get("relation") == "IN":
        target = args.get("target")
        if vocabulary["object_types"].get(target, target) in enclosures: targets.add(target)
    if mechanism["action"]["name"] == "PICK" and args.get("object") and not args["object"].startswith("$"):
        obj = re.escape(args["object"].replace("_", " "))
        for enclosure in enclosures:
            pattern = r"\b(?:pick up|take|grab|retrieve)\s+(?:the\s+)?" + obj + r"\s+from\s+(?:(?:the|a|an|open|closed)\s+)*" + re.escape(enclosure.replace("_", " ")) + r"\b"
            if re.search(pattern, task, re.I): targets.add(enclosure)
    return [{"predicate": {"type": "node_attribute", "node": target, "attribute": "is_open", "operator": "equals", "value": True},
             "basis": "legacy_query_enclosure_access_contract_assumption", "necessity_verified": False,
             "scope": "Declared source/destination enclosure access; scene-specific access unverified"} for target in sorted(targets)]


def document(rule):
    fields = {"action": rule["action"], "types": rule["type_constraints"],
              "postconditions": [p["predicate"] for p in rule["postconditions"]],
              "conditions": [p["predicate"] for p in rule["conditions"]]}
    return {"id": rule["id"], "text": json.dumps(fields, ensure_ascii=False)}
