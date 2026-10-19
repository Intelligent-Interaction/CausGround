"""RoboCasa text frontend for the common retrieval engine.

This module interprets descriptions, instantiates declared contracts and filters
prohibited actions. It neither searches dependencies nor loads task answers.
The full native records remain available to the shared adapter and serializer.
"""
from copy import deepcopy
import re

from .robocasa_native.cg_contracts import enrich_query_contracts
from .robocasa_native.cg_goals import extract_goals
from .robocasa_native.cg_instantiate import template_instances
from .robocasa_native.cg_text import mentioned


def _build_pool(graph, task):
    pool, seen = [], set()
    for mechanism in graph["specific_mechanisms"] + graph.get("role_templates", []):
        if mechanism["id"] not in seen:
            pool.append(deepcopy(mechanism))
            seen.add(mechanism["id"])
    instances, requests = template_instances(graph, task)
    for mechanism in instances:
        if mechanism["id"] not in seen:
            pool.append(mechanism)
            seen.add(mechanism["id"])
    pool, notes = enrich_query_contracts(pool, task, graph)
    return pool, requests, notes


def _exclude_prohibited(pool, prohibitions):
    """Preserve the native action/entity prohibition test, without retrieval."""
    excluded = []
    verbs = {"OPEN": "open", "CLOSE": "close", "PICK": "pick|take|grab|retrieve",
             "PLACE": "place|put", "STIR": "stir", "SCRUB": "scrub|clean",
             "TILT": "tilt|pour", "WAIT": "wait", "SLIDE": "slide"}
    control_values = {"on", "off", "ON", "IN", "NEAR"}
    all_values = [value for mechanism in pool for value in mechanism["bindings"].values()
                  if not value.startswith("$") and value not in control_values]
    for mechanism in pool:
        for prohibition in prohibitions:
            text, action = prohibition["text"], mechanism["action"]
            if re.search(r"\b(?:start|turn|switch|activate)\b", text):
                desired = not bool(re.search(r"\boff\b", text))
                applies = any(effect.get("attribute") == "is_on" and effect.get("after") == desired
                              for effect in mechanism.get("effects", []))
            else:
                applies = action in verbs and bool(re.search(r"\b(?:" + verbs[action] + r")\b", text))
            if not applies:
                continue
            values = [value for value in mechanism["bindings"].values()
                      if value and not value.startswith("$") and value not in control_values]
            entities = mentioned(text, all_values)
            burner = "burner" in text and "burner" in mechanism["bindings"].get("$fixture", "")
            if not entities or mentioned(text, values) or burner:
                excluded.append({"mechanism_ref": mechanism["id"], "api_call": mechanism["api_call"],
                                 "prohibition": prohibition["text"]})
                break
    ids = {item["mechanism_ref"] for item in excluded}
    return [mechanism for mechanism in pool if mechanism["id"] not in ids], excluded


def _variables(value):
    if isinstance(value, str):
        return {value} if value.startswith("$") else set()
    if isinstance(value, dict):
        return set().union(*(_variables(item) for item in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_variables(item) for item in value)) if value else set()
    return set()


def _binding_declarations(extraction, pool):
    """Declare native placeholders without choosing a concrete scene instance."""
    declarations = {}
    for goal in extraction["goals"]:
        for variable in sorted(_variables(goal["predicate"])):
            declaration = declarations.setdefault(variable, {
                "grounded": False, "basis": "native_query_binding_constraint",
                "goal_ids": [], "constraints": []})
            if goal["id"] not in declaration["goal_ids"]:
                declaration["goal_ids"].append(goal["id"])
            for mechanism in pool:
                reference = goal.get("candidate_mechanism_ref")
                if reference and mechanism["id"] != reference:
                    continue
                for role, value in mechanism["bindings"].items():
                    if value != variable:
                        continue
                    constraint = goal.get("binding_constraints", {}).get(
                        role, mechanism.get("role_type_constraints", {}).get(role, {}))
                    item = {"role": role, **deepcopy(constraint)}
                    if item not in declaration["constraints"]:
                        declaration["constraints"].append(item)
                    quantifier = constraint.get("quantifier")
                    if quantifier or "each" in constraint.get("binding_policy", ""):
                        declaration["quantifier"] = quantifier or "each"
                        declaration["inventory_verified"] = False
    for ambiguity in extraction["ambiguities"]:
        equality = ambiguity.get("binding_equality")
        if equality:
            for variable in _variables(equality):
                declarations.setdefault(variable, {"grounded": False,
                    "basis": "native_binding_equality", "constraints": []})
    return declarations


def prepare(graph, task, vocabulary=None):
    """Return a native prepared library and the common extended GoalSpec.

    Only descriptions and the supplied library are used. Native templates stay
    templates; instantiated contracts retain their query provenance. Unenriched
    generic fallbacks are retained in metadata so they cannot bypass phase guards.
    """
    if not isinstance(graph, dict) or not isinstance(graph.get("specific_mechanisms"), list):
        raise ValueError("RoboCasa preparation requires a legacy mechanism library")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be nonempty text")
    if vocabulary is not None and not isinstance(vocabulary, dict):
        raise ValueError("vocabulary must be an object")
    pool, requests, notes = _build_pool(graph, task)
    extraction = extract_goals(task, pool)
    filtered, excluded = _exclude_prohibited(pool, extraction["prohibitions"])
    source_template_ids = {item["id"] for item in graph.get("role_templates", [])}
    prepared = deepcopy(graph)
    prepared["specific_mechanisms"] = [m for m in filtered if m["id"] not in source_template_ids]
    prepared["role_templates"] = [m for m in filtered if m["id"] in source_template_ids]
    prepared["generic_mechanisms"] = []
    unresolved = deepcopy(extraction["unmodeled_text_requirements"])
    if extraction["unparsed_description"]:
        unresolved.append({"text": task, "reason": "No text-grounded goal predicate from native RoboCasa parser"})
    spec = {"goals": [deepcopy(goal["predicate"]) for goal in extraction["goals"]],
            "goal_metadata": [{k: deepcopy(v) for k, v in goal.items() if k != "predicate"}
                              for goal in extraction["goals"]],
            "events": [], "ordering": [], "phase_ordering": deepcopy(extraction["temporal"]),
            "prohibitions": deepcopy(extraction["prohibitions"]),
            "excluded_mechanisms": [item["mechanism_ref"] for item in excluded],
            "binding_constraints": _binding_declarations(extraction, pool),
            "scene_facts": [deepcopy(fact) for note in notes for fact in note.get("scene_facts", [])],
            "ambiguities": deepcopy(extraction["ambiguities"]), "unresolved": unresolved,
            "basis": "native_robocasa_description_parser_not_execution_evidence"}
    return {"graph": prepared, "goalspec": spec, "metadata": {
        "frontend": "robocasa_native_text", "native_extraction": deepcopy(extraction),
        "query_contract_assumptions": deepcopy(notes), "instantiation_requests": deepcopy(requests),
        "excluded_mechanisms": deepcopy(excluded),
        "native_generic_mechanisms": deepcopy(graph.get("generic_mechanisms", [])),
        "task_id_used": False, "episode_labels_used": False, "execution_verified": False}}


def allows_bound_action(node, prohibitions, known_entities):
    """Reapply native prohibitions after generic role bindings are resolved."""
    if not prohibitions:return True
    action=node['action']
    candidate={'id':'candidate','api_call':action['name'],'action':action['name'],
               'bindings':{'$'+a['role']:a['node'] for a in action['arguments']},
               'effects':[fx for v in node.get('effect_variants',[]) for fx in v['effects']]}
    vocabulary={'id':'vocabulary','api_call':'','action':'','bindings':{str(i):v for i,v in enumerate(known_entities)},'effects':[]}
    kept,_=_exclude_prohibited([candidate,vocabulary],prohibitions)
    return any(x['id']=='candidate' for x in kept)
