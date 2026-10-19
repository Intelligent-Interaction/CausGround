"""VirtualHome symbolic-state adapter, extracted from the existing VH mapper.

No action precondition tables or simulator execution dependencies are imported.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import zipfile

from .builder import validate_transition


STATE_ATTRIBUTES = {
    "OPEN": ("is_open", True), "CLOSED": ("is_open", False),
    "ON": ("is_on", True), "OFF": ("is_on", False),
    "PLUGGED_IN": ("is_plugged_in", True), "PLUGGED_OUT": ("is_plugged_in", False),
    "CLEAN": ("is_clean", True), "DIRTY": ("is_clean", False),
    "SITTING": ("posture", "sitting"), "LYING": ("posture", "lying"),
}
PROPERTY_ATTRIBUTES = {
    "SURFACES": "affordance.surface", "GRABBABLE": "affordance.grabbable",
    "SITTABLE": "affordance.sittable", "LIEABLE": "affordance.lieable",
    "HANGABLE": "affordance.hangable", "DRINKABLE": "affordance.drinkable",
    "EATABLE": "affordance.eatable", "RECIPIENT": "affordance.recipient",
    "CUTTABLE": "affordance.cuttable", "POURABLE": "affordance.pourable",
    "CAN_OPEN": "affordance.openable", "HAS_SWITCH": "affordance.switchable",
    "READABLE": "affordance.readable", "LOOKABLE": "affordance.lookable",
    "CONTAINERS": "affordance.container", "CLOTHES": "semantic_type.clothing",
    "PERSON": "semantic_type.person", "BODY_PART": "semantic_type.body_part",
    "COVER_OBJECT": "affordance.cover_object", "HAS_PLUG": "affordance.pluggable",
    "HAS_PAPER": "has_paper", "MOVABLE": "affordance.movable", "CREAM": "semantic_type.cream",
}
SINGLE_TARGET_ACTIONS = "OPEN CLOSE CUT DRINK SWITCHON SWITCHOFF DROP EAT FIND PLUGIN PLUGOUT GRAB GREET LOOKAT POINTAT PULL PUSH PUTOFF PUTON READ RINSE RUN SCRUB SQUEEZE TOUCH TURNTO TYPE WALK WASH WATCH WIPE".split()
ACTION_ROLES = {name: ("target",) for name in SINGLE_TARGET_ACTIONS}
ACTION_ROLES.update({"PUTIN": ("source", "destination"), "PUTBACK": ("source", "destination"),
                     "POUR": ("source", "destination"), "PUTOBJBACK": ("source",),
                     "SIT": ("support",), "LIE": ("support",)})


def _ref(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("VirtualHome node IDs must be integers")
    return f"vh:{value}"


def _symbol(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value.strip().upper()


def _symbols(node, key):
    values = node.get(key, [])
    if not isinstance(values, list):
        raise ValueError(f"node {node.get('id')}: {key} must be a list of strings")
    return [_symbol(value, key) for value in values]


def convert_state(raw):
    if not isinstance(raw, dict) or not isinstance(raw.get("nodes"), list) or not isinstance(raw.get("edges"), list):
        raise ValueError("VirtualHome state requires nodes and edges lists")
    nodes = []
    for node in raw["nodes"]:
        attributes = {}
        for state in _symbols(node, "states"):
            attr, value = STATE_ATTRIBUTES.get(state, ("state." + state.lower(), True))
            if attr in attributes and attributes[attr] != value:
                raise ValueError(f"conflicting states on node {node.get('id')}: {attr}")
            attributes[attr] = value
        for prop in _symbols(node, "properties"):
            attributes[PROPERTY_ATTRIBUTES.get(prop, "affordance." + prop.lower())] = True
        nodes.append({"id": _ref(node["id"]), "type": _symbol(node.get("class_name"), "class_name").lower(),
                      "category": node.get("category", ""),
                      "attributes": dict(sorted(attributes.items()))})
    edges = [{"source": _ref(e["from_id"]), "relation": _symbol(e["relation_type"], "relation_type"),
              "target": _ref(e["to_id"])} for e in raw["edges"]]
    return {"nodes": sorted(nodes, key=lambda n: n["id"]),
            "edges": sorted(edges, key=lambda e: (e["source"], e["relation"], e["target"])),
            "relations_complete": True, "attributes_complete": True, "objects_complete": True}


def convert_transition(raw):
    """Convert a recorded symbolic VH execution; does not run the simulator."""
    try:
        name = _symbol(raw["action"]["name"], "action.name")
        argument_ids = raw["action"]["argument_ids"]
        roles = ACTION_ROLES.get(name)
        if roles is not None and len(roles) != len(argument_ids):
            raise ValueError(f"{name} expects {len(roles)} arguments, got {len(argument_ids)}")
        if roles is None:
            roles = tuple(f"arg_{i}" for i in range(len(argument_ids)))
        record = {
            "id": raw["id"], "before": convert_state(raw["before"]), "after": convert_state(raw["after"]),
            "action": {"name": name, "actor": _ref(raw["action"]["actor_id"]),
                       "arguments": [{"role": role, "node": _ref(node_id)} for role, node_id in zip(roles, argument_ids)]},
            "outcome": raw["outcome"], "provenance": deepcopy(raw.get("provenance", {})),
        }
        validate_transition(record)
        return record
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed VirtualHome transition: {exc}") from exc


def iter_dataset(dataset, exclude_programs=(), include_programs=None):
    """Read complete official episodes from a directory or ZIP, one episode at a time."""
    dataset = Path(dataset)
    excluded = set(exclude_programs)
    archive = zipfile.ZipFile(dataset) if dataset.is_file() else None
    try:
        if archive:
            programs = sorted(n for n in archive.namelist() if "/executable_programs/" in n and n.endswith(".txt"))
            read = archive.read
        else:
            programs = (sorted((dataset / "executable_programs" / p).as_posix() for p in include_programs)
                        if include_programs is not None else sorted(p.as_posix() for p in (dataset / "executable_programs").rglob("*.txt")))
            read = lambda p: Path(p).read_bytes()
        if not programs: raise ValueError("No executable_programs found")
        for program in programs:
            relative = program.split("/executable_programs/", 1)[1]
            if relative in excluded: continue
            states_path = program.replace("/executable_programs/", "/state_list/").rsplit(".", 1)[0] + ".json"
            actions = []
            for line in read(program).decode("utf-8-sig").splitlines():
                match = re.match(r"^\s*\[([A-Za-z_]+)\]", line)
                if match:
                    ids = [int(value.rsplit(".", 1)[-1]) for value in re.findall(r"<[^>]+>\s*\(([^)]+)\)", line)]
                    actions.append((match.group(1).upper(), ids, line.strip()))
            states = json.loads(read(states_path))["graph_state_list"]
            if len(states) != len(actions) + 1:
                raise ValueError(f"Action/state length mismatch: {relative}")
            canonical_before = convert_state(states[0])
            for index, (name, ids, line) in enumerate(actions):
                actors = [n["id"] for n in states[index]["nodes"] if n.get("class_name", "").lower() == "character"]
                if len(actors) != 1: raise ValueError("Expected exactly one VirtualHome character")
                roles = ACTION_ROLES.get(name, tuple(f"arg_{i}" for i in range(len(ids))))
                if len(roles) != len(ids): raise ValueError(f"Invalid arity: {name}")
                after = convert_state(states[index + 1])
                record = {"id": f"vh:{relative}:{index}", "before": canonical_before, "after": after,
                          "action": {"name": name, "actor": _ref(actors[0]),
                                     "arguments": [{"role": r, "node": _ref(n)} for r, n in zip(roles, ids)]},
                          "outcome": "success", "provenance": {"program_path": relative,
                              "state_path": relative.rsplit(".", 1)[0] + ".json", "step_index": index,
                              "action_line": line, "outcome_source": "official_executable_program_state_sequence"}}
                yield record
                canonical_before = after
    finally:
        if archive: archive.close()


def condition_predicate(condition, roles):
    """Translate measured factor definitions, never an action's required-condition list."""
    kind, value = condition["kind"], condition["value"]
    role = "$" + roles[condition.get("argument_index", 0)] if roles else "$actor"
    edge = lambda s, r, t: {"type": "edge", "source": s, "relation": r, "target": t}
    attr = lambda n, a, op, v: {"type": "node_attribute", "node": n, "attribute": a, "operator": op, "value": v}
    if kind == "property":
        return attr(role, PROPERTY_ATTRIBUTES.get(value, "affordance." + value.lower()), "equals", True)
    if kind == "property_any":
        return {"type": "any_of", "predicates": [condition_predicate({**condition, "kind": "property", "value": v}, roles) for v in value.split("|")]}
    if kind in {"state_present", "state_absent"}:
        name, val = STATE_ATTRIBUTES.get(value, ("state." + value.lower(), True))
        return attr(role, name, "equals" if kind == "state_present" else "not_equals", val)
    if kind in {"actor_state_absent_any", "actor_state_present_any"}:
        return {"type": "node_attribute", "node": "$actor", "attribute": "posture",
                "operator": "not_in" if kind == "actor_state_absent_any" else "in",
                "values": [STATE_ATTRIBUTES.get(v, ("state." + v.lower(), True))[1] for v in value.split("|")]}
    if kind == "close": return edge("$actor", "CLOSE", role)
    if kind == "worn_by_actor": return edge(role, "ON", "$actor")
    if kind == "holds": return {"type": "any_of", "predicates": [edge("$actor", h, role) for h in ("HOLDS_LH", "HOLDS_RH")]}
    if kind == "free_hand":
        return {"type": "derived_relation_condition", "operator": "not_all", "meaning": "at_least_one_free_hand",
                "predicates": [edge("$actor", h, "*") for h in ("HOLDS_LH", "HOLDS_RH")]}
    if kind == "not_inside_closed":
        return {"type": "not_exists", "variable": "container", "predicates": [edge(role,"INSIDE","$container"), attr("$container","is_open","equals",False)]}
    if kind == "putin_ready":
        return {"type": "any_of", "meaning": "not_openable_or_open", "predicates": [attr(role,"affordance.openable","not_equals",True),attr(role,"is_open","equals",True)]}
    if kind == "room_context": return {"type": "relation_path", "source": role, "relation": "INSIDE", "target_category": "Rooms", "min_hops": 0}
    if kind == "below_capacity": return {"type": "relation_cardinality", "relation": "ON", "target": role, "operator": "less_than_executor_limit", "limit_table": value.lower()}
    raise ValueError("Unsupported measured VirtualHome factor: " + kind)


def fixed_context_predicate(contexts, roles):
    """Translate recorded scope assumptions, not learned necessary conditions."""
    predicates = []
    for context in contexts:
        reason, satisfied = context.get("reason"), context.get("satisfied_by")
        if satisfied in {"class_exception", "target_is_room_entity"}:
            role = "destination" if context.get("name", "").startswith("destination_") else "source" if context.get("name", "").startswith("source_") else roles[0]
            predicates.append({"type": "node_type", "node": "$" + role, "value": context["class_name"].lower()})
        elif reason == "actor_postured":
            predicates.append({"type": "node_attribute", "node": "$actor", "attribute": "posture",
                               "operator": "in", "values": ["sitting", "lying"]})
        elif reason == "target_not_initially_close":
            predicates.append({"type": "not", "predicate": {"type": "edge", "source": "$actor", "relation": "CLOSE", "target": "$target"}})
        elif reason in {"target_is_body_part", "target_body_part"}:
            predicates.append({"type": "node_attribute", "node": "$target", "attribute": "semantic_type.body_part", "operator": "equals", "value": True})
        elif reason in {"target_is_worn", "target_worn_by_actor"}:
            predicates.append({"type": "edge", "source": "$target", "relation": "ON", "target": "$actor"})
        else:
            predicates.append({"type": "unresolved_context", "requirement": deepcopy(context)})
    return predicates[0] if len(predicates) == 1 else {"type": "all_of", "predicates": predicates}


def convert_mask_profile(raw):
    """Convert recorded interventions, including actually measured rather than requested masks."""
    name = raw["action_type"].upper()
    roles = ACTION_ROLES.get(name, ())
    return {"id": raw["sample_id"], "action": {"name": name, "argument_types": dict(zip(roles, raw["argument_classes"]))},
            "controlled": raw.get("base_replay_matches_official") is True,
            "source": deepcopy(raw.get("source", {})),
            "conditions": [{"name": c["name"], "predicate": condition_predicate(c, roles)} for c in raw["conditions"]],
            "fixed_context": deepcopy(raw.get("fixed_context_conditions", [])),
            "context_predicate": fixed_context_predicate(raw.get("fixed_context_conditions", []), roles),
            "trials": [{"id": str(i), "observed": m.get("observed_mask"), "success": m.get("success")}
                       for i, m in enumerate(raw["masks"])]}
