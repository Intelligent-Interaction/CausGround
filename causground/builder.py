"""Environment-independent, conservative empirical action hyperedges.

This example proposes conditions; it does not establish causal necessity.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
import zlib

from .predicates import (canonical_edge, compound_candidates, evaluate, export_predicate,
                         instantiate, observed_effect_confirmed, index_state)


def signature(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _state(value, label):
    if not isinstance(value, dict) or not isinstance(value.get("nodes"), list) or not isinstance(value.get("edges"), list):
        raise ValueError(f"{label} requires nodes and edges lists")
    ids = set()
    for node in value["nodes"]:
        node_id = _text(node.get("id"), f"{label}.node.id")
        if node_id in ids:
            raise ValueError(f"{label}: duplicate node {node_id}")
        ids.add(node_id)
        _text(node.get("type"), f"{label}.node.type")
        if not isinstance(node.get("attributes"), dict):
            raise ValueError(f"{label}.{node_id}: attributes must be an object")
    edges = set()
    for edge in value["edges"]:
        _text(edge.get("relation"), f"{label}.edge.relation")
        if edge.get("source") not in ids or edge.get("target") not in ids:
            raise ValueError(f"{label}: edge references unknown node")
        key = (edge["source"], edge["relation"], edge["target"])
        if key in edges:
            raise ValueError(f"{label}: duplicate edge {key}")
        edges.add(key)
    for flag in ("relations_complete", "attributes_complete", "objects_complete"):
        if not isinstance(value.get(flag, False), bool):
            raise ValueError(f"{label}.{flag} must be boolean")
    return ids


def validate_transition(record):
    if not isinstance(record, dict):
        raise ValueError("transition must be an object")
    _text(record.get("id"), "transition.id")
    before_ids = _state(record.get("before"), "before")
    _state(record.get("after"), "after")
    action = record.get("action")
    if not isinstance(action, dict):
        raise ValueError("action must be an object")
    _text(action.get("name"), "action.name")
    if action.get("actor") not in before_ids:
        raise ValueError("action actor references unknown node")
    if not isinstance(action.get("arguments"), list):
        raise ValueError("action.arguments must be a list")
    roles, nodes = {"actor"}, {action["actor"]}
    for arg in action["arguments"]:
        role = _text(arg.get("role"), "argument.role")
        if role.startswith("$") or role in roles:
            raise ValueError(f"duplicate/reserved argument role: {role}")
        if arg.get("node") not in before_ids:
            raise ValueError("action argument references unknown node")
        roles.add(role)
        nodes.add(arg["node"])
    if record.get("outcome") not in ("success", "failure", "unknown"):
        raise ValueError("outcome must be success, failure, or unknown")
    return signature(record)  # Reject non-JSON and non-finite values, too.


def observed_changes(before, after):
    """Full symbolic observation delta; unknown/unset is not assumed false."""
    old = {n["id"]: n for n in before["nodes"]}
    new = {n["id"]: n for n in after["nodes"]}
    changes = []
    for node_id in sorted(old.keys() | new.keys()):
        if node_id not in old or node_id not in new:
            changes.append({"operation": "add_node" if node_id in new else "remove_node",
                            "node": node_id, "value": deepcopy(new.get(node_id, old.get(node_id)))})
            continue
        if old[node_id]["type"] != new[node_id]["type"]:
            changes.append({"operation": "update_type", "node": node_id,
                            "before": old[node_id]["type"], "after": new[node_id]["type"]})
        left, right = old[node_id]["attributes"], new[node_id]["attributes"]
        for attr in sorted(left.keys() | right.keys()):
            if attr in left and attr in right and signature(left[attr]) == signature(right[attr]):
                continue
            item = {"operation": "update_attribute" if attr in left and attr in right else
                    "set_attribute" if attr in right else "unset_attribute", "node": node_id, "attribute": attr}
            if attr in left: item["before"] = deepcopy(left[attr])
            if attr in right: item["after"] = deepcopy(right[attr])
            changes.append(item)
    edges = lambda s: {(e["source"], e["relation"], e["target"]) for e in map(canonical_edge, s["edges"])}
    left, right = edges(before), edges(after)
    for operation, items in (("remove_edge", left - right), ("add_edge", right - left)):
        for source, relation, target in sorted(items):
            changes.append({"operation": operation, "edge": {"source": source, "relation": relation, "target": target}})
    return sorted(changes, key=signature)


def _bindings(record):
    result = {record["action"]["actor"]: "$actor"}
    for a in sorted(record["action"]["arguments"], key=lambda a: a["role"]): result.setdefault(a["node"], "$"+a["role"])
    return result


def _role_nodes(record):
    return {"actor": record["action"]["actor"], **{a["role"]: a["node"] for a in record["action"]["arguments"]}}


def _aliases(record):
    bindings = _bindings(record)
    return {"$"+role: bindings[node] for role, node in _role_nodes(record).items() if "$"+role != bindings[node]}


def _local_facts(record):
    bindings = _bindings(record)
    facts = {}
    for node in record["before"]["nodes"]:
        if node["id"] not in bindings:
            continue
        for attr, value in node["attributes"].items():
            if value is not None:
                p = {"kind": "attribute", "node": bindings[node["id"]], "attribute": attr, "value": deepcopy(value)}
                facts[signature(p)] = p
    for edge in record["before"]["edges"]:
        if edge["source"] in bindings and edge["target"] in bindings:
            p = {"kind": "edge", **canonical_edge({"source": bindings[edge["source"]], "relation": edge["relation"], "target": bindings[edge["target"]]})}
            facts[signature(p)] = p
    facts.update(record.get("_extra_predicates", {}))
    return facts


def _truth(predicate, record):
    key = signature(predicate)
    if key in record.get("_extra_truths", {}):
        value = record["_extra_truths"][key]
    else:
        value = evaluate(instantiate(predicate, _bindings(record)), record["before"])
    return "unknown" if value is None else "true" if value else "false"


def _local_effects(record):
    if "_local_effects" in record:
        return record["_local_effects"], record["_has_context"]
    bindings = _bindings(record)
    local, contextual = [], False
    for effect in record["changes"]:
        mapped = deepcopy(effect)
        if "edge" in effect and all(effect["edge"][k] in bindings for k in ("source", "target")):
            for k in ("source", "target"):
                mapped["edge"][k] = bindings[effect["edge"][k]]
            mapped["edge"] = canonical_edge(mapped["edge"])
        elif effect.get("node") in bindings and effect["operation"] == "remove_node":
            mapped = {"operation": "remove_node", "node": bindings[effect["node"]]}
        elif effect.get("node") in bindings and effect["operation"] != "add_node":
            mapped["node"] = bindings[effect["node"]]
        else:
            contextual = True
            continue
        local.append(mapped)
    return [e for _, e in sorted({signature(e): e for e in local}.items())], contextual


class CausGroundBuilder:
    """Append evidence, then snapshot after one or many additions.

    Snapshot aggregation is deliberately simple and retains evidence in memory.
    Conditions describe observed consistency, not validated executability.
    """

    def __init__(self, min_support=1, effect_policy="observed"):
        if type(min_support) is not int or min_support < 1:
            raise ValueError("min_support must be a positive integer")
        self.min_support = min_support
        if effect_policy not in {"observed", "virtualhome"}: raise ValueError("Unsupported effect policy")
        self.effect_policy = effect_policy
        self._records = {}
        self._sources = {}

    def add(self, record):
        value = json.loads(validate_transition(record))
        value.pop("changes", None)
        value.pop("schema_version", None)
        value.pop("head", None)
        value.pop("effect_bindings", None)
        value["before"]["nodes"].sort(key=lambda n: n["id"])
        value["after"]["nodes"].sort(key=lambda n: n["id"])
        for side in ("before", "after"):
            value[side]["edges"].sort(key=lambda e: (e["source"], e["relation"], e["target"]))
        value["action"]["arguments"].sort(key=lambda a: a["role"])
        # Keep exact normalized input for repeated IDs without retaining full scenes in memory.
        source = zlib.compress(signature(value).encode("utf-8"), level=1)
        key = value["id"]
        if key in self._sources and self._sources[key] != source:
            raise ValueError(f"conflicting evidence for id {key}")
        value["schema_version"] = "causground.transition.v3"
        value["changes"] = observed_changes(value["before"], value["after"])
        from .effects import classify_effects, lift_effects, post_candidates
        value["head"] = classify_effects(value, self.effect_policy)
        bindings = _bindings(value)
        extra = compound_candidates(bindings)
        compact = {k: deepcopy(value[k]) for k in ("id", "action", "outcome")}
        compact["before"] = {**{k: v for k, v in value["before"].items() if k not in {"nodes", "edges"}},
                             "nodes": [n for n in value["before"]["nodes"] if n["id"] in bindings],
                             "edges": [e for e in value["before"]["edges"] if e["source"] in bindings and e["target"] in bindings]}
        compact["after"] = {**{k: v for k, v in value["after"].items() if k not in {"nodes", "edges"}},
                            "nodes": [n for n in value["after"]["nodes"] if n["id"] in bindings],
                            "edges": [e for e in value["after"]["edges"] if e["source"] in bindings and e["target"] in bindings]}
        compact["_bindings"] = bindings
        compact["_effect_variant"], value["effect_bindings"] = lift_effects(value, bindings)
        compact["_post_candidates"] = post_candidates(value, bindings)
        compact["_extra_predicates"] = {signature(p): p for p in extra}
        indexed = index_state(value["before"])
        compact["_extra_truths"] = {signature(p): evaluate(instantiate(p, bindings), indexed) for p in extra}
        local_core = {**value, "changes": value["head"]["core_effects"]}
        compact["_local_effects"], compact["_has_context"] = _local_effects(local_core)
        compact["_confirmed_effects"] = set()
        for effect in value["head"]["core_effects"]:
            if observed_effect_confirmed(effect, value["before"], value["after"]):
                mapped, _ = _local_effects({**value, "changes": [effect]})
                compact["_confirmed_effects"].update(signature(e) for e in mapped)
        compact["_confirmed_effects"] = sorted(compact["_confirmed_effects"])
        self._records[key] = deepcopy(compact)
        self._sources[key] = source
        return value

    def _rule(self, number, records, scope):
        predicates = {}
        for record in records:
            predicates.update(_local_facts(record))
        candidates, rejected, required, pending = [], [], [], []
        successes = sum(r["outcome"] == "success" for r in records)
        for token, predicate in sorted(predicates.items()):
            counts = {outcome: {truth: 0 for truth in ("true", "false", "unknown")}
                      for outcome in ("success", "failure", "unknown")}
            for record in records:
                counts[record["outcome"]][_truth(predicate, record)] += 1
            item = {"predicate": predicate, "counts": counts}
            if counts["success"]["false"]:
                rejected.append(item)
            elif counts["success"]["true"]:
                candidates.append(item)
                accepted = counts["success"]["true"] == successes and successes >= self.min_support
                public = {"predicate": export_predicate(predicate), "counts": counts,
                          "basis": "success_consistency", "necessity_verified": False}
                (required if accepted else pending).append(public)
        evidence = {o: sorted(r["id"] for r in records if r["outcome"] == o)
                    for o in ("success", "failure", "unknown")}
        contextual = []
        for record in records:
            _, has_context = _local_effects(record)
            if has_context: contextual.append(record["id"])
        effect_evidence = {}
        for record in records:
            if record["outcome"] != "success": continue
            effects, _ = _local_effects(record)
            confirmed = set(record["_confirmed_effects"])
            for effect in effects:
                token = signature(effect)
                item = effect_evidence.setdefault(token, {"effect": effect, "evidence_ids": [], "unconfirmed_ids": []})
                item["evidence_ids" if token in confirmed else "unconfirmed_ids"].append(record["id"])
        core, context = [], []
        for token, item in sorted(effect_evidence.items()):
            support = len(item["evidence_ids"])
            if support == successes and successes >= self.min_support:
                core.append(deepcopy(item["effect"]))
            else:
                context.append({**item, "evidence_ids": sorted(item["evidence_ids"]), "unconfirmed_ids": sorted(item["unconfirmed_ids"]),
                                "support": support, "successful_executions": successes,
                                "reason": "partial_observation_or_insufficient_support" if item["unconfirmed_ids"] or successes < self.min_support else "not_consistent_across_successes"})
        coverage = defaultdict(set)
        for record in records:
            nodes = {n["id"]: n for n in record["before"]["nodes"]}
            for role, node_id in _role_nodes(record).items():
                coverage[role].add(nodes[node_id]["type"])
        rule = {
            "id": f"{scope}:{number:04d}",
            "scope": scope, "generalization_status": "empirical_observed_types_only",
            "action": {"name": records[0]["action"]["name"], "actor": "$actor",
                       "arguments": [{"role": a["role"], "node": "$" + a["role"]} for a in records[0]["action"]["arguments"]]},
            "observed_argument_types": {k: sorted(v) for k, v in sorted(coverage.items())},
            "tail": {"status": "observational_candidates_only", "condition_schema_status": "success_consistency_filtered",
                     "required_conditions": [], "observational_conditions": required, "pending_conditions": pending,
                     "condition_branches": [],
                     "candidate_conditions": candidates, "rejected_candidates": rejected},
            "head": {"common_observed_changes": core, "variable_observed_changes": context,
                     "effect_basis": "observed_variants_and_postconditions", "successful_executions": successes,
                     "semantic_policy": self.effect_policy, "contextual_change_evidence": sorted(contextual)},
            "evidence": evidence,
        }
        if scope == "specific":
            rule["type_constraints"] = {role: next(iter(types)) for role, types in sorted(coverage.items())}
        if _aliases(records[0]): rule["action"]["role_aliases"] = _aliases(records[0])
        from .effects import summarize
        rule["head"].update(summarize(records, predicates, _truth, self.min_support, rule["id"]))
        return rule

    def snapshot(self):
        generic, specific = defaultdict(list), defaultdict(list)
        for _, record in sorted(self._records.items()):
            action = record["action"]
            base = {"name": action["name"], "roles": [a["role"] for a in action["arguments"]]}
            if _aliases(record): base["role_aliases"] = _aliases(record)
            generic[signature(base)].append(record)
            nodes = {n["id"]: n for n in record["before"]["nodes"]}
            types = {role: nodes[node]["type"] for role, node in _role_nodes(record).items()}
            specific[signature({**base, "types": types})].append(record)
        return {
            "schema_version": "causground.offline.v3",
            "representation": "condition_action_effect_hyperedges",
            "evidence_count": len(self._records),
            "condition_semantics": "Required_conditions are supplied controlled-evidence conclusions. Success consistency remains observational_conditions. Apply condition_branches only within their guards. Empty required_conditions do not mean unconditional executability.",
            "effect_semantics": "Per-execution head preserves classified raw effects. Aggregates preserve context-bound variants, empirical pre-state branches and stable postconditions; common_observed_changes is only a descriptive delta intersection. No branch is certified on unseen data.",
            "candidate_scope": "Actor/argument attributes, their relations and reusable holding/container predicates. Non-argument effect endpoints are context variables; ambiguous and unresolved bindings remain explicit.",
            "effect_policy": self.effect_policy,
            "min_support": self.min_support,
            "generic_rules": [self._rule(i, rs, "generic") for i, (_, rs) in enumerate(sorted(generic.items()), 1)],
            "specific_rules": [self._rule(i, rs, "specific") for i, (_, rs) in enumerate(sorted(specific.items()), 1)],
        }
