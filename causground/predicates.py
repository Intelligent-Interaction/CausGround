"""Reusable state predicates; no action-specific precondition table."""
from copy import deepcopy
import json

SYMMETRIC = {"CLOSE", "BETWEEN", "FAR", "TOUCHING", "PAIRED_WITH"}
HANDS = ("HOLDS_LH", "HOLDS_RH")


def token(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def canonical_edge(edge):
    result = {k: edge[k] for k in ("source", "relation", "target")}
    if result["relation"] in SYMMETRIC:
        result["source"], result["target"] = sorted((result["source"], result["target"]))
    return result


def export_predicate(predicate):
    p = deepcopy(predicate)
    kind = p.pop("kind", None)
    if kind == "attribute":
        p.update(type="node_attribute", operator=p.get("operator", "equals"))
    elif kind == "edge":
        p["type"] = "edge"
    return p


def index_state(state):
    return {**state, "_evaluation_index": ({n["id"]: n for n in state["nodes"]},
            {(e["source"], e["relation"], e["target"]) for e in state["edges"]})}


def evaluate(predicate, state):
    """Three-valued evaluation. Completeness declarations control negative facts."""
    if "_evaluation_index" not in state: state = index_state(state)
    nodes, edges = state["_evaluation_index"]
    complete = state.get("relations_complete", False)
    kind = predicate.get("type", predicate.get("kind"))
    if kind == "not":
        value = evaluate(predicate["predicate"], state)
        return None if value is None else not value
    if kind in {"node_type", "node_category"}:
        node = nodes.get(predicate["node"])
        field = "type" if kind == "node_type" else "category"
        return None if node is None or field not in node else node[field] == predicate["value"]
    if kind == "node_exists":
        return True if predicate["node"] in nodes else False if state.get("objects_complete", False) else None
    if kind == "relation_path":
        start = predicate["source"]
        if start not in nodes: return None
        frontier, seen, unknown = [(start, 0)], set(), False
        while frontier:
            node_id, depth = frontier.pop()
            if node_id in seen: continue
            seen.add(node_id)
            node = nodes.get(node_id, {})
            if depth >= predicate.get("min_hops", 0):
                if node.get("category") == predicate["target_category"]: return True
                if "category" not in node: unknown = True
            frontier.extend((t, depth + 1) for s, r, t in edges if s == node_id and r == predicate["relation"])
        return False if complete and state.get("objects_complete", False) and not unknown else None
    if kind == "relation_cardinality":
        limit = predicate.get("limit")
        if not isinstance(limit, int): return None
        count = len({s for s, r, t in edges if r == predicate["relation"] and t == predicate["target"]})
        return False if count >= limit else True if complete else None
    if kind in {"attribute", "node_attribute"}:
        node = nodes.get(predicate["node"])
        if node is None: return None
        attrs = node["attributes"]
        attr = predicate["attribute"]
        op = predicate.get("operator", "equals")
        if attr in attrs and attrs[attr] is None: return None
        if attr not in attrs:
            if not state.get("attributes_complete", False): return None
            # Closed-world absence is distinct from an explicit false value.
            return op in {"not_equals", "not_in"}
        equal = token(attrs[attr]) == token(predicate.get("value"))
        if op == "equals": return equal
        if op == "not_equals": return not equal
        if op in {"in", "not_in"}:
            found = any(token(attrs[attr]) == token(v) for v in predicate["values"])
            return found if op == "in" else not found
        raise ValueError("Unsupported attribute operator: " + op)
    if kind == "edge":
        source, relation, target = [predicate[k] for k in ("source", "relation", "target")]
        present = any((source == "*" or s == source) and r == relation and (target == "*" or t == target)
                      for s, r, t in edges)
        if not present and relation in SYMMETRIC:
            present = any((source == "*" or t == source) and r == relation and (target == "*" or s == target)
                          for s, r, t in edges)
        return True if present else False if complete else None
    if kind in {"any_of", "all_of", "derived_relation_condition"}:
        values = [evaluate(p, state) for p in predicate["predicates"]]
        if kind == "any_of": return True if True in values else None if None in values else False
        conjunction = False if False in values else None if None in values else True
        if kind == "derived_relation_condition":
            if predicate.get("operator") != "not_all": raise ValueError("Unsupported compound operator")
            return None if conjunction is None else not conjunction
        return conjunction
    if kind == "not_exists":
        variable = "$" + predicate["variable"]
        checks = []
        relation_filter = next((p for p in predicate["predicates"] if p.get("type") == "edge" and p.get("target") == variable), None)
        candidate_ids = ({t for s, r, t in edges if s == relation_filter["source"] and r == relation_filter["relation"]}
                         if relation_filter else set(nodes))
        for node_id in candidate_ids:
            def bind(value):
                if isinstance(value, dict): return {k: bind(v) for k, v in value.items()}
                if isinstance(value, list): return [bind(v) for v in value]
                return node_id if value == variable else value
            checks.append(evaluate({"type": "all_of", "predicates": bind(predicate["predicates"])}, state))
        if True in checks: return False
        return True if complete and state.get("objects_complete", False) and None not in checks else None
    return None


def instantiate(predicate, bindings):
    inverse = {role: node for node, role in bindings.items()}
    def walk(value):
        if isinstance(value, dict): return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list): return [walk(v) for v in value]
        return inverse.get(value, value) if isinstance(value, str) else value
    return walk(predicate)


def compound_candidates(bindings):
    """Generate relation compositions for every argument, independent of action name."""
    actor = "$actor"
    edge = lambda source, relation, target: {"type": "edge", "source": source, "relation": relation, "target": target}
    result = [{"type": "derived_relation_condition", "operator": "not_all", "meaning": "at_least_one_free_hand",
               "predicates": [edge(actor, hand, "*") for hand in HANDS]}]
    result.extend(edge(actor, hand, "*") for hand in HANDS)
    for role in sorted(set(bindings.values()) - {actor}):
        result.append({"type": "any_of", "predicates": [edge(actor, hand, role) for hand in HANDS]})
        result.append({"type": "not_exists", "variable": "container", "predicates": [
            edge(role, "INSIDE", "$container"),
            {"type": "node_attribute", "node": "$container", "attribute": "is_open", "operator": "equals", "value": False}]})
        result.append({"type": "any_of", "meaning": "not_openable_or_open", "predicates": [
            {"type": "node_attribute", "node": role, "attribute": "affordance.openable", "operator": "not_equals", "value": True},
            {"type": "node_attribute", "node": role, "attribute": "is_open", "operator": "equals", "value": True}]})
    return result


def observed_effect_confirmed(effect, before, after):
    op = effect["operation"]
    if op == "update_attribute": return effect.get("before") is not None and effect.get("after") is not None
    if op == "set_attribute": return before.get("attributes_complete", False) and effect.get("after") is not None
    if op == "unset_attribute": return after.get("attributes_complete", False) and effect.get("before") is not None
    if op == "add_edge": return before.get("relations_complete", False)
    if op == "remove_edge": return after.get("relations_complete", False)
    if op == "add_node": return before.get("objects_complete", False)
    if op == "remove_node": return after.get("objects_complete", False)
    return op == "update_type"
