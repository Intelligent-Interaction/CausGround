"""Frozen source serialization and lexical primitives; standard library only."""
from __future__ import annotations
from collections import Counter
import math,json,re

STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "the", "this", "to", "use", "with",
}

ATTR_ALIASES = {
    "current_mode": "mode", "selected_mode": "mode", "washing_mode": "mode",
    "operation_state": "operation_status", "operational_status": "operation_status",
    "working_status": "operation_status", "copy_quantity": "copy_count",
    "quantity": "copy_count", "number_of_copies": "copy_count",
    "ratio": "copy_ratio", "paper_size": "paper_type",
}

def tokens(value: object) -> list[str]:
    result = re.findall(r"[a-z0-9]+", str(value or "").casefold())
    return [item for item in result if item not in STOP]

def norm(value: object) -> str:
    return "_".join(tokens(value))

def canon_attr(value: object) -> str:
    key = norm(value)
    if key.endswith("_status") and key not in {"operation_status"}:
        key = key[:-7]
    return ATTR_ALIASES.get(key, key)

def canon_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    key = norm(value)
    aliases = {
        "normal_operation": "operating_normally", "operate_normally": "operating_normally",
        "operational": "operating_normally", "1_1": "1", "one_to_one": "1",
        "enabled": "on", "activated": "on", "active": "on",
        "disabled": "off", "deactivated": "off", "inactive": "off",
    }
    return aliases.get(key, key)

def compatible_value(goal: object, source: object) -> bool:
    left, right = canon_value(goal), canon_value(source)
    if not left or not right or "unknown" in {left, right}:
        return False
    if left == right:
        return True
    if left == "non_zero":
        try:
            return float(str(source).replace(",", "")) != 0
        except ValueError:
            return right not in {"zero", "0", "none", "off"}
    return False

def compact(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)

def state_facts(state: dict) -> list[tuple[str, str, object]]:
    output = []
    for node in (state or {}).get("nodes", []) or []:
        node_id = norm(node.get("id") or node.get("name"))
        attrs = node.get("attributes") if isinstance(node.get("attributes"), dict) else {}
        for key, value in attrs.items():
            output.append((node_id, canon_attr(key), value))
    return output

def node_doc(action: dict, produced: list[tuple[str, str, object, str]]) -> str:
    # Node retrieval must be supported by the node itself. Repeating the record-level
    # requirement on every occurrence would make unrelated steps look relevant.
    parts = [action["operator"], action["owner_type"]]
    parts.extend(action["concepts"])
    parts.extend(action["directions"])
    for item in action["participants"]:
        parts.extend([item["name"], item["node"], item["node_type"], *item["aliases"]])
    for item in action["parameters"]:
        parts.extend([item.get("attribute", ""), item.get("value", "")])
    for node, attribute, value, _ in produced:
        parts.extend([node, attribute, value])
    return " ".join(str(x) for x in parts if str(x).strip())

class TfidfIndex:
    def __init__(self, docs: list[str]):
        counts = [Counter(tokens(doc)) for doc in docs]
        df = Counter()
        for doc in counts:
            df.update(doc)
        total = len(counts)
        self.idf = {term: math.log((1 + total) / (1 + count)) + 1 for term, count in df.items()}
        self.vectors = [self.vector(counter) for counter in counts]

    def vector(self, counts: Counter) -> dict[str, float]:
        vector = {term: count * self.idf.get(term, 0.0) for term, count in counts.items() if term in self.idf}
        length = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {term: value / length for term, value in vector.items()}

    def rank(self, query: str) -> list[tuple[float, int]]:
        query_vector = self.vector(Counter(tokens(query)))
        scored = []
        for index, vector in enumerate(self.vectors):
            score = sum(value * vector.get(term, 0.0) for term, value in query_vector.items())
            scored.append((score, index))
        return sorted(scored, key=lambda item: (-item[0], item[1]))

def symbolic_names(value: object) -> set[str]:
    """Return deterministic lexical forms; no learned or weighted similarity is used."""
    parts = tokens(value)
    if not parts:
        return set()
    return {"_".join(parts), "_".join(sorted(parts))}

def source_node_names(source_node: str, action: dict) -> set[str]:
    names = symbolic_names(source_node)
    for participant in action.get("participants", []):
        if norm(participant.get("node")) != norm(source_node):
            continue
        for value in [participant.get("node"), participant.get("name"), *participant.get("aliases", [])]:
            names.update(symbolic_names(value))
    return names

def render_action(
    index: int,
    row: dict,
    action: dict,
    produced: list[tuple[str, str, object, str]],
    reason: set[str],
) -> str:
    lines = [f"ACTION_NODE action_{index}", f"  retrieved_by={','.join(sorted(reason))}", f"  operator={action['operator']}"]
    if action["participants"]:
        lines.append("  object_nodes:")
        for item in action["participants"]:
            aliases = ",".join(sorted(set([item["name"], *item["aliases"]])))
            lines.append(f"    - role={item['role']} node={item['node']} type={item['node_type']} aliases=[{aliases}]")
    if action["parameters"]:
        lines.append("  parameters:")
        for item in action["parameters"]:
            lines.append(f"    - {canon_attr(item.get('attribute'))}={compact(item.get('value'))}")
    required = state_facts(row.get("current_state_t") or {})
    state_edges = (row.get("current_state_t") or {}).get("edges", [])[:12]
    if required or state_edges:
        lines.append("  requires:")
        for node, attribute, value in required[:16]:
            lines.append(f"    - {node}.{attribute}={compact(value)}")
    for edge in state_edges:
        source = norm(edge.get("source") or edge.get("from") or edge.get("from_id"))
        target = norm(edge.get("target") or edge.get("to") or edge.get("to_id"))
        relation = str(edge.get("relation") or edge.get("relation_type") or "RELATED_TO").upper()
        if source and target:
            lines.append(f"    - {source} --{relation}--> {target}")
    if produced:
        lines.append("  produces:")
        for node, attribute, value, basis in produced[:16]:
            lines.append(f"    - {node}.{attribute}={compact(value)} basis={basis}")
    if action["concepts"]:
        lines.append("  semantic_aliases=[" + ",".join(action["concepts"]) + "]")
    return "\n".join(lines)
