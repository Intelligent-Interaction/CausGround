"""Strict source grammar. Opaque strings and LF boundaries are preserved."""
import re

HEADER = "TASK_CONDITIONED_CWM"

BASES = (
    "supported_effect", "derived_numeric_direction", "typed_option_effect",
    "post_state_confirmation", "explicit_source_requirement",
    "typed_action_parameter",
)

_ID = r"[A-Za-z0-9_]+"

_ATTR = r"[A-Za-z0-9_]*"

_FACT = re.compile(rf"({_ID})\.({_ATTR})=(.*)")

_EDGE = re.compile(rf"({_ID}) --([A-Z0-9_]+)--> ({_ID})")

_PARAMETER = re.compile(rf"({_ID})=(.*)")

_OBJECT = re.compile(
    rf"role=({_ID}) node=({_ID}) type=({_ID}) aliases=\[([^\r\n]*)\]"
)

class CWMParseError(ValueError):
    """Input is outside the explicitly supported complete CWM grammar."""

def _error(line, message):
    raise CWMParseError("line {}: {}".format(line, message))

def _match(pattern, value, line, kind):
    match = re.fullmatch(pattern, value)
    if match is None:
        _error(line, "invalid {}: {!r}".format(kind, value))
    return match

def _fact(value, line, allow_edge=False):
    match = _FACT.fullmatch(value)
    if match:
        return ["attribute", *match.groups()]
    match = _EDGE.fullmatch(value) if allow_edge else None
    if match:
        return ["edge", *match.groups()]
    _error(line, "invalid fact: {!r}".format(value))

def _take_omission_marker(lines):
    marker = None
    if lines and re.fullmatch(r"\[CWM_NODES_OMITTED_AT_FIXED_LIMIT=[1-9][0-9]*\]", lines[-1][1]):
        _, marker = lines.pop()
    return marker

def _parse_original(text, *, allow_empty=False):
    if not isinstance(text, str):
        raise TypeError("CWM input must be str")
    # Only CRLF and LF are supported. Do not split embedded Unicode separators.
    text = text.replace("\r\n", "\n")
    if "\r" in text or "\x00" in text:
        _error(1, "unsupported control character")
    lines = [(i, line) for i, line in enumerate(text.split("\n"), 1) if line != ""]
    omission_marker = _take_omission_marker(lines)
    pos = 0

    def take():
        nonlocal pos
        if pos == len(lines):
            _error(len(text.split("\n")), "unexpected end of context")
        item = lines[pos]
        pos += 1
        return item

    def expect(expected):
        number, line = take()
        if line != expected:
            _error(number, "expected {!r}, got {!r}".format(expected, line))

    expect(HEADER)
    number, line = take()
    retrieval = _match(r"retrieval=(.+)", line, number, "retrieval header").group(1)
    expect("raw_source_text_exposed=false")
    actions, reasons = [], []
    while pos < len(lines):
        number, line = take()
        action_id = _match(r"ACTION_NODE (action_[0-9]+)", line, number, "action header").group(1)
        number, line = take()
        reason = _match(rf"  retrieved_by=({_ID}(?:,{_ID})*)", line, number, "retrieved_by").group(1)
        reasons.append({"action_index": len(actions), "action_id": action_id, "retrieved_by": reason})
        number, line = take()
        operator = _match(rf"  operator=({_ID})", line, number, "operator").group(1)
        action = {"action_id": action_id, "operator": operator, "sections": [],
                  "object_nodes": [], "parameters": [], "requires": [],
                  "produces": [], "semantic_aliases": None}
        last_rank = 0
        while pos < len(lines) and not lines[pos][1].startswith("ACTION_NODE "):
            number, line = take()
            indent = "  "
            semantic = re.fullmatch(re.escape(indent) + r"semantic_aliases=\[([^\r\n]*)\]", line)
            if semantic:
                if last_rank >= 5:
                    _error(number, "duplicate/out-of-order semantic_aliases")
                action["semantic_aliases"] = semantic.group(1)
                action["sections"].append("semantic_aliases")
                last_rank = 5
                continue
            section = _match(re.escape(indent) + r"(object_nodes|parameters|requires|produces):", line, number, "section").group(1)
            rank = {"object_nodes": 1, "parameters": 2, "requires": 3, "produces": 4}[section]
            if rank <= last_rank:
                _error(number, "duplicate/out-of-order section")
            action["sections"].append(section)
            last_rank = rank
            prefix = "    - "
            count = 0
            while pos < len(lines) and lines[pos][1].startswith(prefix):
                number, line = take()
                value = line[len(prefix):]
                if section == "object_nodes":
                    action[section].append(list(_match(_OBJECT, value, number, "object").groups()))
                elif section == "parameters":
                    action[section].append(list(_match(_PARAMETER, value, number, "parameter").groups()))
                elif section == "requires":
                    action[section].append(_fact(value, number, allow_edge=True))
                else:
                    # rsplit retains any literal ' basis=' inside the value.
                    if " basis=" not in value:
                        _error(number, "produced fact has no basis")
                    fact_value, fact_basis = value.rsplit(" basis=", 1)
                    if fact_basis not in BASES:
                        _error(number, "unknown effect basis {!r}".format(fact_basis))
                    action[section].append([*_fact(fact_value, number), fact_basis])
                count += 1
            # Original render_action may emit an empty requires section for
            # malformed source edges. Preserve it without inventing any edge.
            if count == 0 and section != "requires":
                _error(number, "empty {} section".format(section))
        actions.append(action)
    if not actions and not allow_empty:
        _error(1, "context has no selected actions")
    return {"header": HEADER, "raw_source_text_exposed": "false", "actions": actions,
            "omission_marker": omission_marker}, {
        "retrieval": retrieval, "actions": reasons,
    }
