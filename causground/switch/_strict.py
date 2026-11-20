"""The frozen strict text primitives, packaged without dynamic imports."""
import json,re
from . import _parser as parser

HEADER = (
    "Observed preconditions and effects are summarized from interaction trajectories.\n"
    "When only a few trajectories are available for an action, these summaries may be incomplete or inaccurate.\n"
    "Use them as reference knowledge, not absolute rules. Check the current observation; missing facts are unknown."
)

def _quoted(value):
    return json.dumps(value, ensure_ascii=False)

def _words(value):
    return value.replace("_", " ")

def _name(node):
    # Neither control labels nor aliases are identities, even when unique.
    return "unresolved object " + node if re.fullmatch(r"e\d+", node) else _words(node)

def _signature(action):
    args = []
    for role, node, typ, _aliases in action["object_nodes"]:
        name = _name(node)
        if re.fullmatch(r"e\d+", node):
            name += " (" + _words(typ) + ")"
        args.append(_words(role) + "=" + name)
    args.extend(_words(key) + "=" + _quoted(value) for key, value in action["parameters"])
    operator = action["operator"]
    return operator + "(" + ", ".join(args) + ")"

def _fact_text(fact):
    if fact[0] == "edge":
        relation = "--" + fact[2] + "-->"
        return _name(fact[1]) + " " + relation + " " + _name(fact[3])
    attribute = _words(fact[2]) if fact[2] else "[unspecified attribute]"
    return _name(fact[1]) + "." + attribute + " = " + _quoted(fact[3])

def _context(rules):
    lines = [HEADER]
    for rule in rules:
        lines.extend(["", "Action: " + rule["action"]])
        lines.extend("  " + line for line in rule["lines"])
    if not rules:
        lines.extend(["", "Action: no rules selected",
                      "  observed preconditions: not specified", "  effects: not specified"])
    return "\n".join(lines) + "\n"


def render(task, original):
    """Render all preconditions and supported effects; retain all facts in audit."""
    if not isinstance(task, str) or not task.strip():
        raise ValueError('task must be nonempty text')
    inventory, metadata = parser._parse_original(original, allow_empty=True)
    actions = inventory['actions']
    selected = [i for i, a in enumerate(actions) if a['requires'] or
                any(f[-1] == 'supported_effect' for f in a['produces'])]
    records, rules = [], []
    for i, action in enumerate(actions):
        body = []
        for field, label in [('requires', 'observed preconditions'), ('produces', 'effects')]:
            visible = []
            for j, fact in enumerate(action[field]):
                allowed = field == 'requires' or fact[-1] == 'supported_effect'
                shown = i in selected and allowed
                text = _fact_text(fact) if shown else None
                records.append({'source_index': i, 'action_id': action['action_id'],
                                'field': field, 'fact_index': j, 'fact': fact,
                                'visibility': 'context' if shown else 'audit_only',
                                'reason': 'selected_strict_fact' if shown else
                                ('action_not_selected' if i not in selected else 'basis_excluded_by_strict_policy'),
                                'label': label, 'text': text})
                if shown:
                    visible.append(text)
            body.append(label + ': ' + ('; '.join(visible) if visible else 'not specified'))
        if i in selected:
            rules.append({'source_index': i, 'source_action_id': action['action_id'],
                          'action': _signature(action), 'lines': body})
    text = _context(rules)
    audit = {'schema': 'switch-strict-audit-v1', 'variant': 'strict_no_empty', 'task': task,
             'original_context': original, 'inventory': inventory, 'retrieval_metadata': metadata,
             'selected_source_indices': selected, 'rendered_rules': rules,
             'fact_renderings': records}
    return text, audit
