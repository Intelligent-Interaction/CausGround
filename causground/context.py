"""Compact planning context; evidence and diagnostics remain in the subgraph JSON."""
import json
from collections import Counter, defaultdict

from .knowledge import effect_predicate


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def unique(items):
    return list(dict.fromkeys(items))


def predicate_text(p):
    kind = p.get("type", p.get("kind"))
    if kind in {"all_of", "any_of", "derived_relation_condition"}:
        parts = unique(predicate_text(x) for x in p["predicates"])
        joiner = " OR " if kind == "any_of" else " AND "
        text = "(" + joiner.join(parts) + ")" if len(parts) > 1 else parts[0] if parts else ("FALSE" if kind == "any_of" else "TRUE")
        return "NOT " + text if kind == "derived_relation_condition" else text
    if kind == "not":
        return "NOT " + predicate_text(p["predicate"])
    if kind == "edge":
        return f"{p['source']} {p['relation']} {p['target']}"
    if kind in {"node_attribute", "attribute"}:
        op = {"equals": "=", "not_equals": "!=", "not_in": "not in"}.get(p.get("operator", "equals"), p["operator"] if "operator" in p else "=")
        return f"{p['node']}.{p['attribute']} {op} {compact(p.get('value', p.get('values')))}"
    if kind in {"node_type", "node_category"}:
        return f"{p['node']}.{kind[5:]} = {compact(p['value'])}"
    if kind == "node_exists":
        return f"{p['node']} exists"
    if kind == "relation_path":
        return f"{p['source']} reaches {p['target_category']} via {p['relation']} (at least {p.get('min_hops', 0)} hops)"
    if kind == "relation_cardinality":
        limit = p.get("limit")
        if not isinstance(limit, int):
            limit = "unknown limit" + (" (executor " + str(p["limit_table"]) + ")" if p.get("limit_table") else "")
        return f"count(* {p['relation']} {p['target']}) < {limit}"
    if kind == "not_exists":
        return f"NOT EXISTS ${p['variable']}: " + predicate_text({"type": "all_of", "predicates": p["predicates"]})
    return compact(p)


def action_text(action):
    args = ", ".join(str(a.get("node", a.get("value"))) for a in action["arguments"])
    actor = action.get("actor", "agent")
    return (str(actor) + ": " if actor != "agent" else "") + action["name"] + "(" + args + ")"


def guard_text(p):
    text = predicate_text(p or {"type": "all_of", "predicates": []})
    return "" if text == "TRUE" else text


def conjunction_terms(p):
    """Recognize redundant success formulas without changing OR semantics."""
    kind = p.get("type", p.get("kind"))
    children = p.get("predicates", [])
    if kind == "all_of" or (kind == "any_of" and len(children) == 1):
        return {term for child in children for term in conjunction_terms(child)}
    return {predicate_text(p)}


def binding_text(constraints):
    descriptions = []
    for role, spec in sorted(constraints.items()):
        parts = []
        for item in [spec, *spec.get("constraints", [])]:
            for key in ("kind", "class", "type", "category", "group", "component", "quantifier", "destination_context"):
                if key in item:
                    parts.append(key.replace("_", " ") + "=" + str(item[key]))
            if item.get("binding_policy"):
                parts.append(item["binding_policy"].replace("_", " "))
            if item.get("binding_source") == "query_position_or_same_burner_as_placed_cookware":
                parts.append("use requested position or the burner holding the cookware")
        descriptions.append(role + ": " + ", ".join(unique(parts)))
    return "; ".join(descriptions)


def rule_lines(m):
    lines = []
    requirements = sorted(set(predicate_text(c["predicate"]) for c in m.get("conditions", [])))
    if requirements:
        lines.append("requires: " + "; ".join(requirements))
    known_terms = {t for c in m.get("conditions", []) for t in conjunction_terms(c["predicate"])}
    for b in m.get("condition_branches", []):
        guard = guard_text(b.get("when"))
        required = sorted(set(predicate_text(c["predicate"]) for c in b.get("required_conditions", [])) - set(requirements))
        prefix = "WHEN " + guard + ": " if guard else ""
        if required:
            lines.append(prefix + "requires: " + "; ".join(required))
        formula = (b.get("success_formula") or {}).get("predicate")
        branch_terms = {t for c in b.get("required_conditions", []) for t in conjunction_terms(c["predicate"])}
        if formula and not conjunction_terms(formula).issubset(known_terms | branch_terms):
            lines.append(prefix + "measured success condition: " + predicate_text(formula))

    if not lines and m.get("condition_status") == "observational_candidates_only":
        lines.append("requires: not established")

    observed = m.get("source_format") == "builder"
    label = "observed effects" if observed else "expected effects"
    variants = {v["id"]: v for v in m.get("effect_variants", [])}

    def outcome(v):
        effects = []
        for i, fx in enumerate(v.get("effects", [])):
            p = effect_predicate(fx)
            if p is None:
                if fx.get("operation") == "unset_attribute":
                    text = f"{fx['node']}.{fx['attribute']} becomes unspecified"
                elif fx.get("operation") == "update_type":
                    text = f"{fx['node']}.type = {compact(fx.get('after'))}"
                else:
                    text = "unspecified effect"
            else:
                text = predicate_text(p)
            if i in v.get("unconfirmed_effects", []):
                text += " (unconfirmed)"
            effects.append(text)
        result = "; ".join(sorted(set(effects))) if effects else "no change"
        if v.get("context_bindings"):
            result += " (context objects must be resolved)"
        return result

    covered = set()
    unconditional = set()
    for b in m.get("effect_branches", []):
        ids = [a["variant_id"] for a in b["alternatives"]]
        alternatives = unique(outcome(variants[i]) if i in variants else "unspecified effect" for i in ids)
        covered.update(ids)
        guard = guard_text(b.get("when"))
        text = " OR ".join("(" + a + ")" for a in alternatives) if len(alternatives) > 1 else alternatives[0] if alternatives else "unspecified effect"
        lines.append(label + (" WHEN " + guard if guard else "") + ": " + text)
        if not guard and len(ids) == 1 and ids[0] in variants:
            unconditional.update(outcome(variants[ids[0]]).split("; "))
    remaining = [v for k, v in variants.items() if k not in covered]
    if remaining:
        alternatives = unique(outcome(v) for v in remaining)
        lines.append(label + ": " + (" OR ".join("(" + a + ")" for a in alternatives) if len(alternatives) > 1 else alternatives[0]))
        if not covered and len(remaining) == 1:
            unconditional.update(outcome(remaining[0]).split("; "))
    posts = defaultdict(list)
    for c in m.get("postconditions", []):
        text = predicate_text(c["predicate"])
        guard = guard_text(c.get("when"))
        if text not in unconditional or guard:
            posts[guard].append(text)
    for guard, values in posts.items():
        lines.append(("observed after" if observed else "expected after") + (" WHEN " + guard if guard else "") + ": " + "; ".join(sorted(set(values))))
    if not variants and not posts:
        lines.append("effects: unspecified")
    # Concrete query entities already resolve role-type annotations. Keep constraints
    # on placeholders, which the downstream planner still needs to bind.
    arguments = {"$" + a["role"].lstrip("$"): a.get("node", a.get("value")) for a in m["action"]["arguments"]}
    constraints = {role: spec for role, spec in m.get("role_type_constraints", {}).items()
                   if str(arguments.get(role, role)).startswith("$")}
    if constraints:
        lines.append("bind: " + binding_text(constraints))
    return unique(lines)


def render_context(graph):
    private = graph.get("context_policy") == "mechanisms_only"
    spec = graph.get("goal_spec", {})
    mechanisms = graph.get("mechanisms", [])
    multi_phase = len({m.get("phase", 0) for m in mechanisms}) > 1

    # Merge equivalent text rules, never different requirements, actors or phases.
    groups = {}
    for m in mechanisms:
        body = rule_lines(m)
        key = (action_text(m["action"]), m.get("phase", 0), m.get("candidate_only", False), tuple(body))
        groups.setdefault(key, []).append(m["id"])
    totals = Counter(key[:3] for key in groups)
    counts = Counter()
    labels = {}
    lines = [graph["domain"] + " task context", "Task: " + graph["task"],
             "Reference action rules. Check current state; effects are not guaranteed."]

    references = {}
    goals = []
    if not private:
        metadata = spec.get("goal_metadata", [])
        for i, p in enumerate(spec.get("goals", [])):
            meta = metadata[i] if i < len(metadata) else {}
            label = predicate_text(p)
            tags = []
            if multi_phase or meta.get("phase", 0):
                tags.append("phase " + str(meta.get("phase", 0)))
            if meta.get("scope"):
                tags.append(str(meta["scope"]))
            if tags:
                label += " [" + ", ".join(tags) + "]"
            references["goal:" + str(i)] = label
            goals.append(label)
        for i, e in enumerate(spec.get("events", [])):
            bindings = e.get("bindings", {})
            label = e["action"] + "(" + ", ".join(str(v) for k, v in bindings.items() if k != "$actor") + ")"
            if bindings.get("$actor", "agent") != "agent":
                label = bindings["$actor"] + ": " + label
            if multi_phase or e.get("phase", 0):
                label += " [phase " + str(e.get("phase", 0)) + "]"
            references["event:" + str(i)] = label
            goals.append("perform " + label)
        if goals:
            lines.extend(["", "Goals:"] + ["- " + x for x in unique(goals)])
        if spec.get("binding_constraints"):
            lines.append("Bind task objects: " + binding_text(spec["binding_constraints"]))

    lines.extend(["", "Action rules:"])
    for key, ids in groups.items():
        action, phase, candidate, body = key
        counts[key[:3]] += 1
        tags = []
        if multi_phase or phase:
            tags.append("phase " + str(phase))
        if candidate:
            tags.append("semantic candidate")
        if totals[key[:3]] > 1:
            tags.append("alternative " + str(counts[key[:3]]))
        label = action + (" [" + ", ".join(tags) + "]" if tags else "")
        labels.update({id: label for id in ids})
        lines.append("- " + label)
        lines.extend("  " + x for x in body)
    if not groups:
        lines.append("- No matching rules found.")
    references.update(labels)

    # Describe possible producers instead of implying a mandatory sequence.
    dependencies = defaultdict(list)
    cyclic = False
    for d in graph.get("dependencies", []):
        if d["producer"] not in labels or d["consumer"] not in labels:
            continue  # Goals are already shown alongside action outcomes.
        if d.get("cycle_reference") or d["producer"] == d["consumer"]:
            cyclic = True
            continue
        predicate = predicate_text(d.get("produced_predicate", d["predicate"]))
        guards = []
        for key, name in [("when", "effect"), ("condition_when", "requirement")]:
            guard = guard_text(d.get(key))
            if guard:
                guards.append(name + " WHEN " + guard)
        key = (labels[d["consumer"]], predicate, tuple(guards))
        dependencies[key].append(labels[d["producer"]])
    if dependencies:
        lines.extend(["", "Possible prerequisites (only if needed):"])
        for (consumer, predicate, guards), producers in dependencies.items():
            lines.append("- " + " OR ".join(unique(producers)) + " can provide " + predicate + " for " + consumer + ("; " + "; ".join(guards) if guards else ""))

    known = []
    for item in graph.get("grounding", []):
        if private and item.get("consumer") not in labels:
            continue
        if item.get("status") in {"true", "false"}:
            guard = guard_text(item.get("when"))
            known.append(predicate_text(item["predicate"]) + ": " + item["status"] + (" WHEN " + guard if guard else ""))
    if known:
        lines.extend(["", "Known current state:"] + ["- " + x for x in unique(known)])

    if not private:
        orders = []
        for o in graph.get("text_order_constraints", []):
            if o["before"] in references and o["after"] in references:
                orders.append(references[o["before"]] + " BEFORE " + references[o["after"]])
        for o in spec.get("phase_ordering", []):
            orders.append(f"phase {o['before']} BEFORE phase {o['after']}")
        if not spec.get("phase_ordering"):
            for o in graph.get("phase_order_constraints", []):
                if o["source"] in labels and o["target"] in labels:
                    orders.append(labels[o["source"]] + " BEFORE " + labels[o["target"]])
        if orders:
            lines.extend(["", "Requested order:"] + ["- " + x for x in unique(orders)])
        prohibitions = [p.get("text", "") for p in spec.get("prohibitions", [])]
        if any(prohibitions):
            lines.extend(["", "Do not:"] + ["- " + x for x in unique(prohibitions) if x])

    notes = []
    unresolved = [u for u in graph.get("unresolved", []) if not private or isinstance(u, dict) and u.get("consumer") in labels]
    if any(x.get("status") == "unknown" for x in graph.get("grounding", []) if not private or x.get("consumer") in labels):
        notes.append("Current state is incomplete; verify action requirements before use.")
    if cyclic or graph.get("has_cycle"):
        notes.append("Some rules depend on each other; resolve them from current state, not by executing a cycle.")
    for u in unresolved:
        if not isinstance(u, dict):
            notes.append(str(u))
            continue
        reason = u.get("reason", "")
        if reason in {"possible_outcome_only", "dependency_cycle", "initial_scene_state_unspecified", "requires_current_state_or_resource_resolution"}:
            continue
        if reason == "no_supported_producer":
            if u.get("consumer", "").startswith(("goal:", "event:")) and u.get("predicate"):
                notes.append("No supported action found for " + predicate_text(u["predicate"]) + ".")
            continue
        detail = u.get("text") or u.get("clause") or (predicate_text(u["predicate"]) if u.get("predicate") else (u.get("native_event") or {}).get("action", ""))
        notes.append(reason.replace("_", " ") + (": " + detail if detail else ""))
    if any(not private or t.get("consumer") in labels for t in graph.get("truncated", [])):
        notes.append("Retrieval reached its limit; additional prerequisites may be missing.")
    if not private:
        for c in graph.get("resource_conflicts", []):
            notes.append("Conflicting holding goals: " + ", ".join(c.get("objects", [])) + "; capacity " + str(c.get("capacity", "?")) + ".")
        for t in graph.get("state_threats", []):
            if t.get("threat") in labels and t.get("consumer") in labels and labels[t["threat"]] != labels[t["consumer"]]:
                notes.append(labels[t["threat"]] + " may invalidate " + predicate_text(t["predicate"]) + " needed by " + labels[t["consumer"]] + ".")
    if notes:
        lines.extend(["", "Checks:"] + ["- " + x for x in unique(notes) if x])
    return "\n".join(lines) + "\n"
