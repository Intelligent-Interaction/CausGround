"""Optional controlled-condition evidence. No simulator or action rule tables."""
from collections import defaultdict
from copy import deepcopy

from .builder import signature


def merge_interventions(profiles, incoming, transition=None):
    """Collect measured trials attached to an interaction without counting them twice."""
    if not isinstance(incoming, list):
        raise ValueError("interventions must be a list of recorded trial profiles")
    for profile in incoming:
        if not isinstance(profile, dict) or not isinstance(profile.get("id"), str) or not profile["id"].strip():
            raise ValueError("Each intervention requires a nonempty id")
        action = profile.get("action")
        if (not isinstance(action, dict) or not isinstance(action.get("name"), str)
                or not isinstance(action.get("argument_types", {}), dict)
                or not isinstance(profile.get("conditions"), list) or not isinstance(profile.get("trials"), list)):
            raise ValueError("Intervention requires action, conditions and trials")
        if transition is not None:
            nodes = {node["id"]: node["type"] for node in transition["before"]["nodes"]}
            actual = transition["action"]
            types = {arg["role"]: nodes[arg["node"]] for arg in actual["arguments"]}
            if action["name"] != actual["name"] or any(types.get(k) != v for k, v in action.get("argument_types", {}).items()):
                raise ValueError("Intervention action does not match its source transition")
        previous = profiles.get(profile["id"])
        if previous is not None and signature(previous) != signature(profile):
            raise ValueError("conflicting intervention profile: " + profile["id"])
        profiles[profile["id"]] = deepcopy(profile)


def _formula(table, conditions):
    """Prime success cubes: merging requires all covered cells to be observed successes."""
    conflicts = sorted(k for k, outcomes in table.items() if len(outcomes) > 1)
    cubes = {k for k, outcomes in table.items() if outcomes == {True}}
    primes = set()
    while cubes:
        used, merged = set(), set()
        for left in cubes:
            for i, char in enumerate(left):
                if char != "0": continue
                right = left[:i] + "1" + left[i+1:]
                if right in cubes:
                    used.update((left, right)); merged.add(left[:i] + "-" + left[i+1:])
        primes.update(cubes-used)
        cubes = merged
    predicates = []
    for cube in sorted(primes):
        literals = [deepcopy(c["predicate"]) if bit == "1" else {"type": "not", "predicate": deepcopy(c["predicate"])}
                    for c, bit in zip(conditions, cube) if bit != "-"]
        predicates.append(literals[0] if len(literals) == 1 else {"type": "all_of", "predicates": literals})
    return {"cubes": sorted(primes), "predicate": {"type": "any_of", "predicates": predicates},
            "truth_table_complete": len(table) == 2**len(conditions) and not conflicts,
            "observed_masks": len(table), "conflicting_masks": conflicts,
            "scope": "Observed factor assignments and sampled contexts only; unobserved cells are not assumed successful."}


def learn_interventions(profiles):
    """Each profile is one fixed base context with measured boolean factor trials."""
    groups = defaultdict(list)
    seen = set()
    for row in profiles:
        if row["id"] in seen: raise ValueError("Duplicate intervention profile ID: " + row["id"])
        seen.add(row["id"])
        if not isinstance(row["conditions"], list) or len(row["conditions"]) > 16:
            raise ValueError("Expected 0..16 measured candidate factors")
        key = signature({"action": row["action"], "conditions": row["conditions"], "fixed_context": row.get("fixed_context", []), "context_predicate": row.get("context_predicate")})
        groups[key].append(row)
    rules = []
    for key, group in sorted(groups.items()):
        conditions = group[0]["conditions"]
        n = len(conditions)
        stats = [{"successful_violations": [], "matched_failure_pairs": [], "supported_contexts": set()} for _ in conditions]
        success_contexts, table, skipped = set(), defaultdict(set), 0
        complete_controlled_contexts = 0
        for profile in group:
            trials = []
            for trial in profile["trials"]:
                bits = trial.get("observed")
                if (not isinstance(bits, list) or len(bits) != n or any(type(b) not in (int, bool) or b not in (0,1) for b in bits)
                        or type(trial.get("success")) is not bool):
                    skipped += 1; continue
                mask = ''.join(str(int(b)) for b in bits)
                identity = profile["id"] + ":" + str(trial["id"])
                trials.append((mask, trial["success"], identity))
                table[mask].add(trial["success"])
                if trial["success"]:
                    success_contexts.add(profile["id"])
                    for i, bit in enumerate(bits):
                        if not bit: stats[i]["successful_violations"].append(identity)
            local_table = defaultdict(set)
            for mask, success, _ in trials: local_table[mask].add(success)
            if (profile.get("controlled") is True and len(local_table) == 2**n
                    and all(len(values)==1 for values in local_table.values())):
                complete_controlled_contexts += 1
            if profile.get("controlled") is not True: continue
            failures = defaultdict(list)
            outcomes = defaultdict(set)
            for mask, success, _ in trials: outcomes[mask].add(success)
            for mask, success, identity in trials:
                if not success: failures[mask].append(identity)
            for mask, success, identity in trials:
                if not success or outcomes[mask] != {True}: continue
                for i, bit in enumerate(mask):
                    if bit != "1": continue
                    other = mask[:i] + "0" + mask[i+1:]
                    if outcomes[other] != {False}: continue
                    for failed_id in failures[other]:
                        stats[i]["matched_failure_pairs"].append([identity, failed_id])
                        stats[i]["supported_contexts"].add(profile["id"])
        result = {"id": f"intervention:{len(rules) + 1:04d}",
                  "action": group[0]["action"], "fixed_context": group[0].get("fixed_context", []),
                  "controlled": all(p.get("controlled") is True for p in group),
                  "when": deepcopy(group[0].get("context_predicate", {"type": "unresolved_context", "requirements": group[0]["fixed_context"]} if group[0].get("fixed_context") else {"type": "all_of", "predicates": []})),
                  "profile_ids": sorted(p["id"] for p in group), "skipped_trials": skipped,
                  "required_conditions": [], "pending_conditions": [], "rejected_candidates": [],
                  "success_formula": _formula(table, conditions)}
        result["success_formula"]["controlled_truth_table_complete"] = bool(group) and complete_controlled_contexts == len(group)
        for condition, stat in zip(conditions, stats):
            supported = stat.pop("supported_contexts")
            item = {**deepcopy(condition), **stat, "supported_contexts": sorted(supported),
                    "basis": "controlled_single_factor_contrast", "necessity_verified": False}
            if stat["successful_violations"]: destination = "rejected_candidates"
            elif success_contexts and supported == success_contexts:
                destination = "required_conditions"
                item["necessity_verified"] = True
                item["necessity_scope"] = "sampled_contexts_and_measured_factors"
            else: destination = "pending_conditions"
            result[destination].append(item)
        formula = result["success_formula"]
        if (len(formula["cubes"]) > 1 and formula["truth_table_complete"]
                and complete_controlled_contexts == len(group)):
            result["required_conditions"].append({"name": "learned_success_formula", "predicate": formula["predicate"],
                "basis": "controlled_complete_factor_table", "necessity_verified": True,
                "necessity_scope": "sampled_contexts_and_measured_factors", "profile_ids": result["profile_ids"]})
        rules.append(result)
    return rules


def attach_interventions(graph, profiles):
    """Keep empirical consistency and controlled evidence separately inspectable."""
    graph["intervention_rules"] = learn_interventions(profiles)
    for rule in graph["specific_rules"]:
        rule["tail"].update(required_conditions=[], condition_schema_status="success_consistency_filtered", status="observational_candidates_only")
        rule["tail"].pop("fixed_context", None)
        rule["tail"].pop("intervention_scope", None)
        matching = [r for r in graph["intervention_rules"] if r["action"]["name"] == rule["action"]["name"]
                    and all(rule["type_constraints"].get(k) == v for k, v in r["action"].get("argument_types", {}).items())]
        rule["tail"]["intervention_rule_ids"] = [r["id"] for r in matching]
        rule["tail"]["contextual_intervention_rule_ids"] = [r["id"] for r in matching if r["fixed_context"]]
        rule["tail"]["condition_branches"] = [{
            "id": r["id"], "when": deepcopy(r["when"]), "fixed_context": deepcopy(r["fixed_context"]),
            "required_conditions": deepcopy(r["required_conditions"]), "pending_conditions": deepcopy(r["pending_conditions"]),
            "success_formula": deepcopy(r["success_formula"]), "profile_ids": r["profile_ids"],
            "basis": "controlled_factors_with_explicit_scope" if r["controlled"] else "recorded_factors_without_controlled_contrast",
            "scope": "Measured factors and recorded contexts only; a guard is an assumption, not a learned necessity."
        } for r in matching]
        matching = [r for r in matching if r["controlled"]]
        if matching:
            rule["tail"]["condition_schema_status"] = "scoped_controlled_factors"
            rule["tail"]["status"] = "inspect_applicable_condition_branches"
        matching = [r for r in matching if not r["fixed_context"] and r["when"] == {"type": "all_of", "predicates": []}]
        if matching:
            tail = rule["tail"]
            supported = [{signature(c["predicate"]): c for c in r["required_conditions"]} for r in matching]
            common = set(supported[0]).intersection(*(set(s) for s in supported[1:]))
            tail["required_conditions"] = [{**deepcopy(supported[0][k]), "source_rule_ids": [r["id"] for r in matching]} for k in sorted(common)]
            tail["condition_schema_status"] = "controlled_factors_filtered"
            tail["status"] = "necessity_supported_within_measured_factor_scope"
            tail["fixed_context"] = [r["fixed_context"] for r in matching if r["fixed_context"]]
            tail["intervention_scope"] = "Only supplied candidate factors and sampled contexts were tested; other observational candidates remain unverified. Inspect intervention_rules for rejected factors, pending factors and compound success formulas."
    return graph
