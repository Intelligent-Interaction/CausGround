"""Offline ingestion: text or canonical states -> action evidence -> CausGround."""
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import json
import lzma
import math
import sys
from pathlib import Path

from .builder import CausGroundBuilder, signature, validate_transition
from .state import SYMMETRIC, entity_key, image_data_url, observation_prompt

PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "text_world_state.txt"


def strict_loads(raw):
    def reject(value):
        raise ValueError("Non-finite JSON value: " + value)
    def number(value):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else reject(value)
    return json.loads(raw, parse_constant=reject, parse_float=number)


def iter_records(path):
    path = Path(path)
    opener = {".gz": gzip.open, ".xz": lzma.open}.get(path.suffix, open)
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        if ".jsonl" in path.suffixes:
            for line in stream:
                if line.strip():
                    row = strict_loads(line)
                    if not isinstance(row, dict): raise ValueError("Each transition must be an object")
                    yield row
            return
        else:
            rows = strict_loads(stream.read())
            if isinstance(rows, dict):
                rows = rows.get("trajectories")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Input must be a nonempty list of action transitions")
    yield from rows


def read_records(path):
    return list(iter_records(path))


def _selector(selector):
    if isinstance(selector, str) and selector.strip():
        return
    if (not isinstance(selector, dict) or not selector or set(selector) - {"id", "type", "attributes"}
            or not (selector.get("id") or selector.get("type"))
            or not isinstance(selector.get("attributes", {}), dict)):
        raise ValueError("Entity selector requires an ID or {type/id, attributes?}")
    for field in ("id", "type"):
        if field in selector and (not isinstance(selector[field], str) or not selector[field].strip()):
            raise ValueError("Entity ID/type must be a nonempty string")


def _action_template(action):
    if not isinstance(action, dict) or not isinstance(action.get("name"), str) or not action["name"].strip():
        raise ValueError("action requires a nonempty name")
    _selector(action.get("actor"))
    if not isinstance(action.get("arguments"), list):
        raise ValueError("action.arguments must be a list")
    roles = {"actor"}
    for arg in action["arguments"]:
        role = arg.get("role") if isinstance(arg, dict) else None
        if not isinstance(role, str) or not role.strip() or role.startswith("$") or role in roles:
            raise ValueError("Action argument roles must be unique nonempty names")
        roles.add(role)
        _selector(arg.get("node"))


def bind_action(action, graph):
    """Resolve explicit IDs or exact type/attribute selectors, never guess a match."""
    _action_template(action)
    def resolve(selector):
        selector = {"id": selector} if isinstance(selector, str) else selector
        matches = [n["id"] for n in graph["nodes"]
                   if all(n[k] == v for k, v in selector.items() if k != "attributes")
                   and all(k in n["attributes"] and signature(n["attributes"][k]) == signature(v)
                           for k, v in selector.get("attributes", {}).items())]
        if len(matches) != 1:
            raise ValueError("Action entity missing or ambiguous: " + signature(selector))
        return matches[0]
    return {"name": action["name"], "actor": resolve(action["actor"]),
            "arguments": [{"role": a["role"], "node": resolve(a["node"])} for a in action["arguments"]]}


def _check_identity(before, after):
    """Catch obvious renaming; not a general tracker for indistinguishable objects."""
    old = {n["id"]: n["type"] for n in before["nodes"]}
    if any(n["id"] in old and old[n["id"]] != n["type"] for n in after["nodes"]):
        raise ValueError("Observed entity identity changed type")
    mappings = []
    for state in (before, after):
        mapping = defaultdict(list)
        for node in state["nodes"]:
            mapping[entity_key(node)].append(node["id"])
        mappings.append(mapping)
    left, right = mappings
    for key in left.keys() & right.keys():
        if len(left[key]) == len(right[key]) == 1 and left[key] != right[key]:
            raise ValueError("Observed entity identity changed ID; provide stable structured states")


def _canonical(graph):
    graph = deepcopy(graph)
    for edge in graph["edges"]:
        if edge["relation"] in SYMMETRIC:
            edge["source"], edge["target"] = sorted((edge["source"], edge["target"]))
    return graph


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def run(input_path, output_dir, *, input_format="states", model=None, prompt=None, min_support=1,
        condition_evidence=None, evidence_mode="full", effect_policy=None, workers=1):
    """Model is a callable(messages) -> raw JSON string. No labels are required."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Output already exists; choose a new directory")
    if input_format not in {"states", "text", "image", "virtualhome"}:
        raise ValueError("input_format must be states, text, image or virtualhome")
    if evidence_mode not in {"full", "source"}: raise ValueError("evidence_mode must be full or source")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    rows = iter_records(input_path)
    extracted = input_format in {"text", "image"}
    if extracted and workers != 1:
        raise ValueError("Text/image extraction uses one model process; use workers=1")
    if extracted:
        from .observe import extract_world_state
        if model is None:
            raise ValueError("Observation input requires a model")
        prompt = observation_prompt(input_format == "image") if prompt is None else prompt
        seen = set()
        for row in iter_records(input_path):
            for field in (("id", "pre_obs", "post_obs") if input_format == "text" else ("id", "pre_image", "post_image")):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError("Observation trajectory requires nonempty " + field)
            if input_format == "image":
                for phase in ("pre", "post"):
                    if not isinstance(row.get(phase + "_obs", ""), str):
                        raise ValueError("Observation text must be a string")
                    image_data_url(Path(input_path).parent / row[phase + "_image"])
            if row["id"] in seen:
                raise ValueError("Observation trajectory IDs must be unique")
            seen.add(row["id"])
            if not isinstance(row.get("task", ""), str) or row.get("outcome") not in {"success", "failure", "unknown"}:
                raise ValueError("Invalid task or outcome")
            _action_template(row.get("action"))
    elif input_format == "virtualhome":
        from .virtualhome import iter_dataset
        rows = iter_dataset(input_path)
    effect_policy = effect_policy or ("virtualhome" if input_format == "virtualhome" else "observed")
    builder, normalized = CausGroundBuilder(min_support=min_support, effect_policy=effect_policy), {}
    profiles = {}
    from .interventions import merge_interventions
    if not extracted and workers > 1:
        from .parallel import prepare_transitions
        rows = prepare_transitions(rows, builder, workers, evidence_mode)
    output.mkdir(parents=True, exist_ok=False)
    current_id = None
    input_count = 0
    try:
        if extracted:
            (output / "prompt.txt").write_text(prompt, encoding="utf-8")
        for row in rows:
            input_count += 1
            current_id = row["id"]
            if extracted:
                interventions = row.get("interventions", [])
                states = []
                for phase in ("pre", "post"):
                    image = Path(input_path).parent / row[phase + "_image"] if input_format == "image" else None
                    trace = []
                    try:
                        states.append(extract_world_state(model, task=row.get("task", ""), text=row.get(phase + "_obs", ""),
                            image=image, previous_state=states[0] if states else None, prompt=prompt, trace=trace))
                    finally:
                        for item in trace:
                            _append(output / "extractions.jsonl", {"trajectory_id": current_id, "phase": phase, **item})
                _check_identity(*states)
                row = {"id": current_id, "before": states[0], "after": states[1],
                       "action": bind_action(row["action"], states[0]), "outcome": row["outcome"],
                       "provenance": {"source": input_format + "_extraction", "task": row.get("task", ""),
                                      "pre_obs": row.get("pre_obs", ""), "post_obs": row.get("post_obs", ""),
                                      **({"pre_image": row["pre_image"], "post_image": row["post_image"]} if input_format == "image" else {})}}
                row["interventions"] = interventions
            evidence = row if workers > 1 else builder.add(row)
            merge_interventions(profiles, evidence.get("interventions", []), builder._records[current_id])
            if current_id not in normalized:
                normalized[current_id] = {"action": evidence["action"], "outcome": evidence["outcome"]}
                if evidence_mode == "full": _append(output / "transitions.jsonl", evidence)
                else: _append(output / "sources.jsonl", {"id": current_id, "action": evidence["action"],
                             "outcome": evidence["outcome"], "provenance": evidence.get("provenance", {}),
                             "head": evidence["head"], "effect_bindings": evidence.get("effect_bindings", {}),
                             **({"interventions": evidence["interventions"]} if evidence.get("interventions") else {})})
            if workers > 1 and input_count % 1000 == 0:
                print(f"Processed {input_count} interactions", file=sys.stderr, flush=True)
        if not input_count: raise ValueError("Input contains no transitions")
        if workers > 1:
            print(f"Processed {input_count} interactions; summarizing action rules", file=sys.stderr, flush=True)
        graph = builder.snapshot()
        if condition_evidence is not None:
            merge_interventions(profiles, list(iter_records(condition_evidence)))
        if profiles:
            from .interventions import attach_interventions
            attach_interventions(graph, profiles.values())
        summary = {"status": "complete", "input_format": input_format, "evidence_mode": evidence_mode, "input_records": input_count,
                   "effect_policy": effect_policy,
                   "unique_transitions": len(normalized),
                   "actions": dict(Counter(r["action"]["name"] for r in normalized.values())),
                   "outcomes": dict(Counter(r["outcome"] for r in normalized.values())),
                   "generic_rules": len(graph["generic_rules"]), "specific_rules": len(graph["specific_rules"]),
                   "intervention_rules": len(graph.get("intervention_rules", []))}
        summary["condition_profiles"] = len(profiles)
        _write_json(output / "causground.json", graph)
        _write_json(output / "summary.json", summary)
        return summary
    except Exception as exc:
        _write_json(output / "summary.json", {"status": "failed", "trajectory_id": current_id,
                                              "accepted_transitions": len(normalized), "error": str(exc)})
        raise
