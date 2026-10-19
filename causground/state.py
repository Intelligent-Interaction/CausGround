"""Observation-state validation and text/image extraction messages."""
import base64
import json
import math
from pathlib import Path
import re

SPATIAL = frozenset("ON INSIDE LEFT_OF RIGHT_OF IN_FRONT_OF BEHIND ABOVE BELOW CLOSE FAR TOUCHING".split())
FUNCTIONAL = frozenset("HOLDS_RH HOLDS_LH CONTROLS CAN_SUPPLY_POWER PLUGGED_INTO OPENS_OR_CLOSES ADJUSTS ACTIVATES PAIRED_WITH".split())
SYMMETRIC = frozenset("CLOSE FAR TOUCHING PAIRED_WITH".split())
DIRECTIONAL = frozenset("LEFT_OF RIGHT_OF IN_FRONT_OF BEHIND ABOVE BELOW".split())


def _span(value, observation, image_count=0):
    if isinstance(value, dict):
        if (set(value) == {"image", "description"} and type(value["image"]) is int
                and 0 <= value["image"] < image_count and isinstance(value["description"], str)
                and value["description"].strip()):
            return
        raise ValueError("Invalid image evidence reference")
    if not isinstance(value, str) or not value.strip() or value not in observation:
        raise ValueError("Evidence must be a nonempty exact observation substring")


def validate_extraction(graph, observation, *, image_count=0):
    """Validate structure and evidence references, NOT semantic/visual correctness."""
    if not isinstance(graph, dict) or graph.get("relations_complete") is not False:
        raise ValueError("Observations require relations_complete=false")
    if set(graph) != {"nodes", "edges", "relations_complete"}:
        raise ValueError("Unexpected/missing graph fields")
    if not isinstance(graph["nodes"], list) or not isinstance(graph["edges"], list):
        raise ValueError("nodes and edges must be lists")
    ids = set()
    for node in graph["nodes"]:
        if not isinstance(node, dict) or set(node) != {"id", "type", "attributes", "evidence", "attribute_evidence"}:
            raise ValueError("Invalid node fields")
        nid = node["id"]
        if not isinstance(nid, str) or not nid.strip() or nid in ids:
            raise ValueError("Node IDs must be unique nonempty strings")
        ids.add(nid)
        if not isinstance(node["type"], str) or not re.fullmatch(r"[a-z][a-z0-9_]*", node["type"]):
            raise ValueError("Invalid canonical object type")
        attrs, evidence = node["attributes"], node["attribute_evidence"]
        if not isinstance(attrs, dict) or not isinstance(evidence, dict) or set(attrs) != set(evidence):
            raise ValueError("Every attribute requires evidence")
        _span(node["evidence"], observation, image_count)
        for key, value in attrs.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
                raise ValueError("Invalid attribute name")
            if type(value) not in (str, bool, int, float) or (isinstance(value, float) and not math.isfinite(value)):
                raise ValueError("Attributes must be finite atomic JSON values")
            if key.startswith("is_") and type(value) is not bool:
                raise ValueError("is_* attributes must be JSON booleans")
            if key == "color" and (not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z_]*", value)):
                raise ValueError("color must be a lowercase English string")
            _span(evidence[key], observation, image_count)
    seen = set()
    for edge in graph["edges"]:
        if not isinstance(edge, dict) or set(edge) != {"source", "relation", "target", "family", "evidence"}:
            raise ValueError("Invalid edge fields")
        if not isinstance(edge["source"], str) or not isinstance(edge["target"], str) or edge["source"] not in ids or edge["target"] not in ids:
            raise ValueError("Dangling edge")
        relation = edge["relation"]
        if not isinstance(relation, str) or relation not in SPATIAL | FUNCTIONAL:
            raise ValueError("Unknown relation")
        if edge["family"] != ("spatial" if relation in SPATIAL else "functional"):
            raise ValueError("Wrong relation family")
        if relation in DIRECTIONAL and isinstance(edge["evidence"], dict):
            raise ValueError("Image-relative directions require explicit textual reference-frame evidence")
        ends = (edge["source"], edge["target"])
        if relation in SYMMETRIC: ends = tuple(sorted(ends))
        token = (relation, *ends)
        if token in seen: raise ValueError("Duplicate relation")
        seen.add(token)
        _span(edge["evidence"], observation, image_count)


def entity_key(node):
    """Deliberately small diagnostic matcher; not a general entity resolver."""
    return (node["type"], node.get("attributes", {}).get("color"))


def entity_registry(graph):
    return [{"id": n["id"], "type": n["type"], "attributes":
             {k: v for k, v in n["attributes"].items() if k == "color"}}
            for n in graph["nodes"]]


def observation_prompt(image=False):
    folder = Path(__file__).resolve().parents[1] / "prompts"
    return (folder / ("image_world_state.txt" if image else "text_world_state.txt")).read_text(encoding="utf-8")


def image_data_url(path):
    data = Path(path).read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise ValueError("Image must be a PNG, JPEG or WebP file")
    return "data:" + mime + ";base64," + base64.b64encode(data).decode("ascii")


def build_messages(prompt, task, observation, registry=None, *, image=None):
    if not isinstance(task, str) or not isinstance(observation, str):
        raise ValueError("task and observation must be strings")
    if not observation.strip() and image is None:
        raise ValueError("Provide observation text or an image")
    content = json.dumps({"task": task, "current_obs": observation,
                          "previous_entities": registry or []}, ensure_ascii=False)
    if image is not None:
        content = [{"type": "text", "text": content},
                   {"type": "image_url", "image_url": {"url": image_data_url(image)}}]
    return [{"role": "system", "content": prompt}, {"role": "user", "content": content}]
