"""Pure native predicate helpers used only during task interpretation."""
import json


def predicate_key(predicate):
    if predicate.get("type") == "edge":
        return ("edge", predicate["source"], predicate["relation"],
                predicate["target"], bool(predicate.get("present", True)))
    if predicate.get("type") in {"node_attribute", "attribute"}:
        return ("attribute", predicate.get("node", predicate.get("object")),
                predicate["attribute"], json.dumps(predicate["value"], sort_keys=True))
    return None


def effect_predicate(effect):
    if effect.get("operation") in {"add_edge", "delete_edge"}:
        return {"type": "edge", **effect["edge"],
                "present": effect["operation"] == "add_edge"}
    if effect.get("operation") == "set_attribute":
        return {"type": "node_attribute", "node": effect["node"],
                "attribute": effect["attribute"], "value": effect["after"]}
    return None
