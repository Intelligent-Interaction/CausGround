"""Small lexical action detector used by the CausalGround goal extractor."""
import re

def actions_in(text):
    text = text.lower()
    rules = {
        "PICK": r"\b(pick|take|retrieve|grab|select)\b",
        "PLACE": r"\b(place|put|move|return|stack|gather|sort|pack|weigh|recycle)\b",
        "OPEN": r"\bopen\b",
        "CLOSE": r"\bclose\b",
        "TILT": r"\b(dump|pour)\b",
        "STIR": r"\bstir\b",
        "SCRUB": r"\b(scrub|scrubbing)\b",
        "WAIT": r"\b(wait|once|finish|pops)\b",
        "SLIDE": r"\b(slide|rack)\b",
        "PRESS": r"\b(press|start)\b",
        "ROTATE": r"\b(turn|timer|water|wash|rinse|spout|burner)\b",
    }
    actions = {action for action, pattern in rules.items() if re.search(pattern, text)}
    if "wash" in text or "running water" in text: actions.add("PLACE")
    if "clean" in text and "board" in text: actions.add("SCRUB")
    if re.search(r"\btake\b.*\bto\b", text): actions.add("PLACE")
    return actions
