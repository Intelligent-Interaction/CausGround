"""Convert one text observation or image into a CausGround world state."""
import argparse
import json
from pathlib import Path
import sys

from .pipeline import _canonical, strict_loads
from .state import build_messages, entity_registry, observation_prompt, validate_extraction


def extract_world_state(model, *, task="", text="", image=None, previous_state=None, prompt=None, trace=None):
    """Return a partial world state. A model is callable(messages) -> JSON text."""
    prompt = observation_prompt(image is not None) if prompt is None else prompt
    messages = build_messages(prompt, task, text, entity_registry(previous_state) if previous_state else [], image=image)
    raw = model(messages)
    if trace is not None:
        trace.append({"messages": messages, "raw_content": raw})
    if not isinstance(raw, str):
        raise ValueError("Model must return a JSON string")
    graph = strict_loads(raw)
    validate_extraction(graph, text, image_count=int(image is not None))
    return _canonical(graph)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="")
    text = p.add_mutually_exclusive_group()
    text.add_argument("--text", default="")
    text.add_argument("--text-file", type=Path)
    p.add_argument("--image", type=Path, help="Current PNG, JPEG or WebP image; can accompany text")
    p.add_argument("--previous-state", type=Path, help="Previous world state, used only for identity hints")
    p.add_argument("--backend", choices=("vllm", "dashscope"), default="vllm")
    p.add_argument("--model-path", type=Path)
    p.add_argument("--model")
    p.add_argument("--output", type=Path, required=True, help="New directory for world_state.json and extraction.json")
    args = p.parse_args(argv)
    trace = []
    try:
        if args.output.exists():
            raise FileExistsError("Output already exists; choose a new directory")
        observation = args.text_file.read_text(encoding="utf-8") if args.text_file else args.text
        previous = strict_loads(args.previous_state.read_text(encoding="utf-8")) if args.previous_state else None
        # Validate local inputs before loading a model or calling a provider.
        build_messages("", args.task, observation, image=args.image)
        from .models import make_model
        model = make_model(args.backend, model=args.model, model_path=args.model_path, allow_images=args.image is not None)
        args.output.mkdir(parents=True, exist_ok=False)
        graph = extract_world_state(model, task=args.task, text=observation, image=args.image, previous_state=previous, trace=trace)
        encoded = json.dumps(graph, ensure_ascii=False, indent=2, allow_nan=False)
        (args.output / "world_state.json").write_text(encoded + "\n", encoding="utf-8")
        print(encoded)
    except Exception as exc:
        print("CausGround observation: " + str(exc), file=sys.stderr)
        return 1
    finally:
        if trace:
            with (args.output / "extraction.json").open("x", encoding="utf-8") as stream:
                json.dump(trace[0], stream, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
