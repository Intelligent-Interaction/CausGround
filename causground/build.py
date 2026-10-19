"""python -m causground build --input trajectories.json --output new-directory"""
import argparse
import json
from pathlib import Path
import sys

from .pipeline import run


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build an empirical CausGround from offline action transitions.")
    parser.add_argument("--input", type=Path, required=True, help="JSON list or JSONL, optionally .gz/.xz; includes recorded interventions when available")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing output is never overwritten")
    parser.add_argument("--format", choices=("states", "text", "image", "virtualhome"), default="states")
    parser.add_argument("--min-support", type=int, default=1, help="Minimum successful executions for accepted predicates/effects")
    parser.add_argument("--condition-evidence", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, default=1, help="CPU workers for structured trajectories (default: 1)")
    parser.add_argument("--effect-policy", choices=("observed", "virtualhome"), help="Explicit effect relevance prior; defaults to virtualhome for raw VH input, observed otherwise")
    parser.add_argument("--evidence-mode", choices=("full", "source"), default="full", help="Use source for large existing datasets to avoid copying full scenes")
    parser.add_argument("--backend", choices=("dashscope", "vllm"), default="dashscope", help="Text/image input only")
    parser.add_argument("--model", help="Default: qwen-plus, or local model directory name")
    parser.add_argument("--model-path", type=Path, help="Existing local model directory for vllm")
    parser.add_argument("--timeout", type=float, default=45, help="API timeout in seconds, at most 60")
    parser.add_argument("--prompt", type=Path, help="Optional replacement observation extraction prompt")
    args = parser.parse_args(argv)
    try:
        model = None
        prompt = None
        if args.format in {"text", "image"}:
            from .models import make_model
            model = make_model(args.backend, model=args.model, model_path=args.model_path, timeout=args.timeout,
                               allow_images=args.format == "image")
            prompt = args.prompt.read_text(encoding="utf-8") if args.prompt else None
        result = run(args.input, args.output, input_format=args.format, model=model, prompt=prompt, min_support=args.min_support,
                     condition_evidence=args.condition_evidence, evidence_mode=args.evidence_mode, effect_policy=args.effect_policy,
                     workers=args.workers)
    except Exception as exc:
        print("CausGround: " + str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
