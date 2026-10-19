"""python -m causground.query --graph library.json.gz --task 'Open the fridge.'"""
import argparse
import json
from pathlib import Path
import sys

from .goals import LLMGoalParser
from .knowledge import read_json
from .retrieval import CausGround
from .semantic import EmbeddingRetriever, LexicalRetriever, Qwen3Encoder


def main(argv=None):
    p = argparse.ArgumentParser(description="Retrieve one task's causal/semantic/hybrid CausGround context.")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--env", choices=("virtualhome", "robocasa"), help="Use a bundled environment graph")
    source.add_argument("--graph", type=Path, help="Use your own CausGround graph")
    p.add_argument("--vocabulary", type=Path)
    p.add_argument("--task", required=True)
    p.add_argument("--goals", type=Path, help="Goal specification JSON; bypasses text goal parsing")
    p.add_argument("--profile", default="auto", choices=("auto","standard","robocasa","virtualhome_legacy"))
    p.add_argument("--output-format", default="common", choices=("common","virtualhome-legacy"))
    p.add_argument("--initial-state", type=Path, help="Legacy explicit predicate list JSON")
    p.add_argument("--state", type=Path, help="Optional standard world state JSON")
    p.add_argument("--mode", choices=("causal", "semantic", "hybrid"), default="hybrid")
    p.add_argument("--retriever", choices=("lexical", "embedding"), default="lexical")
    p.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--embedding-backend", choices=("sentence-transformers", "qwen3-transformers"), default="sentence-transformers")
    p.add_argument("--device", help="Optional sentence-transformers device, e.g. cpu")
    p.add_argument("--goal-backend", choices=("grammar", "dashscope", "vllm"), default="grammar")
    p.add_argument("--model", help="LLM model name; grammar needs no model")
    p.add_argument("--model-path", type=Path, help="Existing local vLLM LLM directory")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-nodes", type=int, default=32)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--max-producers", type=int, default=2)
    p.add_argument("--json", action="store_true", help="Print the structured subgraph")
    p.add_argument("--output", type=Path, help="Optional NEW directory for subgraph.json and context.txt")
    args = p.parse_args(argv)
    try:
        if args.output and args.output.exists(): raise ValueError("Use a new output directory")
        if args.env:
            data = Path(__file__).resolve().parents[1] / "data" / args.env
            args.graph = data / "causground.json.gz"
            if args.env == "virtualhome" and args.vocabulary is None:
                args.vocabulary = data / "vocabulary.json"
        graph = CausGround.load(args.graph, vocabulary=args.vocabulary, profile=args.profile)
        parser = None
        if args.goal_backend != "grammar" and args.mode != "semantic" and args.goals is None:
            from .models import make_model
            model=make_model(args.goal_backend, model=args.model, model_path=args.model_path)
            if graph.profile=='virtualhome_legacy':
                from .adapters.virtualhome import LegacyVirtualHomeGoalParser
                parser=LegacyVirtualHomeGoalParser(model,graph.graph)
            else:parser=LLMGoalParser(model)
        ranker = None
        if args.mode != "causal":
            if args.retriever == "embedding":
                encoder = Qwen3Encoder(args.embedding_model, device=args.device or "cpu") if args.embedding_backend == "qwen3-transformers" else None
                ranker = EmbeddingRetriever(args.embedding_model, device=args.device, encoder=encoder)
            else: ranker = LexicalRetriever()
        result = graph.retrieve(args.task, goals=read_json(args.goals) if args.goals else None,
            state=read_json(args.state) if args.state else None, initial_state=read_json(args.initial_state) if args.initial_state else None, mode=args.mode,
            goal_parser=parser, retriever=ranker, top_k=args.top_k, max_nodes=args.max_nodes,
            max_depth=args.max_depth, max_producers=args.max_producers)
        output_graph=result.subgraph
        if args.output_format=='virtualhome-legacy':
            from .adapters.virtualhome import to_legacy_subgraph
            output_graph=to_legacy_subgraph(result,task={'instruction':args.task},seed=read_json(args.goals) if args.goals else None)
        encoded = json.dumps(output_graph, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "subgraph.json").write_text(encoded + "\n", encoding="utf-8")
            (args.output / "context.txt").write_text(result.context, encoding="utf-8")
        print(encoded if args.json else result.context)
    except Exception as exc:
        print("CausGround query: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
