"""Generate the bundled RoboCasa and VirtualHome task subgraphs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from causground import CausGround


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", "--environment", choices=("both", "robocasa", "virtualhome"), default="both")
    parser.add_argument("--output", type=Path, required=True, help="A new output directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {}
    for environment in ("robocasa", "virtualhome"):
        if args.env not in ("both", environment):
            continue
        vh = environment == "virtualhome"
        data = ROOT / "data" / environment
        graph = CausGround.load(data / ("task_graph.json.gz" if vh else "causground.json.gz"))
        tasks = json.loads((data / "tasks.json").read_text(encoding="utf-8"))
        for index, task in enumerate(tasks, 1):
            result = graph.retrieve(task.get("task", task.get("description")), goals=task.get("goals"),
                                    mode="causal", max_nodes=48 if vh else 32,
                                    max_depth=4 if vh else 6, max_producers=2 if vh else 1)
            folder = args.output / environment / f"{index:03d}"
            folder.mkdir(parents=True)
            (folder / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
            (folder / "subgraph.json").write_text(json.dumps(result.subgraph, ensure_ascii=False, indent=2), encoding="utf-8")
            (folder / "context.txt").write_text(result.context, encoding="utf-8")
            print(f"{environment}: {index}/{len(tasks)} {task['id']}", flush=True)
        summary[environment] = len(tasks)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
