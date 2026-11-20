"""Generate task-only SWITCH contexts; no model answering or evaluation scoring."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causground.switch import SwitchCWM


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=Path, default=Path(__file__).resolve().parents[1] / 'data/switch/tasks.json')
    p.add_argument('--graph', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--version', choices=('simple', 'complex'), default='complex')
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a new output directory; existing files are never overwritten')
    tasks = json.loads(args.tasks.read_text(encoding='utf-8'))
    if not isinstance(tasks, list) or any(not isinstance(t, str) or not t.strip() for t in tasks):
        p.error('tasks must be a JSON list of nonempty task strings, without QA or media')
    library = SwitchCWM.load(args.graph)
    args.output.mkdir(parents=True, exist_ok=False)
    summary = []
    for n, task in enumerate(tasks, 1):
        result = library.retrieve(task, version=args.version)
        target = args.output / f'{n:04d}'
        target.mkdir(exist_ok=False)
        (target / 'task.json').write_text(json.dumps({'task': task}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        (target / 'subgraph.json').write_text(json.dumps(result.subgraph, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        (target / 'context.txt').write_text(result.context, encoding='utf-8')
        summary.append({'task': task, 'directory': target.name, 'rules': len(result.subgraph['nodes']),
                        'characters': len(result.context), 'version': result.subgraph['audit']['version'],
                        'goal_source': result.subgraph['audit']['goal_source']})
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'contexts': len(summary), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
