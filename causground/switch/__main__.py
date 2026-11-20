"""python -m causground.switch query --task 'Turn on the kettle.'"""
import argparse
import json
from pathlib import Path
import sys
from . import SwitchCWM


def main(argv=None):
    parser = argparse.ArgumentParser(description='Retrieve SWITCH CWM for your model prompt.')
    sub = parser.add_subparsers(dest='command', required=True)
    query = sub.add_parser('query')
    query.add_argument('--task', required=True)
    query.add_argument('--graph', type=Path)
    query.add_argument('--goals', type=Path)
    query.add_argument('--version', choices=('simple', 'complex'), default='simple')
    query.add_argument('--top-k', type=int, default=4)
    query.add_argument('--context-chars', type=int, default=14000)
    query.add_argument('--output', type=Path, help='New directory for subgraph.json and context.txt')
    query.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.exists():
            raise ValueError('Use a new output directory; existing files are never overwritten')
        library = SwitchCWM.load(args.graph)
        goals = json.loads(args.goals.read_text(encoding='utf-8')) if args.goals else None
        result = library.retrieve(args.task, version=args.version, goals=goals, top_k=args.top_k, context_chars=args.context_chars)
        if args.json or args.output:
            encoded = json.dumps(result.subgraph, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if args.output:
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / 'subgraph.json').write_text(encoded, encoding='utf-8')
            (args.output / 'context.txt').write_text(result.context, encoding='utf-8')
        print(encoded if args.json else result.context, end='')
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print('SWITCH query: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
