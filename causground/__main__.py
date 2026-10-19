"""Command-line entrypoint for building and using CausGround."""
import argparse
from importlib import import_module
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    commands = {"build": "Build a graph from action-level trajectories.",
                "observe": "Convert a text observation or image into a world state.",
                "query": "Retrieve a task subgraph and model context."}
    if args and args[0] in commands:
        return import_module("." + args[0], __package__).main(args[1:])
    # Existing `python -m causground --input ...` commands still build a graph.
    if args and args[0].startswith("-") and args[0] not in {"-h", "--help"}:
        return import_module(".build", __package__).main(args)
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    for name, description in commands.items():
        subparsers.add_parser(name, help=description, add_help=False)
    if not args:
        parser.print_help()
        return 0
    parser.parse_args(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
