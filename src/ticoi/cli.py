"""Command-line interface for the TICOI JSON configuration runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ticoi.config import ConfigError, load_config, run_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ticoi", description="Run TICOI cube processing from a JSON configuration.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="load, process, and write a configured TICOI cube")
    run.add_argument("config", type=Path, metavar="CONFIG", help="JSON configuration file")

    show = commands.add_parser("show-config", help="print the resolved/effective JSON configuration")
    show.add_argument("config", type=Path, metavar="CONFIG", help="JSON configuration file")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "show-config":
            effective = load_config(args.config)
            json.dump(effective, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0

        written = run_config(args.config)
        if isinstance(written, list):
            for path in written:
                print(path)
        else:
            print(written)
        return 0
    except (ConfigError, RuntimeError) as exc:
        print(f"ticoi: error: {exc}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
