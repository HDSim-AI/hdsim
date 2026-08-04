"""Command line entry point.

    hdsim demo          play a recorded negotiation, no key needed
    hdsim demo --list   show the available recordings
    hdsim config        show which model each role is set to use

`demo` is deliberately the first thing in the README. Someone who has just installed the package
can see a household negotiate before deciding whether to configure a provider.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def cmd_demo(args: argparse.Namespace) -> int:
    from . import replay

    if args.list:
        rows = replay.available()
        width = max(len(r["id"]) for r in rows)
        print(f"{'household':<{width}}  members  rounds  result")
        for r in rows:
            print(f"{r['id']:<{width}}  {r['members']:>7}  {r['rounds']:>6}  {r['result']}")
        return 0

    try:
        record = replay.get(args.household)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(replay.render(record, show_personas=args.personas))
    print()
    print("This was a recording. To run a live simulation, set HDSIM_API_KEY and see the README.")
    return 0


def cmd_config(_: argparse.Namespace) -> int:
    from .backends import check

    settings = check()
    width = max(len(r) for r in settings)
    print(f"{'role':<{width}}  model                     endpoint")
    for role, cfg in settings.items():
        print(f"{role:<{width}}  {cfg['model']:<24}  {cfg['base_url']}")
    keys = {cfg["api_key"] for cfg in settings.values()}
    print()
    print("api key: " + ("set" if keys != {"not set"} else "not set"))
    if keys == {"not set"}:
        print("Set HDSIM_API_KEY, or copy .env.example to .env and fill it in.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hdsim",
        description="Household decision simulation through multi-agent negotiation.",
    )
    parser.add_argument("--version", action="version", version=f"hdsim {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="play a recorded negotiation (no API key needed)")
    demo.add_argument("household", nargs="?", default=None, help="recording id")
    demo.add_argument("--list", action="store_true", help="list available recordings")
    demo.add_argument("--personas", action="store_true", help="also print member personas")
    demo.set_defaults(func=cmd_demo)

    config = sub.add_parser("config", help="show the model configured for each role")
    config.set_defaults(func=cmd_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
