"""Command-line entry point: `bagua fetch|fit|eval`."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bagua", description="8-state weather regime model")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, helptext in [
        ("fetch", "download DWD data for a station"),
        ("fit", "classify states and fit the Markov model"),
        ("eval", "evaluate against baselines and write a report"),
    ]:
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--station", default="01443", help="DWD station id (default: Freiburg)")
    args = parser.parse_args(argv)
    raise NotImplementedError(f"`bagua {args.cmd}` is not implemented yet (see README milestones)")


if __name__ == "__main__":
    main()
