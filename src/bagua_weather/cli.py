"""Command-line entry point: `bagua fetch|fit|eval`."""

from __future__ import annotations

import argparse
import json

from .ingest import STATIONS, UTC_ERA_START, fetch_station, load_hourly


def _station_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--station", default="01443", help="DWD station id (default: Freiburg)")
    p.add_argument("--cache", default="data/raw", help="raw-data cache dir")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bagua", description="8-state weather regime model")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="download DWD data for a station")
    _station_args(p_fetch)

    p_fit = sub.add_parser("fit", help="classify states and fit the Markov model")
    _station_args(p_fit)
    p_fit.add_argument("--out", default="data/processed", help="output dir for matrices")
    p_fit.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing")
    p_fit.add_argument("--since", default=UTC_ERA_START, help="UTC start cutoff")

    p_eval = sub.add_parser("eval", help="evaluate against baselines and write a report")
    _station_args(p_eval)
    p_eval.add_argument("--split", default="2016-01-01", help="chronological train/test split")
    p_eval.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing")
    p_eval.add_argument("--since", default=UTC_ERA_START, help="UTC start cutoff")
    p_eval.add_argument("--reports", default="reports", help="report output dir")

    args = parser.parse_args(argv)
    meta = STATIONS.get(args.station)
    if meta is None:
        known = ", ".join(sorted(STATIONS))
        parser.error(f"station {args.station!r} has no verified coordinates (known: {known}); "
                     "add it to ingest.STATIONS from its Metadaten_Geographie file")

    if args.cmd == "fetch":
        found = fetch_station(args.station, cache_dir=args.cache)
        total = sum(len(v) for v in found.values())
        print(f"fetched {total} produkt files for station {args.station} into {args.cache}")
        for product, paths in sorted(found.items()):
            print(f"  {product}: {len(paths)} file(s)")
        return

    if args.cmd == "fit":
        from .fit import fit_pipeline, save_fit

        df = load_hourly(args.station, cache_dir=args.cache, since=args.since)
        fit = fit_pipeline(df, lat=meta["lat"], lon=meta["lon"], alpha=args.alpha)
        save_fit(fit, args.out)
        print(json.dumps(fit["coverage"], indent=2))
        print(f"matrices + report written to {args.out}")
        return

    if args.cmd == "eval":
        from .report import run_eval

        df = load_hourly(args.station, cache_dir=args.cache, since=args.since)
        summary = run_eval(df, lat=meta["lat"], lon=meta["lon"], split=args.split,
                           alpha=args.alpha, reports_dir=args.reports)
        print(json.dumps(summary, indent=2))
        return


if __name__ == "__main__":
    main()
