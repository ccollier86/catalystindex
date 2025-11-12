from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from catalyst_index_sdk.client import CatalystIndexClient


def build_client(args: argparse.Namespace) -> CatalystIndexClient:
    if not args.base_url or not args.token:
        raise SystemExit("--base-url and --token are required")
    return CatalystIndexClient(base_url=args.base_url, token=args.token, timeout=args.timeout)


def cmd_telemetry(args: argparse.Namespace) -> None:
    client = build_client(args)
    metrics = client.get_telemetry_metrics()
    payload = asdict(metrics)
    print(json.dumps(payload, indent=2, default=str))


def cmd_ingest_status(args: argparse.Namespace) -> None:
    client = build_client(args)
    status = client.get_ingestion_job_status(args.job_id)
    print(json.dumps(asdict(status), indent=2))


def cmd_search(args: argparse.Namespace) -> None:
    client = build_client(args)
    response = client.search(
        query=args.query,
        mode=args.mode,
        limit=args.limit,
        tracks=None,
        filters=None,
        alias={"enabled": args.alias_enabled, "limit": args.alias_limit},
        debug=args.debug,
    )
    payload = {
        "mode": response.mode,
        "tracks": response.tracks,
        "results": [asdict(result) for result in response.results],
        "debug": asdict(response.debug) if response.debug else None,
    }
    print(json.dumps(payload, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Catalyst Index CLI helper")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    telemetry_parser = subparsers.add_parser("telemetry", help="Show telemetry metrics")
    telemetry_parser.set_defaults(func=cmd_telemetry)

    status_parser = subparsers.add_parser("ingest-status", help="Inspect ingestion job status")
    status_parser.add_argument("job_id")
    status_parser.set_defaults(func=cmd_ingest_status)

    search_parser = subparsers.add_parser("search", help="Execute a search query")
    search_parser.add_argument("query")
    search_parser.add_argument("--mode", default="economy")
    search_parser.add_argument("--limit", type=int, default=5)
    search_parser.add_argument("--alias-enabled", action="store_true", default=False)
    search_parser.add_argument("--alias-limit", type=int, default=5)
    search_parser.add_argument("--debug", action="store_true")
    search_parser.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
