from __future__ import annotations

import argparse
import importlib
import sys

try:
    import redis  # type: ignore
    from rq import Connection, Worker
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("rq worker dependencies are not installed. Install the 'workers' extra.") from exc

from catalystindex.api.dependencies import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Catalyst Index ingestion worker")
    parser.add_argument(
        "--queue",
        dest="queue",
        default=None,
        help="Override queue name (defaults to settings.jobs.worker.queue_name)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    redis_url = settings.jobs.store.redis_url
    if not redis_url:
        raise RuntimeError("jobs.store.redis_url must be configured to run the worker.")
    queue_name = args.queue or settings.jobs.worker.queue_name
    connection = redis.from_url(redis_url)
    with Connection(connection):
        worker = Worker([queue_name])
        worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
