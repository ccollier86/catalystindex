from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import redis  # type: ignore
    from rq import Queue, Worker
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("rq worker dependencies are not installed. Install the 'workers' extra.") from exc

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
    redis_url = os.getenv("CATALYST_JOBS__store__redis_url", "redis://localhost:6379/1")
    queue_name = args.queue or os.getenv("CATALYST_JOBS__worker__queue_name", "ingestion")
    connection = redis.from_url(redis_url)
    queue = Queue(queue_name, connection=connection)
    worker = Worker([queue])
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
