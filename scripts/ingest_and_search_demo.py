from __future__ import annotations

import argparse
import time
from typing import Dict

from catalyst_index_sdk.client import CatalystIndexClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a demo document and run a search")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--document-id", default="demo-doc")
    parser.add_argument("--title", default="Demo Criteria Document")
    parser.add_argument("--content", default="Criterion A. Exposure to trauma Criterion B. Intrusion")
    parser.add_argument("--schema", default="dsm5")
    parser.add_argument("--query", default="ptsd trauma criteria")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    return parser.parse_args()


def wait_for_completion(client: CatalystIndexClient, job_id: str, poll_interval: float) -> Dict[str, object]:
    while True:
        status = client.get_ingestion_job_status(job_id)
        if status.status.lower() in {"succeeded", "failed", "partial"}:
            return status.__dict__
        time.sleep(poll_interval)


def main() -> None:
    args = parse_args()
    client = CatalystIndexClient(base_url=args.base_url, token=args.token)
    job = client.ingest_document(
        document_id=args.document_id,
        title=args.title,
        content=args.content,
        schema=args.schema,
    )
    status = wait_for_completion(client, job.job_id, args.poll_interval)
    if status["status"].lower() not in {"succeeded", "partial"}:
        raise SystemExit(f"Ingestion failed: {status}")
    response = client.search(query=args.query, mode="premium", limit=3, debug=True)
    print("Top results:\n")
    for result in response.results:
        print(f"- {result.chunk_id} ({result.chunk_tier}) score={result.score:.3f}")
    if response.debug:
        print("\nDebug info:")
        print(response.debug)


if __name__ == "__main__":
    main()
