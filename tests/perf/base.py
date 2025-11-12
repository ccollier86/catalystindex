from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from catalystindex.chunking.engine import ChunkingEngine
from catalystindex.embeddings.hash import HashEmbeddingProvider
from catalystindex.models.common import Tenant
from catalystindex.parsers.registry import default_registry
from catalystindex.policies.resolver import resolve_policy
from catalystindex.services.ingestion import IngestionService
from catalystindex.services.search import SearchService
from catalystindex.storage.term_index import InMemoryTermIndex
from catalystindex.storage.vector_store import InMemoryVectorStore
from catalystindex.telemetry.logger import AuditLogger, MetricsRecorder


@dataclass(slots=True)
class PerfContext:
    """Shared objects for performance scripts."""

    tenant: Tenant
    ingestion: IngestionService
    search: SearchService
    vector_store: InMemoryVectorStore
    term_index: InMemoryTermIndex
    metrics: MetricsRecorder


SAMPLE_DOCUMENTS: Sequence[dict[str, object]] = (
    {
        "document_id": "ptsd-criteria",
        "title": "DSM-5 PTSD Criteria",
        "schema": "dsm5",
        "policy": "DSM Criteria",
        "content": (
            "Criterion A. Exposure to actual or threatened death, serious injury, or sexual violence."\
            " Criterion B. Intrusion symptoms including intrusive memories, distressing dreams, and flashbacks."\
            " Criterion C. Persistent avoidance of stimuli associated with the traumatic event."\
            " Criterion D. Negative alterations in cognitions and mood related to the event."\
            " Criterion E. Marked alterations in arousal and reactivity such as hypervigilance."\
        ),
    },
    {
        "document_id": "cbt-planner",
        "title": "CBT Skills Treatment Planner",
        "schema": "treatment_planner",
        "policy": "Treatment Plan Template",
        "content": (
            "Session overview highlighting behavioral activation and cognitive restructuring modules."\
            " Includes exposure hierarchy building, relaxation training, and homework assignments."\
            " Documents measurable goals, progress indicators, and clinician interventions for anxiety management."\
        ),
    },
    {
        "document_id": "stabilization-handbook",
        "title": "Trauma Stabilization Handbook",
        "schema": "psychoeducation",
        "policy": "Trauma Stabilization",
        "content": (
            "Grounding skills include paced breathing, five-senses exercises, and orientation statements."\
            " Crisis response sections reinforce safety planning and hotline escalation paths."\
            " Psychoeducation covers avoidance cycles, intrusion management, and somatic regulation techniques."\
        ),
    },
)

SAMPLE_QUERIES: Sequence[str] = (
    "What are the PTSD diagnostic criteria?",
    "How do we structure behavioral activation homework?",
    "Which grounding techniques support trauma stabilization?",
)


def build_perf_context(namespace: str = "perf") -> PerfContext:
    """Create shared ingestion/search services backed by in-memory dependencies."""

    metrics = MetricsRecorder(namespace=f"perf-{namespace}")
    audit_logger = AuditLogger()
    vector_store = InMemoryVectorStore()
    term_index = InMemoryTermIndex()
    embedding_provider = HashEmbeddingProvider(dimension=384)
    parser_registry = default_registry()
    chunking_engine = ChunkingEngine(namespace=namespace)

    ingestion = IngestionService(
        parser_registry=parser_registry,
        chunking_engine=chunking_engine,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        term_index=term_index,
        audit_logger=audit_logger,
        metrics=metrics,
    )

    search = SearchService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        term_index=term_index,
        audit_logger=audit_logger,
        metrics=metrics,
        economy_k=8,
        premium_k=16,
        enable_sparse_queries=False,
        premium_rerank_enabled=False,
    )

    tenant = Tenant(org_id="perf-org", workspace_id="demo-workspace", user_id="load-test")
    return PerfContext(
        tenant=tenant,
        ingestion=ingestion,
        search=search,
        vector_store=vector_store,
        term_index=term_index,
        metrics=metrics,
    )


def load_sample_corpus(context: PerfContext, *, repeats: int = 1) -> List[str]:
    """Ingest curated documents multiple times to prime the vector store."""

    ingested_documents: List[str] = []
    for cycle in range(repeats):
        for spec in SAMPLE_DOCUMENTS:
            document_id = f"{spec['document_id']}-{cycle}"
            policy = resolve_policy(str(spec["policy"]), str(spec["schema"]))
            result = context.ingestion.ingest(
                tenant=context.tenant,
                document_id=document_id,
                document_title=str(spec["title"]),
                content=str(spec["content"]),
                policy=policy,
                parser_name="plain_text",
                document_metadata={"source": "perf-suite"},
            )
            if not result.chunks:
                raise RuntimeError(f"Ingestion returned no chunks for {document_id}")
            ingested_documents.append(document_id)
    return ingested_documents


def cycle_queries(iterations: int) -> Iterable[tuple[str, str]]:
    """Yield (mode, query) pairs cycling through sample queries and modes."""

    modes = ("economy", "premium")
    for index in range(iterations):
        mode = modes[index % len(modes)]
        query = SAMPLE_QUERIES[index % len(SAMPLE_QUERIES)]
        yield mode, query


__all__ = [
    "PerfContext",
    "SAMPLE_DOCUMENTS",
    "SAMPLE_QUERIES",
    "build_perf_context",
    "load_sample_corpus",
    "cycle_queries",
]
