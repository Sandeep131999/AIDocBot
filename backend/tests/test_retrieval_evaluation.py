import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.retrievers import BaseRetriever

from src.agent import nodes
from src.observability import evaluator as evaluator_module
from src.observability.evaluator import Evaluator
from src.retrieval import vector_store
from src.retrieval.document_loader import _load_json, load_and_split
from src.routing import tools as retrieval_tools


def test_json_loader_preserves_record_metadata(tmp_path):
    json_path = tmp_path / "knowledge.json"
    json_path.write_text(json.dumps([
        {"label": "PageOne", "url": "http://example.test/1", "title": "First", "content": "one"},
        {"label": "PageTwo", "url": "http://example.test/2", "title": "Second", "content": "two"},
    ]), encoding="utf-8")

    records = _load_json(str(json_path))

    assert [record.metadata["document_id"] for record in records] == ["PageOne", "PageTwo"]
    assert records[0].metadata["url"] == "http://example.test/1"
    assert records[0].metadata["title"] == "First"
    assert records[0].page_content.startswith("First\nPageOne")

    chunks = load_and_split(str(json_path))
    assert {chunk.metadata["document_id"] for chunk in chunks} == {"PageOne", "PageTwo"}
    assert all("label" in chunk.metadata and "url" in chunk.metadata for chunk in chunks)


def test_retriever_tool_returns_documents_as_artifact(monkeypatch):
    document = Document(page_content="content", metadata={"document_id": "PageOne"})

    class FakeRetriever(BaseRetriever):
        docs: list[Document]

        def _get_relevant_documents(self, query, *, run_manager):
            return self.docs

    retriever = FakeRetriever(docs=[document])
    monkeypatch.setattr(retrieval_tools, "get_retriever", lambda k: retriever)
    tool = retrieval_tools.get_retriever_tool()

    content, artifact = tool.func(query="page")

    assert "content" in content
    assert artifact == [document]


def test_hit_precision_recall_and_ranking_deduplicate_chunks():
    metrics = Evaluator().evaluate_query(
        retrieved=["page-a", "page-a", "page-b", "page-c"],
        relevant=["page-a", "page-b"],
        k=5,
    )

    assert metrics["hit_at_k"] == 1.0
    assert metrics["precision_at_k"] == 0.4
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == 1.0


def test_rrf_keeps_distinct_records_from_same_file():
    docs = [
        Document(
            page_content="first",
            metadata={"file_hash": "same-file", "document_id": "page-a"},
        ),
        Document(
            page_content="second",
            metadata={"file_hash": "same-file", "document_id": "page-b"},
        ),
    ]

    fused = vector_store._rrf_fusion(docs, [])

    assert {doc.metadata["document_id"] for doc in fused} == {"page-a", "page-b"}


def test_grader_captures_retriever_tool_artifact(monkeypatch):
    document = Document(
        page_content="A book can be borrowed for 30 days.",
        metadata={"document_id": "SCIILibrary"},
    )

    class FakeGrader:
        async def ainvoke(self, _messages):
            return nodes.GradeResult(relevant=True, confidence=0.95, reason="matches")

    monkeypatch.setattr(nodes, "get_grader_llm", lambda: FakeGrader())
    tool_message = ToolMessage(
        content="A book can be borrowed for 30 days.",
        tool_call_id="call-1",
        artifact=[document],
    )
    result = asyncio.run(nodes.grade_documents_node({
        "question": "How long can a book be borrowed?",
        "messages": [tool_message],
        "retrieved_docs": [],
    }))

    assert result["retrieved_docs"] == [document]
    assert result["relevant_docs"] == [document]


def test_golden_dataset_aggregates_at_requested_k(monkeypatch):
    documents = [
        Document(page_content="book rules", metadata={"document_id": "SCIILibrary"}),
        Document(page_content="building info", metadata={"document_id": "BuildingInfo"}),
    ]

    class FakeGraph:
        async def ainvoke(self, _state, config: dict):
            index = int(config["configurable"]["thread_id"].rsplit("-", 1)[1])
            return {"answer": "answer", "retrieved_docs": [documents[index % len(documents)]]}

    monkeypatch.setattr(
        evaluator_module,
        "run_ragas_evaluation",
        AsyncMock(return_value={"faithfulness": 0.9}),
    )
    dataset = [
        {"question": "q1", "ground_truth": "a1", "relevant_doc_ids": ["SCIILibrary"]},
        {"question": "q2", "ground_truth": "a2", "relevant_doc_ids": ["BuildingInfo"]},
    ]

    result = asyncio.run(evaluator_module.golden_dataset_eval(dataset, graph=FakeGraph(), k=5))

    assert result["num_questions"] == 2
    assert result["k"] == 5
    assert result["ragas"]["faithfulness"] == 0.9
    assert result["retrieval"]["hit_at_k"]["mean"] == 1.0


def test_golden_dataset_ids_match_json_record_labels():
    root = Path(__file__).resolve().parents[1]
    dataset = json.loads(
        (root / "evaluations" / "knowledge_base_golden.json").read_text(encoding="utf-8")
    )
    expected_ids = {
        document_id
        for item in dataset
        for document_id in item["relevant_doc_ids"]
    }
    records = _load_json(str(root / "uploads" / "knowledge.json"))

    assert expected_ids <= {record.metadata["document_id"] for record in records}