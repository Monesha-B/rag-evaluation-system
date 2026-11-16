from pathlib import Path

import pytest
from rag_eval.core import aggregate, answer_f1, evaluate, lexical_grounding, parse_ids, rank_documents, read_csv, validate

DATA = Path(__file__).parents[1] / "data"


def examples():
    return read_csv(DATA / "documents.csv"), read_csv(DATA / "questions.csv")


def test_sample_dataset_valid():
    validate(*examples())


def test_retrieval_finds_refund_policy():
    docs, _ = examples()
    assert rank_documents("How many days can customers return unused products?", docs, 1)[0]["doc_id"] == "refunds"


def test_metrics_exact_match():
    assert answer_f1("Refunds take five days", "Refunds take five days") == 1
    assert lexical_grounding("refund days", "Refund days") == 1
    assert parse_ids("a|b|a") == ["a", "b"]


def test_evaluation_metrics_bounds():
    rows = evaluate(*examples(), k=2)
    assert len(rows) >= 5
    metrics = aggregate(rows)
    assert all(0 <= score <= 1 for score in metrics.values())
    assert metrics["hit_at_k"] > 0


def test_supplied_answer_is_respected():
    docs, cases = examples()
    cases = [dict(cases[0], response="Custom model answer")]
    assert evaluate(docs, cases, use_supplied=True)[0]["answer"] == "Custom model answer"


def test_bad_document_id_rejected():
    docs, cases = examples()
    cases[0]["relevant_doc_ids"] = "does_not_exist"
    with pytest.raises(ValueError, match="Unknown relevant_doc_ids"):
        evaluate(docs, cases)


def test_k_must_be_positive():
    docs, _ = examples()
    with pytest.raises(ValueError, match="k must be"):
        rank_documents("refund", docs, 0)
