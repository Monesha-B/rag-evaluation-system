"""Dependency-free TF-IDF retrieval and transparent evaluation metrics.

Offline lexical scores are proxies and are NOT RAGAS LLM-judge scores.
"""
from __future__ import annotations

import csv
import io
import math
import re
from collections import Counter
from pathlib import Path

STOP = frozenset("a an the is are was were of to for in on and or from with how what when where who do does can i it my our your by at much long be which after will".split())


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if t not in STOP]


def read_csv(source) -> list[dict[str, str]]:
    if hasattr(source, "read"):
        content = source.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8-sig")
    else:
        content = Path(source).read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(content)))
    if not rows:
        raise ValueError("CSV must contain a header and at least one data row")
    return [{str(k).strip(): str(v or "").strip() for k, v in row.items() if k is not None} for row in rows]


def validate(documents: list[dict], cases: list[dict]) -> None:
    if not documents or not cases:
        raise ValueError("At least one document and one question are required")
    for col in ("doc_id", "text"):
        if any(not d.get(col, "").strip() for d in documents):
            raise ValueError(f"Each document needs a nonempty '{col}'")
    ids = [d["doc_id"] for d in documents]
    if len(set(ids)) != len(ids):
        raise ValueError("Document doc_id values must be unique")
    for col in ("question", "reference_answer", "relevant_doc_ids"):
        if any(not c.get(col, "").strip() for c in cases):
            raise ValueError(f"Each question needs a nonempty '{col}'")
    for c in cases:
        expected = set(parse_ids(c["relevant_doc_ids"]))
        if expected - set(ids):
            raise ValueError(f"Unknown relevant_doc_ids for '{c['question']}': {expected - set(ids)}")


def parse_ids(raw: str) -> list[str]:
    return list(dict.fromkeys(p.strip() for p in raw.split("|") if p.strip()))


def rank_documents(question: str, documents: list[dict], k: int = 3) -> list[dict]:
    """Simple cosine TF-IDF retriever with deterministic tie-breaking."""
    if k < 1:
        raise ValueError("k must be >= 1")
    q = Counter(tokens(question))
    bodies = [Counter(tokens(d["text"])) for d in documents]
    n = len(bodies)
    df = Counter(t for bag in bodies for t in bag)
    idf = {t: math.log((n + 1) / (df[t] + 1)) + 1 for t in set(df) | set(q)}

    def norm(bag):
        return math.sqrt(sum((c * idf[t]) ** 2 for t, c in bag.items()))

    qnorm = norm(q)
    scored = []
    for doc, bag in zip(documents, bodies):
        score = (sum(q[t] * bag[t] * idf[t] ** 2 for t in q) / (qnorm * norm(bag))) if qnorm and norm(bag) else 0.0
        scored.append({"doc_id": doc["doc_id"], "text": doc["text"], "similarity": round(score, 5)})
    return sorted(scored, key=lambda d: (-d["similarity"], d["doc_id"]))[:k]


def offline_answer(question: str, retrieved: list[dict]) -> str:
    """Extract best matching evidence sentence; demo only, not an LLM."""
    qt = set(tokens(question))
    candidates = []
    for rank, doc in enumerate(retrieved):
        for sentence in re.split(r"(?<=[.!?])\s+", doc["text"]):
            sentence = sentence.strip()
            if sentence:
                st = set(tokens(sentence))
                candidates.append((len(qt & st), -rank, sentence))
    if not candidates or max(c[0] for c in candidates) == 0:
        return "Not enough evidence found."
    return max(candidates)[2]


def answer_f1(response: str, reference: str) -> float:
    a, b = Counter(tokens(response)), Counter(tokens(reference))
    if not a or not b:
        return 0.0
    overlap = sum((a & b).values())
    p, r = overlap / sum(a.values()), overlap / sum(b.values())
    return round(2 * p * r / (p + r), 4) if p + r else 0.0


def lexical_grounding(response: str, context: str) -> float:
    words = tokens(response)
    return round(sum(t in set(tokens(context)) for t in words) / len(words), 4) if words else 0.0


def evaluate(documents: list[dict], cases: list[dict], k: int = 3, generate=None, use_supplied=False):
    validate(documents, cases)
    rows = []
    for case in cases:
        retrieved = rank_documents(case["question"], documents, k)
        relevant = set(parse_ids(case["relevant_doc_ids"]))
        ids = [r["doc_id"] for r in retrieved]
        hits = relevant.intersection(ids)
        reciprocal = next((1.0 / (idx + 1) for idx, doc_id in enumerate(ids) if doc_id in relevant), 0.0)
        context = "\n\n".join(f"[{d['doc_id']}] {d['text']}" for d in retrieved)
        if use_supplied and case.get("response"):
            response = case["response"]
        elif generate:
            response = generate(case["question"], retrieved)
        else:
            response = offline_answer(case["question"], retrieved)
        rows.append({
            "question": case["question"], "reference_answer": case["reference_answer"],
            "answer": response, "retrieved_ids": " | ".join(ids),
            "relevant_ids": " | ".join(sorted(relevant)),
            "hit_at_k": int(bool(hits)),
            "precision_at_k": round(len(hits) / len(ids), 4) if ids else 0.0,
            "recall_at_k": round(len(hits) / len(relevant), 4),
            "reciprocal_rank": round(reciprocal, 4),
            "answer_token_f1": answer_f1(response, case["reference_answer"]),
            "lexical_grounding_proxy": lexical_grounding(response, context),
            "retrieved_contexts": [d["text"] for d in retrieved],
        })
    return rows


def aggregate(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    fields = ("hit_at_k", "precision_at_k", "recall_at_k", "reciprocal_rank", "answer_token_f1", "lexical_grounding_proxy")
    return {f: round(sum(float(r[f]) for r in rows) / len(rows), 4) for f in fields}
