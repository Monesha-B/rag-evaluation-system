"""Optional genuine RAGAS LLM metrics; requires extra dependencies and Gemini API calls."""
from __future__ import annotations


def judge(rows: list[dict], api_key: str, model: str = "gemini-2.5-flash", limit: int = 3) -> list[dict]:
    try:
        from google import genai
        from ragas.llms import llm_factory
        from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness
    except ImportError as exc:
        raise RuntimeError("Install optional RAGAS dependencies: pip install -r requirements-ragas.txt") from exc
    if not api_key:
        raise ValueError("Gemini key required for RAGAS metrics")
    client = genai.Client(api_key=api_key)
    llm = llm_factory(model, provider="google", client=client)
    scorers = {
        "ragas_context_precision": ContextPrecision(llm=llm),
        "ragas_context_recall": ContextRecall(llm=llm),
        "ragas_faithfulness": Faithfulness(llm=llm),
    }
    output = []
    for row in rows[:limit]:
        inputs = {"user_input": row["question"], "retrieved_contexts": row["retrieved_contexts"], "reference": row["reference_answer"], "response": row["answer"]}
        result = {"question": row["question"]}
        for name, metric in scorers.items():
            param_keys = ("response", "retrieved_contexts") if name == "ragas_faithfulness" else ("user_input", "retrieved_contexts", "reference")
            params = {key: inputs[key] for key in param_keys}
            # Current RAGAS collections API supports synchronous .score().
            score = metric.score(**params)
            result[name] = round(float(score.value), 4)
        output.append(result)
    return output
