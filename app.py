"""Streamlit Community Cloud entrypoint."""
import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from rag_eval.core import aggregate, evaluate, read_csv

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="RAG Evaluation System", page_icon="📊", layout="wide")
st.title("RAG Evaluation System")
st.caption("Evaluate retrieval quality and answer grounding with a free offline baseline; optional Gemini and RAGAS judging.")


def configured_api_key():
    if os.environ.get("GOOGLE_API_KEY"):
        return os.environ["GOOGLE_API_KEY"]
    try:
        return st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:  # Streamlit raises if no local secrets file exists
        return ""


with st.sidebar:
    st.header("Configuration")
    k = st.slider("Top-k documents", 1, 8, 3)
    mode = st.selectbox("Answer source", ["Offline extractive demo", "Existing responses in question CSV", "Gemini generation"])
    api_key = st.text_input("Gemini API key (optional)", value=configured_api_key(), type="password", help="Your key is not saved to files or GitHub")
    model = st.text_input("Gemini model", value=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    st.divider()
    st.caption("Offline metrics are transparent lexical proxies, not LLM or RAGAS grades.")

c1, c2 = st.columns(2)
with c1:
    uploaded_docs = st.file_uploader("Upload documents.csv (optional)", type=["csv"], key="docs")
with c2:
    uploaded_cases = st.file_uploader("Upload questions.csv (optional)", type=["csv"], key="questions")
with st.expander("CSV format"):
    st.code("documents.csv: doc_id,text\nquestions.csv: question,reference_answer,relevant_doc_ids,response (optional)\nrelevant_doc_ids: doc_1|doc_2", language="text")

try:
    docs = read_csv(uploaded_docs if uploaded_docs else ROOT / "data/documents.csv")
    cases = read_csv(uploaded_cases if uploaded_cases else ROOT / "data/questions.csv")
except (ValueError, UnicodeDecodeError, OSError) as exc:
    st.error(f"Could not load CSV: {exc}")
    st.stop()

st.write(f"Loaded **{len(docs)} documents** and **{len(cases)} evaluation questions**.")
with st.expander("Preview data"):
    st.dataframe(pd.DataFrame(docs), use_container_width=True)
    st.dataframe(pd.DataFrame(cases), use_container_width=True)

if st.button("Run evaluation", type="primary"):
    try:
        generator = None
        if mode == "Gemini generation":
            from rag_eval.gemini import make_generator
            generator = make_generator(api_key, model)
        with st.spinner("Evaluating questions..."):
            rows = evaluate(docs, cases, k=k, generate=generator, use_supplied=(mode == "Existing responses in question CSV"))
        st.session_state.results = rows
        st.session_state.meta = {"k": k, "mode": mode}
    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")

if "results" in st.session_state:
    rows = st.session_state.results
    stats = aggregate(rows)
    st.subheader("Evaluation results")
    cols = st.columns(6)
    labels = ["Hit@k", "Precision@k", "Recall@k", "MRR", "Answer token F1", "Lexical grounding*"]
    for col, label, key in zip(cols, labels, stats):
        col.metric(label, f"{stats[key]:.1%}")
    st.caption("*Answer token F1 measures word overlap with references. Lexical grounding measures token overlap with retrieved text; neither establishes factual correctness or true faithfulness.")
    chart = pd.DataFrame(rows).set_index("question")[["hit_at_k", "recall_at_k", "answer_token_f1", "lexical_grounding_proxy"]]
    st.bar_chart(chart)
    visible = [{k: v for k, v in row.items() if k != "retrieved_contexts"} for row in rows]
    st.dataframe(pd.DataFrame(visible), use_container_width=True)
    st.download_button("Download CSV report", data=pd.DataFrame(visible).to_csv(index=False), file_name="rag_evaluation_report.csv", mime="text/csv")

    with st.expander("Optional: run real RAGAS LLM judging on up to 3 cases"):
        st.caption("Requires pip install -r requirements-ragas.txt and Gemini API access. RAGAS calls may incur API usage or charges based on your account.")
        limit = st.number_input("Number of rows", min_value=1, max_value=min(3, len(rows)), value=1)
        if st.button("Run RAGAS judging"):
            try:
                from rag_eval.ragas_judge import judge
                with st.spinner("Calling Gemini for RAGAS judging..."):
                    judged = judge(rows, api_key, model, int(limit))
                st.dataframe(pd.DataFrame(judged), use_container_width=True)
                st.download_button("Download RAGAS CSV", pd.DataFrame(judged).to_csv(index=False), "ragas_scores.csv", "text/csv")
            except Exception as exc:
                st.error(f"RAGAS failed: {exc}")
