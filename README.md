# RAG Evaluation System

A deployable **RAG retrieval and answer evaluation** dashboard. Runs with free offline metrics; optional **Google Gemini** answers and **RAGAS** LLM-based context precision, context recall, and faithfulness. Created as a replacement package for a previously shared ZIP whose original bytes were not accessible.

## Stack
Python 3.11, Streamlit, Pandas, stdlib TF-IDF, Pytest, GitHub Actions, optional Google GenAI and RAGAS.

## Mac setup
```bash
cd ~/Downloads
unzip -o rag-evaluation-system.zip -d ~/Downloads
cd ~/Downloads/rag-evaluation-system
conda create -n rag-eval python=3.11 -y
conda activate rag-eval
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m streamlit run app.py
```

## Workflow
1. Start the app and click **Run evaluation** (works with no API key).
2. Replace the CSV files via uploads to evaluate a different RAG dataset.
3. Optional: `python -m pip install -r requirements-gemini.txt` for Gemini generation; supply a Gemini API key in sidebar or Streamlit secrets.
4. Optional: `python -m pip install -r requirements-ragas.txt` for RAGAS judging. The extra may increase deployment size; use selectively.
5. Download the evaluation CSV report.

## Data contracts
`data/documents.csv` columns: `doc_id,text`. Unique document IDs and one text chunk per row.

`data/questions.csv` columns: `question,reference_answer,relevant_doc_ids` and optional `response`. Use pipe-separated IDs when multiple relevant chunks exist: `doc1|doc2`. Uploaded responses are evaluated as-is in the **Existing responses** mode; otherwise the offline demo extracts a sentence from retrieved context. No fabricated ground-truth answer is passed off as a generated answer.

## Metrics
- **Hit@k**, **Precision@k**, **Recall@k**, **MRR**: document-ID labels, deterministic retrieval scoring.
- **Answer token F1**: lexical word overlap with reference answer (not an LLM evaluation).
- **Lexical grounding**: overlap with retrieved evidence (not factual faithfulness).
- **RAGAS** (optional with Gemini): LLM-judged Context Precision, Context Recall, Faithfulness. **These are separate scores**; offline metrics are never called RAGAS results.

## GitHub + deployment
```bash
git init
git add .
git commit -m "Build RAG evaluation dashboard with tests and optional Gemini"
git branch -M main
gh repo create rag-evaluation-system --public --source=. --remote=origin --push
```
If `gh` is unavailable, create an empty public GitHub repo in the browser, then run:
```bash
git remote add origin https://github.com/YOUR_USERNAME/rag-evaluation-system.git
git push -u origin main
```
Open **GitHub → Actions → Python tests** and confirm tests pass. On https://share.streamlit.io choose **New app**, your repository, branch `main`, and entrypoint `app.py`. For optional Gemini use the app's **Advanced settings → Secrets** with `GOOGLE_API_KEY = "..."` (never commit a real key).

## Limits and safety
- Uploaded documents may contain private data; Gemini mode sends retrieved document excerpts and questions to Google. Use anonymized public test data for portfolio demos.
- No data is written to disk by the web app. Do not upload confidential documents on a public shared app.
- Free API quotas may change; API calls are never made in offline mode.
- Optional RAGAS metrics depend on API availability, package versions, and model quota; no claim of running live RAGAS judging without verification.
