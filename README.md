# Disney+ Hotstar Reviews — Dashboard & Notebooks

This repository contains lexicon‑labeled App Store and Play Store reviews for Disney+ Hotstar plus a Streamlit dashboard to explore preprocessing, sentiment labels, and model evaluation results.

Files of interest
- `dashboard.py` — Streamlit app that loads the CSVs, lets you filter by platform/date/sentiment, inspects preprocessing steps, and compares model results (SVM + TF‑IDF and SVM + IndoBERT).
- `lex_labeled_review_app.csv`, `lex_labeled_review_play.csv` — source datasets.
- `Tesis_Appstore_FIX.ipynb`, `Tesis_Playstore_FIX.ipynb` — notebooks used to preprocess and train models. They include helper cells to export results as JSON.
- `exported_model_results_app.json`, `exported_model_results_play.json` — sample notebook-exported models (if present in the repo).

Quick start (Windows PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv

Lexicon-labeled Disney+ Hotstar reviews from the App Store and Play Store, plus a Streamlit dashboard for exploring preprocessing steps, sentiment distributions, and model evaluation results.

## Repository Layout

- `dashboard.py` – thin entry point that imports and runs `src.dashboard.main()` so existing commands keep working.
- `src/dashboard.py` – full Streamlit application (data loading, filters, visualizations, model comparison).
- `data/lex_labeled_review_app.csv`, `data/lex_labeled_review_play.csv` – cleaned review datasets with lexicon-derived sentiment labels.
- `outputs/exported_model_results_app.json`, `outputs/exported_model_results_play.json` – JSON exports produced by the notebooks (GridSearchCV summaries, metrics, confusion matrices when available).
- `notebooks/Tesis_Appstore_FIX.ipynb`, `notebooks/Tesis_Playstore_FIX.ipynb` – preprocessing + modeling workflows; include helper cells for exporting JSON summaries.
- `.streamlit/config.toml` – enabling Streamlit autoreload on save.
- `run_dashboard.ps1` – PowerShell helper that activates `.venv` (if present) and launches the dashboard.
- `requirements.txt` – minimal dependency list used for local runs.
- `combined_reviews.csv` – optional merged dataset (App + Play) kept for reference.

## Quick Start (Windows PowerShell)

```powershell
cd C:\Users\Lenovo\Downloads\hasil-tesis
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Launch Streamlit (defaults to port 8501; override if you like)
streamlit run dashboard.py --server.port 8502

# Or rely on the helper script, which activates .venv when available
.\run_dashboard.ps1
- IndoBERT embedding + SVM is heavy: use the notebook in Colab to run embeddings/GridSearchCV, export results to JSON, then load into the dashboard to visualize results without recomputing.
Open http://localhost:8502 in your browser once the app reports that it is running. Press `Ctrl+C` in the terminal to stop Streamlit.

### Dependencies

- Core dashboard: `pandas`, `numpy`, `streamlit`, `plotly`, `matplotlib`, `wordcloud`, `scikit-learn`, `nltk`, `sastrawi`.
- IndoBERT workflow: `transformers` and `torch` (large downloads; GPU recommended). The dashboard detects the packages and warns when they are absent.

## Using Notebook Exports

1. Train or tune models in the notebooks (often via Google Colab for IndoBERT).
2. Run the provided export cell to produce `exported_model_results_*.json` files.
3. Copy the JSON files into `outputs/` (the dashboard also falls back to the repo root if needed).
4. In the dashboard open **Model Performance Comparison → Import precomputed model results (JSON)**.
5. Use **Preview** to inspect the JSON summary or **Load** to inject it into the TF-IDF or IndoBERT result panes. Loaded results stay in `st.session_state` until you refresh the app.

The importer normalizes GridSearchCV exports. When `classification_report` or `confusion_matrix` is missing, the UI explains why and falls back to showing cross-validation summaries.

## Data Refresh Workflow

1. Update the notebooks and re-export `lex_labeled_review_*.csv` and JSON summaries.
2. Replace the files under `data/` and `outputs/` with the newest artifacts.
3. Restart the dashboard (or let Streamlit autoreload) to see the latest data.

## Troubleshooting

- **Port already in use** – run `streamlit run dashboard.py --server.port 8601` (or another free port).
- **Large dependency installs** – omit `transformers`/`torch` if you only plan to view precomputed IndoBERT results.
- **Auto reload not working** – ensure you are running Streamlit >= 1.25 and keep `.streamlit/config.toml` intact.

Generated on 2025-11-01 – aligns with the reorganized repository layout.
- `lex_labeled_review_play.csv` – Play Store reviews preprocessed with the same pipeline.
