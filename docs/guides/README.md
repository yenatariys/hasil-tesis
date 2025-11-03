# Disney+ Hotstar Reviews Dashboard

Streamlit dashboard plus notebooks for exploring lexicon-labelled Disney+ Hotstar reviews from the App Store and Play Store. The app mirrors the preprocessing steps from the research notebooks, visualises sentiment distributions, and lets you import notebook-generated model metrics.

## Repository Layout

- `dashboard.py` – thin entry point that forwards to `src.dashboard.main()`.
- `src/dashboard.py` – full Streamlit application with filters, playground, and visualisations.
- `data/lex_labeled_review_app.csv`, `data/lex_labeled_review_play.csv` – main datasets used by the dashboard.
- `data/positive.tsv`, `data/negative.tsv` – lexicon weights used in the preprocessing playground.
- `notebooks/Tesis_Appstore_FIX.ipynb`, `notebooks/Tesis_Playstore_FIX.ipynb` – end-to-end preprocessing and modelling workflows.
- `outputs/exported_model_results_*.json` – optional GridSearchCV exports loaded by the dashboard for comparison views.
- `run_dashboard.ps1` – helper script that activates `.venv` (when present) and launches Streamlit.

## Dashboard Highlights

- **Sentiment overview** – stacked bars surface platform-level review counts with inline review/percentage labels.
- **Trend explorer** – monthly line chart plus a period-comparison tab (2020–2022 vs 2023–2025) with counts and percentage annotations.
- **Platform evaluation** – MAE/RMSE/correlation table accompanied by a counts-only rating-consistency heatmap with in-cell totals; download the metrics for single-platform views as CSV.
- **Model performance** – load precomputed TF-IDF or IndoBERT JSON results, toggle whether imports override retraining, and retrain either pipeline directly from the UI.
- **Prediction playground & filtered reviews** – test sentences against any loaded pipeline and inspect the curated review table (date, platform, rating, sentiment, original and processed text).

## Prerequisites

- Python 3.10 or newer (project tested on Python 3.12).
- Windows PowerShell 5.1 or PowerShell 7 for the commands below.
- Git (optional) if you plan to clone this repository instead of downloading a ZIP.

## Setup (Windows PowerShell)

```powershell
cd C:\Users\Lenovo\Downloads\hasil-tesis
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

To launch the dashboard once dependencies are installed:

```powershell
streamlit run dashboard.py
```

The default Streamlit port is 8501. Provide `--server.port 8502` (or another free port) if you need to avoid conflicts. Alternatively, run the helper script:

```powershell
.\run_dashboard.ps1
```

## Optional Dependencies

- `nltk`, `sastrawi`, and `wordcloud` enrich the preprocessing playground; install them via `pip install nltk Sastrawi wordcloud` if they are missing.
- `transformers` and `torch` are only required when you want to recompute IndoBERT embeddings or fine-tune models locally. The dashboard works without them and will surface clear notices when advanced features are unavailable.

## Working With Notebooks

Use the notebooks under `notebooks/` to retrain models, regenerate lexicon-labelled datasets, and export JSON summaries. Each notebook contains cells that produce `lex_labeled_review_*.csv` and `exported_model_results_*.json`. Place the refreshed CSVs in `data/` and JSON exports in `outputs/` before restarting the dashboard.

## Troubleshooting

- **Missing optional packages** – the dashboard falls back to simulated behaviour and surfaces callouts; install the optional packages listed above for full functionality.
- **WordCloud import error** – remove the word cloud panel or install `wordcloud` via `pip install wordcloud`.
- **Large dependency downloads** – skip `transformers` and `torch` if you only plan to visualise precomputed results.
- **Auto-reload disabled** – ensure `.streamlit/config.toml` is present so Streamlit watches files for changes.

## Helpful Commands

- Update Python packages: `pip install --upgrade -r requirements.txt`.
- Clean Streamlit cache: `streamlit cache clear`.
- Run unit checks without starting the UI: `python -m compileall src`.
