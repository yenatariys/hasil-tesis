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
.\.venv\Scripts\Activate.ps1
```

2. Install required packages (minimal; adjust as needed):

```powershell
pip install pandas numpy streamlit plotly scikit-learn matplotlib wordcloud nltk sastrawi
# Optional (heavy): transformers torch
```

3. Run the dashboard on port 8502:

```powershell
# Option A: simple run
streamlit run dashboard.py --server.port 8502

# Option B: recommended — enables auto-reload on save (if your Streamlit version supports runOnSave)
# Using the project config (.streamlit/config.toml) the app will auto-reload when files are saved.
.\n+# Or use the included helper script which will activate a local venv if present:
.\run_dashboard.ps1
```

Open http://localhost:8502 in your browser.

Notes on dependencies
- Light / interactive features: `pandas`, `streamlit`, `plotly`, `scikit-learn`, `matplotlib`, `wordcloud`, `nltk`, `sastrawi`.
- IndoBERT embeddings require `transformers` and `torch` (large installs and GPU recommended). The dashboard guards IndoBERT features and will show a warning if these packages are not installed.

Using exported notebook results (recommended for speed)
- The notebooks include cells that export model search/evaluation results to JSON files named `exported_model_results_app.json` and `exported_model_results_play.json`.
- Place such JSON files in the same directory as `dashboard.py` (the app will auto-detect files matching `exported_model_results_*.json`).
- In the dashboard go to "Model Performance Comparison" → Expand "Import precomputed model results (JSON)". Detected workspace JSONs will be listed with Preview and Load buttons. Loading places the precomputed results into the TF‑IDF or IndoBERT model slots (heuristics used; you can retrain in-app to override).

How the dashboard handles heavy models
- TF‑IDF + Linear SVM is reasonably fast and can be run inside the dashboard.
- IndoBERT embedding + SVM is heavy: use the notebook in Colab to run embeddings/GridSearchCV, export results to JSON, then load into the dashboard to visualize results without recomputing.

Notebook export helper (Colab)
- The notebooks include a Colab helper cell that mounts Google Drive, copies the exported JSON(s) to Drive, and triggers downloads. Run that cell in Colab after training to copy results locally.

Troubleshooting
- Port conflict: If port 8502 is already in use, pick a different port using `--server.port <PORT>`.
- If the dashboard warns about missing IndoBERT dependencies, install `transformers` and `torch`, or produce the JSON from the notebook and load it instead.

Next steps you might want
- Auto-load workspace JSONs on app start (opt-in) — the dashboard already detects workspace JSONs and offers preview/load.
- Synchronize model hyperparameter defaults in `dashboard.py` with the notebook experiments — I can update those if you want.

If you'd like, I can also add a `requirements.txt` or a small PowerShell script to create the venv and install packages automatically.

---
Generated on 2025-11-01 — created to make running and reproducing the dashboard easier.
# Disney+ Hotstar Sentiment Dashboard

Interactive Streamlit dashboard for exploring lexicon-based sentiment analysis of Disney+ Hotstar reviews collected from the App Store and Play Store. The app also exposes the Natural Language Processing (NLP) pipeline steps and lets you benchmark SVM classifiers trained on TF-IDF features as well as IndoBERT embeddings.

## Repository Layout

- `lex_labeled_review_app.csv` – App Store reviews with preprocessing outputs and lexicon sentiment labels.
- `lex_labeled_review_play.csv` – Play Store reviews preprocessed with the same pipeline.
- `combined_reviews.csv` – Optional merged dataset (App + Play) created by `combine_reviews.py`.
- `combine_reviews.py` – Helper script to regenerate `combined_reviews.csv` with a `Platform` column.
- `dashboard.py` – Streamlit application that powers the visual dashboard.
- `requirements.txt` – Python dependencies for data handling, visualization, and model benchmarking.
- `tesis_appstore_fix.py`, `tesis_playstore_fix.py`, `Tesis_Appstore_FIX.ipynb`, `Tesis_Playstore_FIX.ipynb` – Research notebooks/scripts detailing the preprocessing and modeling experiments used to generate the CSVs.

## Quick Start

1. **Set up environment**
   ```powershell
   cd C:\Users\Lenovo\Downloads\hasil-tesis
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Update combined dataset (optional)**
   If you have refreshed `lex_labeled_review_app.csv` and `lex_labeled_review_play.csv`, regenerate the merged file:
   ```powershell
   python combine_reviews.py
   ```

3. **Launch the dashboard**
   ```powershell
   python -m streamlit run dashboard.py --server.port 8600
   ```
   Open the app in your browser at `http://localhost:8600`. The Streamlit process must remain running; press `Ctrl+C` in the terminal to stop it.

## Dashboard Highlights

- **Filters** – Slice reviews by platform, sentiment class, rating range, review date, or keywords in either original or translated text.
- **KPI Cards** – Total reviews, average rating, positive-share percentage, and date of the latest review within the current selection.
- **Rating & Sentiment Visuals**
  - Grouped bar chart comparing rating counts by platform.
  - Sentiment breakdown by platform.
  - Monthly sentiment trend lines.
- **Play Store Evaluation Summary** – MAE, RMSE, Pearson, and Spearman correlation between original ratings and lexicon-generated ratings, plus scatter and heatmap visualizations.
- **Pre-processing Explorer** – Step-by-step view of the selected review across translation, cleaning, tokenization, stopword removal, stemming, and final `ulasan_bersih`, including token-count evolution.
- **Model Performance Comparison**
  - **SVM + TF-IDF**: Runs cross-validated grid search over n-gram ranges and C values, showing the best macro-F1, classification report, confusion matrix, and n-gram summary table.
  - **SVM + IndoBERT**: Generates CLS embeddings with IndoBERT (configurable checkpoint, max token length, and batch size), tunes C for Linear SVM, and reports metrics. Requires `transformers` and `torch`.
- **Wordclouds & Keyword Analysis** – Frequency-based insights drawn from cleaned and translated text.
- **Review Table** – Filtered review snapshots including original, translated, and lexicon scores.

## Updating Source Data

1. Run the preprocessing notebooks/scripts (`Tesis_Appstore_FIX.ipynb`, `Tesis_Playstore_FIX.ipynb`) to produce refreshed `lex_labeled_review_app.csv` and `lex_labeled_review_play.csv`.
2. Re-run `combine_reviews.py` if the combined dataset is required.
3. Restart the Streamlit app to load the latest data.

## Troubleshooting

- **Missing dependencies** – Rerun `pip install -r requirements.txt` inside your active virtual environment.
- **Port in use** – Specify a different port, e.g. `--server.port 8601`.
- **IndoBERT training disabled** – Install `torch` and `transformers`, or use the TF-IDF model tab if GPU resources are limited.

## License

This project is intended for academic research on sentiment analysis. Adapt or extend as needed for your use case.
