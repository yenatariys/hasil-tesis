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
