# Project Structure Documentation

## Directory Layout

### 📁 `/data`
Contains all data files used in the project.

- **`/raw`** - Original, unprocessed datasets
- **`/processed`** - Cleaned and labeled review data
  - `lex_labeled_review_app.csv` - App Store labeled reviews
  - `lex_labeled_review_play.csv` - Play Store labeled reviews
  - `combined_reviews.csv` - Combined dataset
- **`/lexicon`** - Sentiment lexicon dictionaries
  - `positive.tsv` - Positive sentiment words
  - `negative.tsv` - Negative sentiment words

### 📁 `/notebooks`
Jupyter notebooks for analysis and experimentation.

- **`/appstore`** - App Store specific notebooks
  - `Tesis-Appstore-FIX.ipynb` - Complete App Store analysis
- **`/playstore`** - Play Store specific notebooks
  - `Tesis-Playstore-FIX.ipynb` - Complete Play Store analysis
- **`/exploratory`** - Exploratory data analysis notebooks

### 📁 `/scripts`
Python scripts organized by purpose.

- **`/analysis`** - Data analysis and statistics
  - `analyze_empty_strings.py` - Empty string analysis
  - `calculate_dataset_statistics.py` - Dataset statistics calculator
  - `check_token_reduction.py` - Token reduction verification
  - `language_distribution_analysis.py` - Language analysis
  - `rawtext_distribution.py` - Text distribution analysis
  - `year_variation.py` - Temporal variation analysis
  
- **`/evaluation`** - Model evaluation and results extraction
  - `extract_modeling_results.py` - Extract modeling phase results
  - `extract_evaluation_data_both.py` - Extract evaluation data for both platforms
  - `verify_evaluation_results.py` - Verify evaluation results
  
- **Root scripts** - Utility scripts
  - `generate_wordcloud_from_frequencies.py` - Generate wordcloud visualizations
  - `word_frequency_analysis.py` - Word frequency analysis

### 📁 `/outputs`
All generated outputs from experiments and analyses.

- **`/models`** - Trained machine learning models
  - `svm_pipeline_tfidf_app.pkl` - App Store TF-IDF + SVM model
  - `svm_pipeline_tfidf_play.pkl` - Play Store TF-IDF + SVM model
  - `svm_pipeline_bert_app.pkl` - App Store IndoBERT + SVM model
  - `svm_pipeline_bert_play.pkl` - Play Store IndoBERT + SVM model
  
- **`/results`** - Experiment results organized by type
  - **`/evaluation`** - Evaluation results (JSON)
    - `evaluation_results_appstore.json` - App Store evaluation
    - `evaluation_results_playstore.json` - Play Store evaluation
    - `evaluation_results_combined.json` - Combined evaluation
  - **`/language_analysis`** - Language analysis results
    - `app_store_language_distribution.csv` - App Store language data
    - `play_store_language_distribution.csv` - Play Store language data
    - `language_distribution_summary.csv` - Summary statistics
    - `word_frequency_analysis.txt` - Word frequency by sentiment
  - **`/model_exports`** - Exported model results (JSON)
    - `exported_model_results_app.json` - App Store model results
    - `exported_model_results_play.json` - Play Store model results
    - `modeling_results_summary.json` - Modeling phase summary
  - `token_reduction_verification.txt` - Token reduction statistics
  
- **`/reports`** - Comprehensive evaluation reports
  - `EVALUATION_RESULTS_APPSTORE.md` - Complete App Store evaluation (7 sections)
  - `EVALUATION_RESULTS_PLAYSTORE.md` - Complete Play Store evaluation (7 sections)
  - `PLATFORM_COMPARISON_ANALYSIS.md` - Cross-platform comparative analysis
  - `README.md` - Reports overview
  


### 📁 `/docs`
Comprehensive project documentation.

- **`/analysis`** - Analysis documentation and visualizations
  - **`/wordclouds`** - Word cloud visualizations by platform
    - **`/app_store`** - App Store wordclouds (negatif, netral, positif)
    - **`/play_store`** - Play Store wordclouds (negatif, netral, positif)
  - `WORD_FREQUENCY_RESULTS.md` - Word frequency analysis documentation
  
- **`/guides`** - User guides and tutorials
  - `DOCUMENTATION_GUIDE.md` - Documentation guide
  - `LANGUAGE_DISTRIBUTION_RESULTS.md` - Language analysis results
  - `README.md` - Guides overview
  
- **`/technical`** - Technical documentation
  - `data_preparation_phase.md` - Data preparation documentation
  - `README.md` - Technical docs overview
  
- **`/thesis`** - Thesis documentation
  - `README.md` - Thesis chapters overview

### 📁 `/dashboard`
Interactive Streamlit dashboard application.

- **`/pages`** - Dashboard page components
  - Multi-page dashboard structure
  
- **`/utils`** - Utility functions for dashboard
  - Helper functions and data loaders
  
- **`/assets`** - Dashboard assets
  - CSS stylesheets
  - Images and icons
  
- `dashboard.py` - Main dashboard application
- `run_dashboard.ps1` - PowerShell launcher script

### 📁 `/notebooks` (Root notebooks)
- `Tesis-Appstore-FIX.ipynb` - App Store complete analysis
- `Tesis-Playstore-FIX.ipynb` - Play Store complete analysis

### 📄 Root Files
- `README.md` - Project overview and quick start
- `requirements.txt` - Python dependencies
- `PROJECT_STRUCTURE.md` - This file
- `.gitignore` - Git ignore rules

---

## File Naming Conventions

### Data Files
- `*_app.csv` - App Store data
- `*_play.csv` - Play Store data
- `combined_*.csv` - Combined platform data

### Model Files
- `svm_pipeline_*.pkl` - SVM pipeline models
- `*_tfidf_*.pkl` - TF-IDF based models
- `*_bert_*.pkl` - BERT based models

### Result Files
- `*_results_*.json` - JSON result files
- `*_summary.*` - Summary files
- `UPPERCASE_*.md` - Important documentation

### Script Files
- `extract_*.py` - Data extraction scripts
- `calculate_*.py` - Calculation scripts
- `*_analysis.py` - Analysis scripts

---

## Data Flow

```
Raw Data (App Store / Play Store Reviews)
    ↓
[Data Cleaning & Preprocessing]
    ↓
Lexicon-Based Labeling
    ↓
[Feature Extraction: TF-IDF / IndoBERT]
    ↓
[Model Training: SVM with GridSearchCV]
    ↓
[Evaluation: Confusion Matrix, Classification Report]
    ↓
Results & Visualization
    ↓
Dashboard Display
```

---

## Quick Navigation

- **Start here:** `/README.md`
- **Run analysis:** `/notebooks/`
- **View results:** `/outputs/reports/`
- **Read documentation:** `/docs/`
- **Launch dashboard:** `/dashboard/dashboard.py`
- **Analysis scripts:** `/scripts/analysis/`
- **Evaluation scripts:** `/scripts/evaluation/`

---

**Maintained by:** Yenatari S  
**Last Updated:** November 6, 2025
