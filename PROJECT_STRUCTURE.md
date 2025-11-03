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

- **`/data_preparation`** - Data cleaning and preparation
  - `calculate_dataset_statistics.py` - Dataset statistics calculator
  
- **`/modeling`** - Model training and optimization
  - (Model training scripts)
  
- **`/analysis`** - Statistical and exploratory analysis
  - `language_distribution_analysis.py` - Language analysis
  - `rawtext_distribution.py` - Text distribution analysis
  - `year_variation.py` - Temporal variation analysis
  
- **`/evaluation`** - Model evaluation and results extraction
  - `extract_modeling_results.py` - Extract modeling phase results
  - `extract_evaluation_data_both.py` - Extract evaluation data for both platforms

### 📁 `/outputs`
All generated outputs from experiments and analyses.

- **`/models`** - Trained machine learning models
  - `svm_pipeline_tfidf_app.pkl` - App Store TF-IDF + SVM model
  - `svm_pipeline_tfidf_play.pkl` - Play Store TF-IDF + SVM model
  - `svm_pipeline_bert_app.pkl` - App Store IndoBERT + SVM model
  - `svm_pipeline_bert_play.pkl` - Play Store IndoBERT + SVM model
  
- **`/results`** - Experiment results (JSON, CSV)
  - `modeling_results_summary.json` - Modeling phase summary
  - `evaluation_results_appstore.json` - App Store evaluation
  - `evaluation_results_playstore.json` - Play Store evaluation
  - `evaluation_results_combined.json` - Combined evaluation
  - Language distribution CSVs
  
- **`/reports`** - Generated markdown reports
  - `MODELING_RESULTS.md` - Modeling phase report
  - `EVALUATION_RESULTS_APPSTORE.md` - App Store evaluation report
  - `EVALUATION_RESULTS_PLAYSTORE.md` - Play Store evaluation report
  - `EVALUATION_RESULTS_COMBINED.md` - Cross-platform comparison
  
- **`/visualizations`** - Plots, charts, and visual outputs
  - (Generated visualizations)

### 📁 `/docs`
Comprehensive project documentation.

- **`/thesis`** - Thesis chapters and academic documentation
  - `THESIS_MODELING_PHASE.md` - Modeling phase chapter
  - (Other thesis chapters)
  
- **`/technical`** - Technical documentation following CRISP-DM
  - `data_preparation_phase.md` - Data preparation documentation
  - `modeling_phase.md` - Modeling methodology
  - (Other technical docs)
  
- **`/guides`** - User guides and tutorials
  - `DOCUMENTATION_GUIDE.md` - Documentation guide
  - `DOCUMENTATION_SUMMARY.md` - Documentation summary
  - `LANGUAGE_DISTRIBUTION_RESULTS.md` - Language analysis results
  - Original `README.md` backup

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

---

**Maintained by:** Yenatari S  
**Last Updated:** November 3, 2025
