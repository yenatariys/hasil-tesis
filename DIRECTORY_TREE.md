# Project Directory Tree

```
hasil-tesis/
│
├── 📁 data/                               # All data files
│   ├── 📁 raw/                            # Original datasets
│   ├── 📁 processed/                      # Cleaned & labeled data
│   │   ├── lex_labeled_review_app.csv     # App Store labeled reviews
│   │   ├── lex_labeled_review_play.csv    # Play Store labeled reviews
│   │   └── combined_reviews.csv           # Combined dataset
│   └── 📁 lexicon/                        # Sentiment dictionaries
│       ├── positive.tsv                   # Positive words
│       └── negative.tsv                   # Negative words
│
├── 📁 notebooks/                          # Jupyter notebooks
│   ├── 📁 appstore/                       # App Store analysis
│   │   └── Tesis-Appstore-FIX.ipynb      # Complete App Store notebook
│   ├── 📁 playstore/                      # Play Store analysis
│   │   └── Tesis-Playstore-FIX.ipynb     # Complete Play Store notebook
│   └── 📁 exploratory/                    # EDA notebooks
│
├── 📁 scripts/                            # Python scripts
│   ├── 📁 data_preparation/               # Data cleaning
│   │   └── calculate_dataset_statistics.py
│   ├── 📁 modeling/                       # Model training (empty - ready for future)
│   ├── 📁 analysis/                       # Statistical analysis
│   │   ├── language_distribution_analysis.py
│   │   ├── rawtext_distribution.py
│   │   └── year_variation.py
│   └── 📁 evaluation/                     # Model evaluation
│       ├── extract_modeling_results.py
│       └── extract_evaluation_data_both.py
│
├── 📁 outputs/                            # Generated outputs
│   ├── 📁 models/                         # Trained ML models
│   │   ├── svm_pipeline_tfidf_app.pkl    # App Store TF-IDF model
│   │   ├── svm_pipeline_tfidf_play.pkl   # Play Store TF-IDF model
│   │   ├── svm_pipeline_bert_app.pkl     # App Store BERT model
│   │   └── svm_pipeline_bert_play.pkl    # Play Store BERT model
│   ├── 📁 results/                        # Results (JSON, CSV)
│   │   ├── modeling_results_summary.json
│   │   ├── evaluation_results_appstore.json
│   │   ├── evaluation_results_playstore.json
│   │   ├── evaluation_results_combined.json
│   │   ├── exported_model_results_app.json
│   │   ├── exported_model_results_play.json
│   │   └── *_distribution.csv files
│   ├── 📁 reports/                        # Markdown reports
│   │   ├── MODELING_RESULTS.md
│   │   ├── EVALUATION_RESULTS_APPSTORE.md
│   │   ├── EVALUATION_RESULTS_PLAYSTORE.md
│   │   ├── EVALUATION_RESULTS_COMBINED.md
│   │   └── APP_STORE_UPDATE_SUMMARY.md
│   └── 📁 visualizations/                 # Plots & charts (ready for future)
│
├── 📁 docs/                               # Documentation
│   ├── 📁 thesis/                         # Thesis chapters
│   │   └── THESIS_MODELING_PHASE.md      # Modeling chapter
│   ├── 📁 technical/                      # CRISP-DM docs
│   │   ├── data_preparation_phase.md
│   │   └── modeling_phase.md
│   └── 📁 guides/                         # User guides
│       ├── DOCUMENTATION_GUIDE.md
│       ├── DOCUMENTATION_SUMMARY.md
│       └── LANGUAGE_DISTRIBUTION_RESULTS.md
│
├── 📁 dashboard/                          # Streamlit dashboard
│   ├── 📁 pages/                          # Dashboard pages
│   │   └── dashboard.py
│   ├── 📁 utils/                          # Utility functions (ready)
│   ├── 📁 assets/                         # CSS, images (ready)
│   ├── dashboard.py                       # Main app
│   └── run_dashboard.ps1                  # Launcher script
│
├── 📁 .streamlit/                         # Streamlit config
├── 📁 .git/                               # Git repository
│
├── 📄 README.md                           # Project overview ⭐
├── 📄 PROJECT_STRUCTURE.md                # This file
└── 📄 requirements.txt                    # Python dependencies

```

## Quick Access Guide

### 🚀 Getting Started
- **Start here:** `README.md`
- **Project structure:** `PROJECT_STRUCTURE.md`
- **Install deps:** `requirements.txt`

### 📊 Data Files
- **Processed data:** `data/processed/`
- **Lexicons:** `data/lexicon/`
- **Raw data:** `data/raw/` (add original files here)

### 📓 Analysis & Experiments
- **App Store notebook:** `notebooks/appstore/Tesis-Appstore-FIX.ipynb`
- **Play Store notebook:** `notebooks/playstore/Tesis-Playstore-FIX.ipynb`
- **Analysis scripts:** `scripts/analysis/`

### 🤖 Models & Results
- **Trained models:** `outputs/models/`
- **Result data:** `outputs/results/`
- **Reports:** `outputs/reports/`

### 📚 Documentation
- **Thesis:** `docs/thesis/`
- **Technical:** `docs/technical/`
- **Guides:** `docs/guides/`

### 🎨 Dashboard
- **Run:** `dashboard/dashboard.py`
- **Or:** `dashboard/run_dashboard.ps1`

## File Count Summary

| Directory | Purpose | File Count |
|-----------|---------|------------|
| `data/` | Data storage | ~6 files |
| `notebooks/` | Analysis notebooks | 2 main notebooks |
| `scripts/` | Python scripts | 6 scripts |
| `outputs/models/` | ML models | 4 .pkl files |
| `outputs/results/` | Results | 12+ JSON/CSV files |
| `outputs/reports/` | Reports | 6+ markdown files |
| `docs/` | Documentation | 8+ markdown files |
| `dashboard/` | Dashboard app | 2+ Python files |

**Total organized files:** 50+ files

## Benefits of New Structure

✅ **Organized by purpose** - Easy to find what you need  
✅ **Scalable** - Ready for future additions  
✅ **Clear separation** - Data, code, outputs, docs separated  
✅ **Professional** - Follows industry best practices  
✅ **Documented** - README in each subdirectory  
✅ **Git-friendly** - Logical structure for version control  

---

**Last Updated:** November 3, 2025  
**Maintained by:** Yenatari S
