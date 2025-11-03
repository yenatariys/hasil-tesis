"""
Project Reorganization Script
This script reorganizes the hasil-tesis project into a clean, structured layout
"""

import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create organized directory structure"""
    
    base_dirs = {
        'data': {
            'raw': 'Raw data files (original datasets)',
            'processed': 'Processed/cleaned data files',
            'lexicon': 'Lexicon dictionaries for sentiment analysis'
        },
        'notebooks': {
            'appstore': 'App Store analysis notebooks',
            'playstore': 'Play Store analysis notebooks',
            'exploratory': 'Exploratory data analysis notebooks'
        },
        'scripts': {
            'data_preparation': 'Data preparation and cleaning scripts',
            'modeling': 'Model training and evaluation scripts',
            'analysis': 'Statistical analysis scripts',
            'evaluation': 'Evaluation phase scripts'
        },
        'outputs': {
            'models': 'Trained model files (.pkl)',
            'results': 'Experiment results (JSON, CSV)',
            'reports': 'Generated markdown reports',
            'visualizations': 'Plots and charts'
        },
        'docs': {
            'thesis': 'Thesis chapters and documentation',
            'technical': 'Technical documentation (CRISP-DM phases)',
            'guides': 'User guides and README files'
        },
        'dashboard': {
            'pages': 'Dashboard page components',
            'utils': 'Dashboard utility functions',
            'assets': 'Dashboard assets (CSS, images)'
        }
    }
    
    print("Creating directory structure...")
    for main_dir, subdirs in base_dirs.items():
        os.makedirs(main_dir, exist_ok=True)
        print(f"✅ Created: {main_dir}/")
        
        if isinstance(subdirs, dict):
            for subdir, description in subdirs.items():
                path = os.path.join(main_dir, subdir)
                os.makedirs(path, exist_ok=True)
                print(f"   ✅ Created: {main_dir}/{subdir}/ - {description}")
                
                # Create README in each subdirectory
                readme_path = os.path.join(path, 'README.md')
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(f"# {subdir.replace('_', ' ').title()}\n\n{description}\n")


def move_files():
    """Move files to appropriate directories"""
    
    file_moves = {
        # Data files
        'lex_labeled_review_app.csv': 'data/processed/',
        'lex_labeled_review_play.csv': 'data/processed/',
        'combined_reviews.csv': 'data/processed/',
        'data/negative.tsv': 'data/lexicon/',
        'data/positive.tsv': 'data/lexicon/',
        
        # Scripts - Data Preparation
        'calculate_dataset_statistics.py': 'scripts/data_preparation/',
        'language_distribution_analysis.py': 'scripts/analysis/',
        'rawtext_distribution.py': 'scripts/analysis/',
        'year_variation.py': 'scripts/analysis/',
        
        # Scripts - Modeling & Evaluation
        'extract_modeling_results.py': 'scripts/evaluation/',
        'extract_evaluation_data_both.py': 'scripts/evaluation/',
        
        # Dashboard files
        'dashboard.py': 'dashboard/',
        'run_dashboard.ps1': 'dashboard/',
        'src/dashboard.py': 'dashboard/pages/',
        
        # Documentation - Thesis
        'THESIS_MODELING_PHASE.md': 'docs/thesis/',
        
        # Documentation - Technical
        'data_preparation_phase.md': 'docs/technical/',
        'modeling_phase.md': 'docs/technical/',
        
        # Documentation - Guides
        'DOCUMENTATION_GUIDE.md': 'docs/guides/',
        'DOCUMENTATION_SUMMARY.md': 'docs/guides/',
        'LANGUAGE_DISTRIBUTION_RESULTS.md': 'docs/guides/',
        'README.md': 'docs/guides/',
        
        # Output files - Models
        'outputs/svm_pipeline_tfidf_app.pkl': 'outputs/models/',
        'outputs/svm_pipeline_tfidf_play.pkl': 'outputs/models/',
        'outputs/svm_pipeline_bert_app.pkl': 'outputs/models/',
        'outputs/svm_pipeline_bert_play.pkl': 'outputs/models/',
        
        # Output files - Results
        'outputs/exported_model_results_app.json': 'outputs/results/',
        'outputs/exported_model_results_play.json': 'outputs/results/',
        'outputs/modeling_results_summary.json': 'outputs/results/',
        'outputs/evaluation_results_appstore.json': 'outputs/results/',
        'outputs/evaluation_results_playstore.json': 'outputs/results/',
        'outputs/evaluation_results_combined.json': 'outputs/results/',
        'outputs/language_distribution_summary.csv': 'outputs/results/',
        'outputs/app_store_language_distribution.csv': 'outputs/results/',
        'outputs/play_store_language_distribution.csv': 'outputs/results/',
        
        # Output files - Reports
        'outputs/MODELING_RESULTS.md': 'outputs/reports/',
        'outputs/EVALUATION_RESULTS_APPSTORE.md': 'outputs/reports/',
        'outputs/EVALUATION_RESULTS_PLAYSTORE.md': 'outputs/reports/',
        'outputs/EVALUATION_RESULTS_COMBINED.md': 'outputs/reports/',
        'outputs/APP_STORE_UPDATE_SUMMARY.md': 'outputs/reports/',
        'outputs/DATA_SPLIT_DOCUMENTATION_UPDATE.md': 'outputs/reports/',
        'outputs/EVALUATION_RESULTS.md': 'outputs/reports/',
    }
    
    print("\nMoving files to organized structure...")
    moved_count = 0
    skipped_count = 0
    
    for src, dest_dir in file_moves.items():
        if os.path.exists(src):
            # Create destination directory if it doesn't exist
            os.makedirs(dest_dir, exist_ok=True)
            
            # Get filename from source path
            filename = os.path.basename(src)
            dest_path = os.path.join(dest_dir, filename)
            
            # Move file
            try:
                shutil.move(src, dest_path)
                print(f"✅ Moved: {src} → {dest_path}")
                moved_count += 1
            except Exception as e:
                print(f"❌ Error moving {src}: {e}")
                skipped_count += 1
        else:
            print(f"⚠️  Skipped (not found): {src}")
            skipped_count += 1
    
    print(f"\n📊 Summary: {moved_count} files moved, {skipped_count} skipped")


def create_new_readme():
    """Create comprehensive README.md in root"""
    
    readme_content = """# Disney+ Hotstar Sentiment Analysis
## Multi-Platform Review Analysis using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

### 📖 Project Overview

This project performs comprehensive sentiment analysis on Disney+ Hotstar app reviews from both **App Store** and **Play Store** platforms. Using lexicon-based labeling and machine learning models (TF-IDF + SVM and IndoBERT + SVM), we analyze user sentiment patterns across platforms.

---

## 📁 Project Structure

```
hasil-tesis/
│
├── data/                          # Data files
│   ├── raw/                       # Original datasets
│   ├── processed/                 # Cleaned and labeled data
│   └── lexicon/                   # Sentiment lexicon dictionaries
│
├── notebooks/                     # Jupyter notebooks
│   ├── appstore/                  # App Store analysis
│   ├── playstore/                 # Play Store analysis
│   └── exploratory/               # EDA notebooks
│
├── scripts/                       # Python scripts
│   ├── data_preparation/          # Data cleaning and preparation
│   ├── modeling/                  # Model training scripts
│   ├── analysis/                  # Statistical analysis
│   └── evaluation/                # Model evaluation scripts
│
├── outputs/                       # Generated outputs
│   ├── models/                    # Trained models (.pkl files)
│   ├── results/                   # Results (JSON, CSV)
│   ├── reports/                   # Markdown reports
│   └── visualizations/            # Plots and charts
│
├── docs/                          # Documentation
│   ├── thesis/                    # Thesis chapters
│   ├── technical/                 # CRISP-DM documentation
│   └── guides/                    # User guides
│
├── dashboard/                     # Streamlit dashboard
│   ├── pages/                     # Dashboard pages
│   ├── utils/                     # Utility functions
│   └── assets/                    # CSS, images
│
├── .streamlit/                    # Streamlit configuration
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yenatariys/hasil-tesis.git
   cd hasil-tesis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard:**
   ```bash
   streamlit run dashboard/dashboard.py
   ```
   Or use PowerShell script:
   ```powershell
   .\\dashboard\\run_dashboard.ps1
   ```

---

## 📊 Key Features

### Data Analysis
- ✅ Multi-platform data collection (App Store & Play Store)
- ✅ Lexicon-based sentiment labeling
- ✅ Comprehensive EDA and statistical analysis
- ✅ Language distribution analysis
- ✅ Temporal variation analysis

### Machine Learning Models
- ✅ TF-IDF + SVM (Support Vector Machine)
- ✅ IndoBERT + SVM (Indonesian BERT embeddings)
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ Stratified train-test split (80:20)
- ✅ Class imbalance handling

### Evaluation Metrics
- ✅ Confusion matrices
- ✅ Classification reports (Precision, Recall, F1-Score)
- ✅ Cross-platform performance comparison
- ✅ Rating vs Lexicon score correlation analysis
- ✅ WordCloud visualization for each sentiment

### Interactive Dashboard
- ✅ Real-time sentiment visualization
- ✅ Cross-platform comparison charts
- ✅ Model performance metrics
- ✅ Interactive filters and controls

---

## 📈 Results Summary

### App Store Performance
- **TF-IDF + SVM:** 66.87% accuracy, 0.57 macro F1-score
- **IndoBERT + SVM:** 66.27% accuracy, 0.47 macro F1-score
- **Initial Distribution:** 66% Negatif, 18% Netral, 16% Positif

### Play Store Performance
- **TF-IDF + SVM:** 73.21% accuracy, 0.38 macro F1-score
- **IndoBERT + SVM:** 72.62% accuracy, 0.33 macro F1-score
- **Initial Distribution:** 82% Negatif, 11% Netral, 7% Positif

### Key Insights
- ✅ Play Store has higher negative sentiment (82% vs 66%)
- ✅ TF-IDF outperforms IndoBERT on macro F1-score
- ✅ App Store shows better minority class performance
- ✅ Both platforms struggle with Netral and Positif classes

---

## 📚 Documentation

Comprehensive documentation available in `docs/` directory:

- **Thesis Documentation:** `docs/thesis/`
  - Complete thesis chapters with results
  
- **Technical Documentation:** `docs/technical/`
  - CRISP-DM methodology phases
  - Data preparation documentation
  - Modeling phase documentation
  
- **User Guides:** `docs/guides/`
  - Project overview and setup
  - Analysis guides
  - Result interpretation

---

## 🔧 Technologies Used

- **Python 3.8+**
- **Machine Learning:** scikit-learn, transformers (IndoBERT)
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn, wordcloud
- **Dashboard:** Streamlit
- **NLP:** TF-IDF, BERT embeddings
- **Version Control:** Git

---

## 📝 CRISP-DM Methodology

This project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology:

1. ✅ **Business Understanding** - Sentiment analysis for app improvement
2. ✅ **Data Understanding** - 838 reviews per platform, multi-class sentiment
3. ✅ **Data Preparation** - Cleaning, lexicon labeling, stratified split
4. ✅ **Modeling** - TF-IDF + SVM, IndoBERT + SVM with hyperparameter tuning
5. ✅ **Evaluation** - Confusion matrices, classification reports, cross-platform analysis
6. 🔄 **Deployment** - Interactive Streamlit dashboard

---

## 👤 Author

**Yenatari S**
- GitHub: [@yenatariys](https://github.com/yenatariys)

---

## 📄 License

This project is part of a thesis research.

---

## 🙏 Acknowledgments

- Indonesian sentiment lexicon for initial labeling
- IndoBERT model for Indonesian language processing
- Disney+ Hotstar for providing the review data platform

---

**Last Updated:** November 3, 2025
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("\n✅ Created comprehensive README.md in root directory")


def create_project_structure_doc():
    """Create PROJECT_STRUCTURE.md documentation"""
    
    structure_doc = """# Project Structure Documentation

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
"""
    
    with open('PROJECT_STRUCTURE.md', 'w', encoding='utf-8') as f:
        f.write(structure_doc)
    
    print("✅ Created PROJECT_STRUCTURE.md documentation")


def main():
    print("="*60)
    print("PROJECT REORGANIZATION SCRIPT")
    print("="*60)
    
    try:
        # Step 1: Create directory structure
        create_directory_structure()
        
        # Step 2: Move files
        move_files()
        
        # Step 3: Create new README
        create_new_readme()
        
        # Step 4: Create project structure documentation
        create_project_structure_doc()
        
        print("\n" + "="*60)
        print("✅ PROJECT REORGANIZATION COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Review the new structure in each directory")
        print("2. Check README.md for project overview")
        print("3. Read PROJECT_STRUCTURE.md for detailed layout")
        print("4. Update any hardcoded paths in your scripts if needed")
        print("\nNew structure:")
        print("  - data/          (raw, processed, lexicon)")
        print("  - notebooks/     (appstore, playstore, exploratory)")
        print("  - scripts/       (data_preparation, modeling, analysis, evaluation)")
        print("  - outputs/       (models, results, reports, visualizations)")
        print("  - docs/          (thesis, technical, guides)")
        print("  - dashboard/     (pages, utils, assets)")
        
    except Exception as e:
        print(f"\n❌ Error during reorganization: {e}")
        print("Please review the error and try again.")


if __name__ == "__main__":
    main()
