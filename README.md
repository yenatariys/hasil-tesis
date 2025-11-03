# Disney+ Hotstar Sentiment Analysis
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
   .\dashboard\run_dashboard.ps1
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
