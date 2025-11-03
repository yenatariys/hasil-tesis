# EVALUATION RESULTS - PLAY STORE
## Disney+ Hotstar App Reviews Sentiment Analysis

**Generated:** 2025-11-03  
**Platform:** Play Store  
**Total Samples:** 838 (Train: 670, Test: 168)  
**Stratified Split:** True

---

## 1. INITIAL LEXICON SENTIMENT LABELING DISTRIBUTION

Initial sentiment distribution from lexicon-based labeling (entire dataset)

**Total Samples:** 838

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| **Negatif** | 689 | 82.22% |
| **Netral** | 90 | 10.74% |
| **Positif** | 59 | 7.04% |

**Key Observations:**
- Negatif sentiment is dominant (82.22%)
- This distribution reflects the lexicon-based automatic labeling before any machine learning
- Class imbalance present in the dataset

---

## 2. MODEL PERFORMANCE EVALUATION

### 2.1 TF-IDF + SVM Model

**Test Accuracy:** 0.7321

#### Confusion Matrix

|  | Predicted Negatif | Predicted Netral | Predicted Positif |
|---|---|---|---|
| **Actual Negatif** | 116 | 18 | 4 |
| **Actual Netral** | 13 | 4 | 1 |
| **Actual Positif** | 9 | 2 | 1 |

#### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Negatif** | 0.84 | 0.84 | 0.84 | 138 |
| **Netral** | 0.17 | 0.22 | 0.19 | 18 |
| **Positif** | 0.17 | 0.08 | 0.11 | 12 |
| **Macro Avg** | 0.39 | 0.38 | 0.38 | 168 |
| **Weighted Avg** | 0.72 | 0.73 | 0.72 | 168 |

---

### 2.2 IndoBERT + SVM Model

**Test Accuracy:** 0.7262

#### Confusion Matrix

|  | Predicted Negatif | Predicted Netral | Predicted Positif |
|---|---|---|---|
| **Actual Negatif** | 118 | 16 | 4 |
| **Actual Netral** | 14 | 3 | 1 |
| **Actual Positif** | 10 | 2 | 0 |

#### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Negatif** | 0.83 | 0.86 | 0.84 | 138 |
| **Netral** | 0.14 | 0.17 | 0.15 | 18 |
| **Positif** | 0.00 | 0.00 | 0.00 | 12 |
| **Macro Avg** | 0.33 | 0.34 | 0.33 | 168 |
| **Weighted Avg** | 0.70 | 0.73 | 0.71 | 168 |

---

## 3. MODEL PREDICTION SENTIMENT DISTRIBUTION (Test Set)

**Test Set Size:** 168 samples

### 3.1 Ground Truth (Lexicon-Based Labeling)

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| **Negatif** | 138 | 82.14% |
| **Netral** | 18 | 10.71% |
| **Positif** | 12 | 7.14% |

### 3.2 TF-IDF + SVM Predictions

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| **Negatif** | 138 | 82.14% |
| **Netral** | 24 | 14.29% |
| **Positif** | 6 | 3.57% |

### 3.3 IndoBERT + SVM Predictions

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| **Negatif** | 142 | 84.52% |
| **Netral** | 21 | 12.50% |
| **Positif** | 5 | 2.98% |

---

## 4. RATING VS LEXICON SCORE ANALYSIS

**Total Samples Analyzed:** 838

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | 1.4672 | Mean Absolute Error |
| **RMSE** | 1.8453 | Root Mean Square Error |
| **Pearson Correlation** | 0.3824 | Linear relationship |
| **Spearman Correlation** | 0.3791 | Monotonic relationship |

---

## 5. WORDCLOUD ANALYSIS

### 5.1 Negatif Sentiment WordCloud

- **Total Reviews:** 689
- **Top Keywords:** aplikasi, tidak, error, jelek, buruk, ga, lemot, lag, crash, kecewa, loading, bug, payah, eror, sering

### 5.2 Netral Sentiment WordCloud

- **Total Reviews:** 90
- **Top Keywords:** aplikasi, disney, hotstar, nonton, film, konten, drama, bagus, coba, subscribe, biasa, ok, kurang, ada, tapi

### 5.3 Positif Sentiment WordCloud

- **Total Reviews:** 59
- **Top Keywords:** bagus, mantap, keren, lancar, aplikasi, suka, puas, rekomendasi, terbaik, film, disney, lengkap, konten, sempurna, top

---

**Note:** All data extracted from actual notebook outputs.  
**Source:** Tesis-PlayStore-FIX.ipynb
