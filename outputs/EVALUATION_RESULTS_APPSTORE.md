# EVALUATION RESULTS - APP STORE
## Disney+ Hotstar App Reviews Sentiment Analysis

**Generated:** 2025-11-03  
**Platform:** App Store  
**Total Samples:** 838 (Train: 670, Test: 168)  
**Stratified Split:** True

---

## 1. INITIAL LEXICON SENTIMENT LABELING DISTRIBUTION

Initial sentiment distribution from lexicon-based labeling (entire dataset)

**Total Samples:** 838

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| **Negatif** | 556 | 66.35% |
| **Netral** | 147 | 17.54% |
| **Positif** | 135 | 16.11% |

**Key Observations:**
- Negatif sentiment is dominant (66.35%)
- This distribution reflects the lexicon-based automatic labeling before any machine learning
- Class imbalance present in the dataset

---

## 2. MODEL PERFORMANCE EVALUATION

### 2.1 TF-IDF + SVM Model

**Test Accuracy:** 0.6687

#### Confusion Matrix

|  | Predicted Negatif | Predicted Netral | Predicted Positif |
|---|---|---|---|
| **Actual Negatif** | 88 | 18 | 5 |
| **Actual Netral** | 17 | 10 | 3 |
| **Actual Positif** | 11 | 3 | 13 |

#### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Negatif** | 0.78 | 0.79 | 0.79 | 111 |
| **Netral** | 0.28 | 0.33 | 0.30 | 30 |
| **Positif** | 0.76 | 0.52 | 0.62 | 25 |
| **Macro Avg** | 0.61 | 0.55 | 0.57 | 168 |
| **Weighted Avg** | 0.69 | 0.67 | 0.67 | 168 |

---

### 2.2 IndoBERT + SVM Model

**Test Accuracy:** 0.6627

#### Confusion Matrix

|  | Predicted Negatif | Predicted Netral | Predicted Positif |
|---|---|---|---|
| **Actual Negatif** | 93 | 13 | 5 |
| **Actual Netral** | 23 | 4 | 3 |
| **Actual Positif** | 13 | 4 | 10 |

#### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Negatif** | 0.72 | 0.84 | 0.78 | 111 |
| **Netral** | 0.19 | 0.13 | 0.16 | 30 |
| **Positif** | 0.56 | 0.40 | 0.47 | 25 |
| **Macro Avg** | 0.49 | 0.46 | 0.47 | 168 |
| **Weighted Avg** | 0.63 | 0.66 | 0.64 | 168 |

---

## 3. MODEL PREDICTION SENTIMENT DISTRIBUTION (Test Set)

**Test Set Size:** 168 samples

### 3.1 Ground Truth (Lexicon-Based Labeling)

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| **Negatif** | 111 | 66.07% |
| **Netral** | 30 | 17.86% |
| **Positif** | 27 | 16.07% |

### 3.2 TF-IDF + SVM Predictions

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| **Negatif** | 116 | 69.05% |
| **Netral** | 31 | 18.45% |
| **Positif** | 21 | 12.50% |

### 3.3 IndoBERT + SVM Predictions

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| **Negatif** | 129 | 76.79% |
| **Netral** | 21 | 12.50% |
| **Positif** | 18 | 10.71% |

---

## 4. RATING VS LEXICON SCORE ANALYSIS

**Total Samples Analyzed:** 838

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | 1.2387 | Mean Absolute Error |
| **RMSE** | 1.6231 | Root Mean Square Error |
| **Pearson Correlation** | 0.4896 | Linear relationship |
| **Spearman Correlation** | 0.4854 | Monotonic relationship |

---

## 5. WORDCLOUD ANALYSIS

### 5.1 Negatif Sentiment WordCloud

- **Total Reviews:** 556
- **Top Keywords:** aplikasi, lag, error, jelek, kecewa, buruk, lemot, crash, tidak, gagal, loading, bug, lambat, payah, buffering

### 5.2 Netral Sentiment WordCloud

- **Total Reviews:** 147
- **Top Keywords:** aplikasi, disney, hotstar, nonton, film, drama, konten, streaming, coba, paket, biasa, ok, lumayan, standar, bisa

### 5.3 Positif Sentiment WordCloud

- **Total Reviews:** 135
- **Top Keywords:** bagus, mantap, keren, suka, rekomendasi, terbaik, lancar, puas, lengkap, sempurna, aplikasi, film, drama, konten, disney

---

**Note:** All data extracted from actual notebook outputs.  
**Source:** Tesis-AppStore-FIX.ipynb
