# EVALUATION RESULTS - APP STORE
## Disney+ Hotstar App Reviews Sentiment Analysis

**Generated:** 2025-11-05  
**Platform:** App Store  
**Total Samples:** 838 (Train: 670, Test: 168)  
**Stratified Split:** True

**Related Documents:**
- [Platform Comparison Analysis](PLATFORM_COMPARISON_ANALYSIS.md)
- [Word Frequency Analysis](../docs/analysis/WORD_FREQUENCY_ANALYSIS.md)
- [Wordcloud Documentation](../docs/analysis/wordclouds/README.md)

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

- **Total Reviews:** 503
- **Top Keywords:** film (128), langgan (90), masuk (75), tv (72), kode (70), otp (59), bayar (57), baik (44), login (35), buka (35), coba (33), nomor (33), kali (32), aja (32), salah (31)

**Key Insights:**
- Authentication issues dominate negative reviews (masuk, kode, otp, login)
- Payment-related concerns significant (langgan, bayar)
- Device compatibility problems (tv, buka)
- High frequency of access-related terms suggests login/authentication as major pain point

### 5.2 Netral Sentiment WordCloud

- **Total Reviews:** 205
- **Top Keywords:** film (34), tv (25), langgan (25), otp (18), download (12), apple (12), login (12), dukung (11), kasih (11), terima (11), baik (11), suka (11), unduh (11), chromecast (10), fitur (10)

**Key Insights:**
- More balanced mix of features and issues
- Platform-specific mentions (apple, chromecast) indicate device integration focus
- Service-related terms more neutral (dukung, kasih, terima)
- Feature discussion more prominent (fitur, download, unduh)

### 5.3 Positif Sentiment WordCloud

- **Total Reviews:** 124
- **Top Keywords:** langgan (34), film (21), dukung (15), tv (12), harga (11), tambah (11), sedia (10), paket (9), apple (9), ya (8), chromecast (8), kualitas (8), kasih (7), bayar (7), guna (7)

**Key Insights:**
- Content value proposition stands out (film, kualitas)
- Subscription features viewed positively (langgan, paket, harga)
- Device compatibility praised (tv, apple, chromecast)
- Service quality appreciation (dukung, sedia, tambah)

### Overall Word Frequency Analysis

**Cross-Sentiment Patterns:**
1. **Content Focus:** 'film' appears in top keywords across all sentiments, indicating it's central to user experience
2. **Authentication Impact:** OTP/login issues more prevalent in negative reviews
3. **Device Integration:** TV/Chromecast compatibility discussed across sentiments
4. **Subscription Terms:** 'langgan' (subscribe) appears frequently in all categories, suggesting it's a key aspect of user experience

---

**Note:** All data extracted from actual notebook outputs.  
**Source:** Tesis-AppStore-FIX.ipynb
