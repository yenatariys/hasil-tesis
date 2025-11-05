# EVALUATION RESULTS - PLAY STORE
## Disney+ Hotstar App Reviews Sentiment Analysis

**Generated:** 2025-11-05  
**Platform:** Play Store  
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

- **Total Reviews:** 467
- **Top Keywords:** langgan (117), film (95), nonton (78), bayar (77), download (48), tonton (45), baik (40), hp (39), gambar (39), login (38), tv (37), buka (35), masuk (34), suara (34), kode (30)

**Key Insights:**
- Subscription and payment issues prominent (langgan, bayar)
- Streaming functionality problems (nonton, tonton, gambar, suara)
- Device/platform concerns (hp, tv, download)
- Technical issues affect core functionality (login, buka, masuk)

### 5.2 Netral Sentiment WordCloud

- **Total Reviews:** 227
- **Top Keywords:** film (38), login (21), nonton (20), langgan (20), apk (14), suka (11), bayar (11), baik (9), putar (9), paket (9), video (8), gk (8), telkomsel (8), pakai (8), pake (7)

**Key Insights:**
- More balanced view of app functionality
- Platform-specific mentions (apk, telkomsel)
- Mix of positive and negative terms (suka, baik vs gk)
- Focus on core features (film, nonton, putar)

### 5.3 Positif Sentiment WordCloud

- **Total Reviews:** 105
- **Top Keywords:** langgan (48), film (14), mantap (13), nonton (12), oke (11), paket (7), update (7), video (7), putar (7), jaring (7), loading (6), gratis (5), baik (5), kadang (5), uang (5)

**Key Insights:**
- Strong subscription satisfaction (langgan appears most)
- Positive performance feedback (mantap, oke)
- Content appreciation (film, video)
- Value perception (paket, gratis, uang)

### Overall Word Frequency Analysis

**Cross-Platform Patterns:**
1. **Subscription Focus:** 'langgan' is highly frequent across all sentiments, particularly in positive reviews
2. **Content Centrality:** 'film' and streaming terms (nonton, putar) consistently important
3. **Technical Terms:** Different focus from App Store - more about streaming quality than authentication
4. **Platform Specifics:** Mobile-centric issues (hp, apk) unique to Play Store
5. **Provider Integration:** Telkomsel mentions indicate carrier billing/integration importance

---

**Note:** All data extracted from actual notebook outputs.  
**Source:** Tesis-PlayStore-FIX.ipynb
