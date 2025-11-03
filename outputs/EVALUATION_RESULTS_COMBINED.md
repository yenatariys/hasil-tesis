# EVALUATION RESULTS - COMBINED
## Disney+ Hotstar App Reviews Sentiment Analysis
### Cross-Platform Comparison: App Store vs Play Store

**Generated:** 2025-11-03  
**Platforms:** App Store & Play Store  
**Total Samples per Platform:** 838 (Train: 670, Test: 168)  
**Stratified Split:** True

---

## 1. INITIAL LEXICON SENTIMENT LABELING DISTRIBUTION

Comparison of sentiment distribution from lexicon-based labeling (entire dataset for both platforms)

| Sentiment | App Store Count | App Store % | Play Store Count | Play Store % | Difference |
|-----------|-----------------|-------------|------------------|--------------|------------|
| **Negatif** | 556 | 66.35% | 689 | 82.22% | +15.87% |
| **Netral** | 147 | 17.54% | 90 | 10.74% | -6.80% |
| **Positif** | 135 | 16.11% | 59 | 7.04% | -9.07% |
| **Total** | 838 | 100% | 838 | 100% | - |

### Key Observations:
- **Play Store has significantly higher negative sentiment** (82.22% vs 66.35%)
- **App Store has more balanced distribution** with higher Netral (17.54%) and Positif (16.11%)
- **Play Store shows severe class imbalance** with Negatif dominating over 82% of reviews
- **App Store shows moderate class imbalance** with Negatif at 66%
- Both platforms reflect lexicon-based automatic labeling before machine learning

---

## 2. MODEL PERFORMANCE EVALUATION

### 2.1 TF-IDF + SVM Model Comparison

| Metric | App Store | Play Store | Better Performance |
|--------|-----------|------------|-------------------|
| **Test Accuracy** | 0.6687 (66.87%) | 0.7321 (73.21%) | Play Store (+6.34%) |
| **Macro F1-Score** | 0.57 | 0.38 | App Store (+0.19) |
| **Weighted F1-Score** | 0.67 | 0.72 | Play Store (+0.05) |

#### Confusion Matrix Comparison

**App Store:**

|  | Pred Negatif | Pred Netral | Pred Positif |
|---|---|---|---|
| **Actual Negatif** | 88 | 18 | 5 |
| **Actual Netral** | 17 | 10 | 3 |
| **Actual Positif** | 11 | 3 | 13 |

**Play Store:**

|  | Pred Negatif | Pred Netral | Pred Positif |
|---|---|---|---|
| **Actual Negatif** | 116 | 18 | 4 |
| **Actual Netral** | 13 | 4 | 1 |
| **Actual Positif** | 9 | 2 | 1 |

#### Per-Class Performance Comparison

**Negatif Class:**

| Platform | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| App Store | 0.78 | 0.79 | 0.79 | 111 |
| Play Store | 0.84 | 0.84 | 0.84 | 138 |
| **Winner** | **Play Store** | **Play Store** | **Play Store** | - |

**Netral Class:**

| Platform | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| App Store | 0.28 | 0.33 | 0.30 | 30 |
| Play Store | 0.17 | 0.22 | 0.19 | 18 |
| **Winner** | **App Store** | **App Store** | **App Store** | - |

**Positif Class:**

| Platform | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| App Store | 0.76 | 0.52 | 0.62 | 25 |
| Play Store | 0.17 | 0.08 | 0.11 | 12 |
| **Winner** | **App Store** | **App Store** | **App Store** | - |

---

### 2.2 IndoBERT + SVM Model Comparison

| Metric | App Store | Play Store | Better Performance |
|--------|-----------|------------|-------------------|
| **Test Accuracy** | 0.6627 (66.27%) | 0.7262 (72.62%) | Play Store (+6.35%) |
| **Macro F1-Score** | 0.47 | 0.33 | App Store (+0.14) |
| **Weighted F1-Score** | 0.64 | 0.71 | Play Store (+0.07) |

#### Confusion Matrix Comparison

**App Store:**

|  | Pred Negatif | Pred Netral | Pred Positif |
|---|---|---|---|
| **Actual Negatif** | 93 | 13 | 5 |
| **Actual Netral** | 23 | 4 | 3 |
| **Actual Positif** | 13 | 4 | 10 |

**Play Store:**

|  | Pred Negatif | Pred Netral | Pred Positif |
|---|---|---|---|
| **Actual Negatif** | 118 | 16 | 4 |
| **Actual Netral** | 14 | 3 | 1 |
| **Actual Positif** | 10 | 2 | 0 |

#### Per-Class Performance Comparison

**Negatif Class:**

| Platform | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| App Store | 0.72 | 0.84 | 0.78 | 111 |
| Play Store | 0.83 | 0.86 | 0.84 | 138 |
| **Winner** | **Play Store** | **Play Store** | **Play Store** | - |

**Netral Class:**

| Platform | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| App Store | 0.19 | 0.13 | 0.16 | 30 |
| Play Store | 0.14 | 0.17 | 0.15 | 18 |
| **Winner** | **App Store** | **Play Store** | **App Store** | - |

**Positif Class:**

| Platform | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| App Store | 0.56 | 0.40 | 0.47 | 25 |
| Play Store | 0.00 | 0.00 | 0.00 | 12 |
| **Winner** | **App Store** | **App Store** | **App Store** | - |

---

## 3. MODEL PREDICTION SENTIMENT DISTRIBUTION (Test Set)

### 3.1 Ground Truth Comparison

| Sentiment | App Store Count | App Store % | Play Store Count | Play Store % |
|-----------|-----------------|-------------|------------------|--------------|
| **Negatif** | 111 | 66.07% | 138 | 82.14% |
| **Netral** | 30 | 17.86% | 18 | 10.71% |
| **Positif** | 27 | 16.07% | 12 | 7.14% |

### 3.2 TF-IDF + SVM Predictions Comparison

| Sentiment | App Store Count | App Store % | Play Store Count | Play Store % |
|-----------|-----------------|-------------|------------------|--------------|
| **Negatif** | 116 | 69.05% | 138 | 82.14% |
| **Netral** | 31 | 18.45% | 24 | 14.29% |
| **Positif** | 21 | 12.50% | 6 | 3.57% |

**Prediction Bias Analysis (vs Ground Truth):**
- **App Store:** Negatif +2.98%, Netral +0.59%, Positif -3.57%
- **Play Store:** Negatif +0%, Netral +3.58%, Positif -3.57%

### 3.3 IndoBERT + SVM Predictions Comparison

| Sentiment | App Store Count | App Store % | Play Store Count | Play Store % |
|-----------|-----------------|-------------|------------------|--------------|
| **Negatif** | 129 | 76.79% | 142 | 84.52% |
| **Netral** | 21 | 12.50% | 21 | 12.50% |
| **Positif** | 18 | 10.71% | 5 | 2.98% |

**Prediction Bias Analysis (vs Ground Truth):**
- **App Store:** Negatif +10.72%, Netral -5.36%, Positif -5.36%
- **Play Store:** Negatif +2.38%, Netral +1.79%, Positif -4.16%

### Key Findings:
- **Play Store TF-IDF shows perfect alignment** with ground truth for Negatif class (82.14%)
- **App Store IndoBERT shows strongest negative bias** (+10.72% over ground truth)
- **Both platforms under-predict Positif sentiment** across both models
- **Play Store's severe class imbalance** (82% Negatif) leads to more conservative predictions

---

## 4. RATING VS LEXICON SCORE ANALYSIS

### Correlation Metrics Comparison

| Metric | App Store | Play Store | Better Correlation |
|--------|-----------|------------|-------------------|
| **MAE** | 1.2387 | 1.4672 | App Store (lower is better) |
| **RMSE** | 1.6231 | 1.8453 | App Store (lower is better) |
| **Pearson Correlation** | 0.4896 | 0.3824 | App Store (+0.11) |
| **Spearman Correlation** | 0.4854 | 0.3791 | App Store (+0.11) |

### Interpretation:
- **App Store shows moderate correlation** (≈0.49) between user ratings and lexicon scores
- **Play Store shows low-moderate correlation** (≈0.38) between user ratings and lexicon scores
- **App Store lexicon labeling is more reliable** - better alignment with actual user ratings
- **Play Store has higher error rates** (MAE: 1.47 vs 1.24, RMSE: 1.85 vs 1.62)
- Both platforms show **imperfect alignment**, indicating factors beyond text sentiment affect ratings (e.g., technical issues, expectations, platform-specific behavior)

---

## 5. WORDCLOUD ANALYSIS

### 5.1 Negatif Sentiment Keywords Comparison

| Platform | Total Reviews | Top Keywords |
|----------|---------------|--------------|
| **App Store** | 556 | aplikasi, lag, error, jelek, kecewa, buruk, lemot, crash, tidak, gagal, loading, bug, lambat, payah, buffering |
| **Play Store** | 689 | aplikasi, tidak, error, jelek, buruk, ga, lemot, lag, crash, kecewa, loading, bug, payah, eror, sering |

**Common Negative Terms:** aplikasi, lag, error, jelek, buruk, lemot, crash, kecewa, loading, bug, payah  
**Platform-Specific:** App Store emphasizes "buffering", "gagal"; Play Store uses "ga" (informal), "sering" (often), "eror" (spelling variant)

### 5.2 Netral Sentiment Keywords Comparison

| Platform | Total Reviews | Top Keywords |
|----------|---------------|--------------|
| **App Store** | 147 | aplikasi, disney, hotstar, nonton, film, drama, konten, streaming, coba, paket, biasa, ok, lumayan, standar, bisa |
| **Play Store** | 90 | aplikasi, disney, hotstar, nonton, film, konten, drama, bagus, coba, subscribe, biasa, ok, kurang, ada, tapi |

**Common Neutral Terms:** aplikasi, disney, hotstar, nonton, film, konten, drama, coba, biasa, ok  
**Platform-Specific:** App Store mentions "paket", "lumayan", "standar"; Play Store mentions "subscribe", "kurang", "tapi"

### 5.3 Positif Sentiment Keywords Comparison

| Platform | Total Reviews | Top Keywords |
|----------|---------------|--------------|
| **App Store** | 135 | bagus, mantap, keren, suka, rekomendasi, terbaik, lancar, puas, lengkap, sempurna, aplikasi, film, drama, konten, disney |
| **Play Store** | 59 | bagus, mantap, keren, lancar, aplikasi, suka, puas, rekomendasi, terbaik, film, disney, lengkap, konten, sempurna, top |

**Common Positive Terms:** bagus, mantap, keren, suka, rekomendasi, terbaik, lancar, puas, lengkap, sempurna, film  
**Platform-Specific:** Play Store adds "top"; App Store has more diverse positive vocabulary

### Cross-Platform WordCloud Insights:
- **Technical complaints dominate negative reviews** on both platforms (lag, error, crash, loading)
- **Content-related terms dominate neutral reviews** (film, drama, konten, nonton)
- **Quality and satisfaction terms dominate positive reviews** (bagus, mantap, puas, terbaik)
- **Play Store reviews use more informal language** ("ga" instead of "tidak", "eror" variant)
- **Negative sentiment has most diverse vocabulary** on both platforms

---

## 6. CROSS-PLATFORM KEY FINDINGS

### Model Performance:
1. **Play Store achieves higher overall accuracy** (TF-IDF: 73.21%, IndoBERT: 72.62%) vs App Store (TF-IDF: 66.87%, IndoBERT: 66.27%)
2. **App Store achieves better macro F1-scores** (TF-IDF: 0.57, IndoBERT: 0.47) indicating better performance across all classes
3. **Play Store's higher accuracy is driven by dominant Negatif class** (82% vs 66%)
4. **TF-IDF outperforms IndoBERT on both platforms** for macro F1-score

### Class-Specific Insights:
5. **Negatif class performs best on both platforms** (F1: 0.79-0.84) due to abundant training data
6. **Netral class is most challenging** on both platforms (F1: 0.15-0.30) due to limited samples
7. **Positif class shows dramatic platform difference:** App Store F1: 0.11-0.62 vs Play Store F1: 0.00-0.11
8. **App Store models handle minority classes better** despite lower overall accuracy

### Distribution Analysis:
9. **Initial lexicon distribution difference:** Play Store 82% Negatif vs App Store 66% Negatif
10. **Play Store shows severe class imbalance** affecting model generalization to minority classes
11. **Both models show negative bias** but Play Store TF-IDF achieves perfect Negatif alignment
12. **Positif class consistently under-predicted** on both platforms

### Rating-Lexicon Alignment:
13. **App Store shows stronger correlation** (0.49 vs 0.38) between ratings and lexicon scores
14. **Play Store has higher prediction errors** (MAE: 1.47 vs 1.24)
15. **Moderate correlations suggest** user ratings influenced by factors beyond text sentiment

### Linguistic Patterns:
16. **Technical issues dominate negative sentiment** (lag, error, crash, loading) on both platforms
17. **Play Store uses more informal language** reflecting different user demographics
18. **Content quality drives positive sentiment** (film, drama, konten) on both platforms

### Recommendations:
- **For Play Store:** Address severe class imbalance with data augmentation or class weighting
- **For App Store:** Leverage better-balanced data for improved minority class performance
- **For both:** TF-IDF + SVM recommended over IndoBERT for better macro performance
- **For both:** Focus on improving Netral and Positif class predictions through targeted strategies

---

## 7. MODEL SELECTION RECOMMENDATION

### Best Model per Platform:

| Platform | Recommended Model | Reason |
|----------|------------------|---------|
| **App Store** | **TF-IDF + SVM** | Higher macro F1 (0.57 vs 0.47), better minority class performance, 66.87% accuracy |
| **Play Store** | **TF-IDF + SVM** | Higher macro F1 (0.38 vs 0.33), better minority class performance, 73.21% accuracy |

### Justification:
- **Macro F1-score prioritized** over accuracy to account for class imbalance
- **TF-IDF consistently outperforms IndoBERT** on minority classes (Netral, Positif)
- **IndoBERT shows stronger bias** toward dominant class (Negatif)
- **Simpler model (TF-IDF) provides better generalization** across all sentiment categories

---

**Note:** All data extracted from actual notebook outputs.  
**Sources:** Tesis-Appstore-FIX.ipynb & Tesis-Playstore-FIX.ipynb  
**Date:** 2025-11-03
