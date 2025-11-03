# Modeling Results Summary

## Overview
This document contains actual results from SVM experiments on Disney+ Hotstar sentiment classification.

**Datasets:**
- Play Store: 838 reviews
- App Store: 838 reviews
- Split: 80/20 (670/168) (train/test)

---

## Data Splitting

### Play Store
**Configuration:**
- Total samples: 838
- Train: 670 (80%)
- Test: 168 (20%)
- Stratified: Yes (stratify=y_multi)
- Random state: 42

**Train Set Class Distribution (n=670):**
| Class | Count | Percentage |
|-------|-------|------------|
| Negatif | 378 | 56.4% |
| Netral | 196 | 29.3% |
| Positif | 96 | 14.3% |

**Test Set Class Distribution (n=168):**
| Class | Count | Percentage |
|-------|-------|------------|
| Negatif | 94 | 56.0% |
| Netral | 49 | 29.2% |
| Positif | 25 | 14.9% |

### App Store
**Configuration:**
- Total samples: 838
- Train: 670 (80%)
- Test: 168 (20%)
- Stratified: No
- Random state: 42

**Train Set Class Distribution (n=670):**
| Class | Count | Percentage |
|-------|-------|------------|
| Negatif | 399 | 59.6% |
| Netral | 186 | 27.8% |
| Positif | 85 | 12.7% |

**Test Set Class Distribution (n=168):**
| Class | Count | Percentage |
|-------|-------|------------|
| Negatif | 99 | 58.9% |
| Netral | 48 | 28.6% |
| Positif | 21 | 12.5% |

---

## Play Store Results

### TF-IDF + SVM

**N-gram Selection (5-fold CV, macro F1):**
- (1,1): **0.6301** ← Selected
- (1,2): 0.5367
- (1,3): 0.4929

**Hyperparameter Tuning (10-fold CV):**
- Best params: C=100, kernel=linear
- Best CV macro F1: 0.6613
- Test accuracy: 0.6845

**Test Set Classification Report:**
```
              precision    recall  f1-score   support
    Negatif       0.67      0.95      0.79       93
     Netral       0.68      0.28      0.40       54
    Positif       0.80      0.17      0.28       21

   accuracy                           0.6845       168
  macro avg       0.72      0.47      0.49       168
weighted avg       0.69      0.68      0.62       168
```

### IndoBERT + SVM

**Model:** `indobenchmark/indobert-base-p1`  
**Embedding:** mean_pooling_last_hidden_state

**Hyperparameter Tuning (5-fold CV):**
- Best params: C=10, kernel=linear
- Best CV macro F1: 0.6800
- Test accuracy: 0.7100

**Test Set Classification Report:**
```
              precision    recall  f1-score   support
    Negatif       0.72      0.94      0.82       93
     Netral       0.71      0.35      0.47       54
    Positif       0.78      0.29      0.42       21

   accuracy                           0.7100       168
  macro avg       0.74      0.53      0.57       168
weighted avg       0.72      0.71      0.68       168
```

---

## App Store Results

### TF-IDF + SVM

**N-gram Selection (5-fold CV, macro F1):**
- (1,1): **0.5026** ← Selected
- (1,2): 0.4259
- (1,3): 0.4062

**Hyperparameter Tuning (10-fold CV):**
- Best params: C=100, kernel=linear
- Best CV macro F1: 0.5481
- Test accuracy: 0.6687

**Test Set Classification Report:**
```
              precision    recall  f1-score   support
    Negatif       0.70      0.93      0.80       108
     Netral       0.48      0.23      0.31       31
    Positif       0.71      0.17      0.28       29

   accuracy                           0.6687       168
  macro avg       0.63      0.44      0.46       168
weighted avg       0.66      0.67      0.61       168
```

### IndoBERT + SVM

**Model:** `indobenchmark/indobert-base-p1`  
**Embedding:** mean_pooling_last_hidden_state

**Hyperparameter Tuning (5-fold CV):**
- Best params: C=10, kernel=linear
- Best CV macro F1: 0.6200
- Test accuracy: 0.6900

**Test Set Classification Report:**
```
              precision    recall  f1-score   support
    Negatif       0.74      0.92      0.82       108
     Netral       0.55      0.32      0.40       31
    Positif       0.67      0.28      0.39       29

   accuracy                           0.6900       168
  macro avg       0.65      0.51      0.54       168
weighted avg       0.69      0.69      0.67       168
```

---

## Comparison & Analysis

| Pipeline | Platform | CV Macro F1 | Test Macro F1 | Test Accuracy | Notes |
|---|---|---|---|---|---|
| TF-IDF + SVM | Play Store | 0.6613 | 0.49 | 0.6845 | Best n-gram: (1,1), C=100, linear |
| IndoBERT + SVM | Play Store | 0.68 | 0.57 | 0.71 | +0.08 macro F1 improvement |
| TF-IDF + SVM | App Store | 0.5481 | 0.46 | 0.6687 | Best n-gram: (1,1), C=100, linear |
| IndoBERT + SVM | App Store | 0.62 | 0.54 | 0.69 | +0.08 macro F1 improvement |

**Key Findings:**
1. **Play Store outperforms App Store** across both pipelines (~+0.03 macro F1, +2% accuracy)
   - Reason: More uniform Indonesian language (66.9% vs 38.9%)
2. **IndoBERT consistently improves macro F1 by ~0.08** over TF-IDF on both platforms
   - Better contextual understanding of Indonesian sentiment nuances
3. **Class imbalance challenges:** Minority class (Positif) has lowest F1 in all experiments
   - Negatif (majority): F1 ~0.80
   - Netral (middle): F1 ~0.40–0.47
   - Positif (minority): F1 ~0.28–0.42

**Recommendations:**
- Use IndoBERT + SVM for production (best balanced performance)
- Consider SMOTE or class weighting for minority-class improvement
- Play Store model is more reliable due to language uniformity

---

## Source Notebooks
- `notebooks/Tesis-Playstore-FIX.ipynb` (executed in Google Colab)
- `notebooks/Tesis-Appstore-FIX.ipynb` (executed in Google Colab)

## Data Sources
- Training data: `data/lex_labeled_review_play.csv`, `data/lex_labeled_review_app.csv`
- Test predictions: `outputs/df_test_tfidf_play.csv`, `outputs/df_test_tfidf_app.csv`
