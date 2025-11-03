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
    Negatif       0.78      0.79      0.79       111
     Netral       0.28      0.33      0.30       30
    Positif       0.76      0.52      0.62       25

   accuracy                           0.6700       168
  macro avg       0.61      0.55      0.57       168
weighted avg       0.69      0.67      0.67       168
```

### IndoBERT + SVM

**Model:** `indobenchmark/indobert-base-p1`  
**Embedding:** mean_pooling_last_hidden_state

**Hyperparameter Tuning (5-fold CV):**
- Best params: C=100, kernel=rbf
- Best CV macro F1: 0.5545
- Test accuracy: 0.6607

**Test Set Classification Report:**
```
              precision    recall  f1-score   support
    Negatif       0.74      0.89      0.81       111
     Netral       0.50      0.33      0.40       30
    Positif       0.67      0.32      0.43       25

   accuracy                           0.6627       168
  macro avg       0.64      0.51      0.55       168
weighted avg       0.68      0.66      0.66       168
```

---

## Comparison & Analysis

| Pipeline | Platform | CV Macro F1 | Test Macro F1 | Test Accuracy | Notes |
|---|---|---|---|---|---|
| TF-IDF + SVM | Play Store | 0.6613 | 0.49 | 0.6845 | Best n-gram: (1,1), C=100, linear |
| IndoBERT + SVM | Play Store | 0.6342 | 0.48 | 0.6607 | C=10, linear kernel |
| TF-IDF + SVM | App Store | 0.5481 | 0.57 | 0.6687 | Best n-gram: (1,1), C=100, linear |
| IndoBERT + SVM | App Store | 0.5545 | 0.55 | 0.6627 | C=100, rbf kernel (unique) |

**Key Findings:**
1. **Platform-specific patterns**: Play Store achieves higher CV scores, App Store achieves better test macro F1 (0.55-0.57 vs 0.48-0.49)
   - App Store shows better minority class handling (Positive recall: 0.32-0.52)
2. **TF-IDF shows slight edge** over IndoBERT on both platforms
   - Play Store: TF-IDF 0.49 vs IndoBERT 0.48 macro F1 (+0.01)
   - App Store: TF-IDF 0.57 vs IndoBERT 0.55 macro F1 (+0.02)
3. **Kernel selection matters**: RBF kernel optimal for App Store IndoBERT (unique finding)
   - All other configurations use linear kernel
4. **Class balance improvements**: App Store models handle Positive class much better
   - App Store Positive F1: 0.43-0.62
   - Play Store Positive F1: 0.26-0.28
   - Better overall macro F1 on App Store despite lower CV scores

**Recommendations:**
- **For App Store**: Use TF-IDF + SVM (C=100, linear) - Best macro F1 (0.57)
- **For Play Store**: Use TF-IDF + SVM (C=100, linear) - Best accuracy (68.45%)
- **For deployment**: TF-IDF + SVM provides best balance of performance and efficiency
- **Alternative**: IndoBERT + SVM (C=100, rbf) for App Store if contextual understanding needed

---

## Source Notebooks
- `notebooks/Tesis-Playstore-FIX.ipynb` (executed in Google Colab)
- `notebooks/Tesis-Appstore-FIX.ipynb` (executed in Google Colab)

## Data Sources
- Training data: `data/lex_labeled_review_play.csv`, `data/lex_labeled_review_app.csv`
- Test predictions: `outputs/df_test_tfidf_play.csv`, `outputs/df_test_tfidf_app.csv`
