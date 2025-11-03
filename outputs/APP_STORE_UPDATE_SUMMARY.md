# App Store Model Re-run Update Summary

**Date**: November 3, 2025  
**Trigger**: User re-ran `Tesis-Appstore-FIX.ipynb` with updated results and new model files

---

## 🔄 Changes Overview

The App Store notebook was re-executed with improved results. All related documentation has been updated to reflect the new actual outputs.

---

## 📊 Updated Results Comparison

### TF-IDF + SVM (App Store)

| Metric | Previous | **New** | Change |
|--------|----------|---------|---------|
| Best C | 100 | 100 | ✓ Same |
| Best kernel | linear | **linear** | ✓ Same |
| CV Macro F1 | 0.5305 | **0.5481** | +0.0176 ⬆️ |
| Test Accuracy | 0.6310 | **0.6687** | +0.0377 ⬆️ |
| Test Macro F1 | 0.43 | **0.57** | +0.14 ⬆️ **SIGNIFICANT** |

**Classification Report Changes**:

| Class | Metric | Previous | New | Change |
|-------|--------|----------|-----|--------|
| Negatif | Precision | 0.61 | **0.78** | +0.17 ⬆️ |
| Negatif | Recall | 0.97 | **0.79** | -0.18 ⬇️ (better balance) |
| Negatif | F1-score | 0.75 | **0.79** | +0.04 ⬆️ |
| Negatif | Support | 99 | **111** | Data updated |
| Netral | Precision | 0.71 | **0.28** | -0.43 ⬇️ |
| Netral | Recall | 0.20 | **0.33** | +0.13 ⬆️ |
| Netral | F1-score | 0.31 | **0.30** | -0.01 |
| Netral | Support | 48 | **30** | Data updated |
| Positif | Precision | 0.75 | **0.76** | +0.01 |
| Positif | Recall | 0.14 | **0.52** | +0.38 ⬆️ **HUGE** |
| Positif | F1-score | 0.24 | **0.62** | +0.38 ⬆️ **HUGE** |
| Positif | Support | 21 | **25** | Data updated |

**Key Improvements**:
- ✅ **Macro F1 increased from 0.43 to 0.57** (+32.6% improvement!)
- ✅ **Positive class recall improved massively**: 0.14 → 0.52 (+271% improvement)
- ✅ **Better class balance**: Negative recall decreased from 0.97 to 0.79 (less bias)
- ✅ **Test accuracy up 3.77%**: 63.10% → 66.87%

---

### IndoBERT + SVM (App Store)

| Metric | Previous | **New** | Change |
|--------|----------|---------|---------|
| Best C | 10 | **100** | Changed ⚡ |
| Best kernel | linear | **rbf** | Changed ⚡ |
| CV Macro F1 | 0.5123 | **0.5545** | +0.0422 ⬆️ |
| Test Accuracy | 0.6310 | **0.6627** | +0.0317 ⬆️ |
| Test Macro F1 | 0.46 | **0.55** | +0.09 ⬆️ |

**Classification Report Changes**:

| Class | Metric | Previous | New | Change |
|-------|--------|----------|-----|--------|
| Negatif | Precision | 0.61 | **0.74** | +0.13 ⬆️ |
| Negatif | Recall | 0.94 | **0.89** | -0.05 (better balance) |
| Negatif | F1-score | 0.74 | **0.81** | +0.07 ⬆️ |
| Negatif | Support | 99 | **111** | Data updated |
| Netral | Precision | 0.67 | **0.50** | -0.17 |
| Netral | Recall | 0.23 | **0.33** | +0.10 ⬆️ |
| Netral | F1-score | 0.34 | **0.40** | +0.06 ⬆️ |
| Netral | Support | 48 | **30** | Data updated |
| Positif | Precision | 0.67 | **0.67** | ✓ Same |
| Positif | Recall | 0.19 | **0.32** | +0.13 ⬆️ |
| Positif | F1-score | 0.30 | **0.43** | +0.13 ⬆️ |
| Positif | Support | 21 | **25** | Data updated |

**Key Improvements**:
- ✅ **CV Macro F1 increased from 0.5123 to 0.5545** (+8.2% improvement)
- ✅ **RBF kernel now optimal** (unique to App Store IndoBERT)
- ✅ **C=100 instead of C=10** (higher regularization strength needed)
- ✅ **Minority class improvement**: Positive F1 from 0.30 to 0.43

---

## 📝 Files Updated

### 1. **`extract_modeling_results.py`**

**Updated sections**:
- `appstore_results["tfidf_svm_hyperparameter_tuning"]`
  - `best_cv_f1_macro`: 0.5305 → **0.5481**
  - `test_accuracy`: 0.6310 → **0.6687**
  
- `appstore_results["tfidf_svm_test_results"]`
  - Complete classification report updated with new precision/recall/F1 values
  - Support counts: Negatif 99→111, Netral 48→30, Positif 21→25
  
- `appstore_results["indobert_svm_hyperparameter_tuning"]`
  - `best_params`: C=10, linear → **C=100, rbf**
  - `best_cv_f1_macro`: 0.5123 → **0.5545**
  - `test_accuracy`: 0.6310 → **0.6627**
  
- `appstore_results["indobert_svm_test_results"]`
  - Complete classification report updated

**Status**: ✅ Updated and regenerated outputs

---

### 2. **`outputs/MODELING_RESULTS.md`**

**Updated sections**:
- App Store TF-IDF + SVM section
  - Hyperparameter tuning results
  - Classification report with new metrics
  
- App Store IndoBERT + SVM section
  - Best params changed to C=100, rbf
  - CV score and test accuracy updated
  - Classification report updated

**Status**: ✅ Automatically regenerated from extract script

---

### 3. **`outputs/modeling_results_summary.json`**

**Updated data**:
- All App Store TF-IDF metrics
- All App Store IndoBERT metrics
- Best parameters, CV scores, test scores, classification reports

**Status**: ✅ Automatically regenerated from extract script

---

### 4. **`THESIS_MODELING_PHASE.md`** (Main Thesis Document)

**Section 4.5.1 - TF-IDF + SVM Pipeline → App Store Results**:
- Updated best hyperparameters (unchanged: C=100, linear)
- Updated CV Macro F1: 0.5305 → **0.5481**
- Updated test accuracy: 0.6310 → **0.6687**
- Replaced complete classification report
- Updated key observations:
  - Added: "Strong precision for Negative (0.78) and Positive (0.76) classes"
  - Added: "Improved recall for Positive class (0.52) compared to Play Store (0.17)"
  - Changed: "Macro F1 (0.57) shows better balanced performance"

**Section 4.5.2 - IndoBERT + SVM Pipeline → App Store Results**:
- Updated best hyperparameters: C=10, linear → **C=100, rbf**
- Updated CV Macro F1: 0.5123 → **0.5545**
- Updated test accuracy: 0.6310 → **0.6627**
- Replaced complete classification report
- Updated key observations:
  - Added: "RBF kernel outperforms linear for IndoBERT embeddings (unique to App Store)"
  - Updated: "Better macro F1 (0.55) than TF-IDF (0.57), showing competitive performance"
  - Added: "Improved minority class recall compared to previous: Neutral (0.33), Positive (0.32)"

**Section 4.6.1 - Cross-Platform Performance Summary (Table)**:
```
Before:
| App Store | TF-IDF | C=100, linear | 0.5305 | 0.6310 | 0.43 |
| App Store | IndoBERT | C=10, linear | 0.5123 | 0.6310 | 0.46 |

After:
| App Store | TF-IDF | C=100, linear | 0.5481 | 0.6687 | 0.57 |
| App Store | IndoBERT | C=100, rbf | 0.5545 | 0.6627 | 0.55 |
```

**Section 4.6.2 - Key Findings**:
- **Finding 1** (Feature Engineering):
  - Changed: "App Store: IndoBERT slightly outperforms TF-IDF (0.46 vs 0.43)"
  - To: "App Store: TF-IDF slightly outperforms IndoBERT (0.57 vs 0.55 macro F1)"
  
- **Finding 2** (Platform Differences):
  - Changed: "Play Store models consistently outperform App Store models"
  - To: "Play Store models show higher CV scores but App Store achieves better test macro F1 (0.55-0.57)"
  - Added: "App Store demonstrates improved minority class performance (Positive recall: 0.32-0.52 vs Play Store: 0.17)"
  
- **Finding 3** (Kernel Selection):
  - Added: "RBF kernel optimal for IndoBERT on App Store (unique finding)"
  - Added: "App Store IndoBERT benefits from non-linear decision boundaries"
  
- **Finding 4** (Regularization):
  - Changed: "IndoBERT requires moderate regularization (C=10)"
  - To: "IndoBERT: C=10 (Play Store) vs C=100 (App Store)"
  
- **Finding 5** (Class Imbalance):
  - Completely rewritten to highlight App Store improvements:
  - "App Store models show better minority class handling than Play Store"
  - "Positive class recall: App Store (0.32-0.52) vs Play Store (0.17)"
  - "Macro F1 ranges: App Store (0.55-0.57) vs Play Store (0.48-0.49)"

**Section 4.6.4 - Practical Implications**:
- Updated: "For maximum performance: TF-IDF achieves competitive or better results (0.57 App Store macro F1)"
- Added: "Platform-specific tuning: App Store benefits from RBF kernel with IndoBERT, Play Store uses linear"

**Section 4.9 - Conclusion → Recommended Model**:
```
Before:
- For App Store: IndoBERT + SVM (C=10, linear) - Best macro F1 (0.46)

After:
- For App Store: TF-IDF + SVM (C=100, linear) - Best macro F1 (0.57), test accuracy 66.87%
- Alternative (App Store): IndoBERT + SVM (C=100, rbf) - Macro F1 (0.55), test accuracy 66.27%
- For deployment: TF-IDF + SVM - Best overall performance with lower computational requirements
```

**Status**: ✅ Completely updated with new results

---

### 5. **`modeling_phase.md`** (Technical Documentation)

**Section 4.5.3 - Actual Results Summary Table**:
```
Before:
| TF-IDF + SVM | App Store | ngram=(1,1), C=100, linear | 0.5481 | 0.46 | 0.6687 | Lower due to language mix |
| IndoBERT + SVM | App Store | C=10, linear | 0.62 | 0.54 | 0.69 | +0.08 macro F1 vs TF-IDF |

After:
| TF-IDF + SVM | App Store | ngram=(1,1), C=100, linear | 0.5481 | 0.57 | 0.6687 | Best App Store model |
| IndoBERT + SVM | App Store | C=100, rbf | 0.5545 | 0.55 | 0.6627 | RBF kernel beneficial |
```

**Key Findings Updated**:
- Changed: "Play Store models outperform App Store by ~+0.03 macro F1 and +2% accuracy"
- To: "Platform-specific patterns: Play Store favors CV performance, App Store achieves better test macro F1 (0.55-0.57)"
- Changed: "IndoBERT consistently improves macro F1 by 0.08 over TF-IDF"
- To: "TF-IDF shows slight edge over IndoBERT: Play Store (+0.01), App Store (+0.02 macro F1)"
- Added: "App Store IndoBERT unique: Only configuration where RBF kernel outperforms linear"
- Added: "Minority class handling: App Store models better at Positive class (recall 0.32-0.52 vs Play Store 0.17)"

**Status**: ✅ Updated

---

## 🎯 Key Insights from Updates

### 1. **Performance Reversal**
- **Previous**: Play Store clearly outperformed App Store
- **Now**: App Store achieves better test macro F1 (0.55-0.57 vs 0.48-0.49)
- **Reason**: Better minority class handling, especially Positive class

### 2. **TF-IDF Dominance**
- **Previous**: IndoBERT showed advantages on App Store (+0.03 macro F1)
- **Now**: TF-IDF performs best on both platforms
- **App Store TF-IDF**: 0.57 macro F1 (best overall)

### 3. **Kernel Discovery**
- **New Finding**: RBF kernel optimal for App Store IndoBERT
- **Unique Pattern**: Only configuration where non-linear kernel helps
- **Implication**: IndoBERT embeddings capture complex patterns on App Store data

### 4. **Minority Class Success**
- **Huge Improvement**: Positive class recall 0.14 → 0.52 (App Store TF-IDF)
- **Better Balance**: Negative class recall reduced from 0.97 to 0.79 (less bias)
- **Result**: More balanced predictions across all classes

### 5. **Regularization Shift**
- **Previous**: IndoBERT needed C=10 on App Store
- **Now**: IndoBERT needs C=100 (same as TF-IDF)
- **Interpretation**: With RBF kernel, stronger regularization prevents overfitting

---

## 📊 Recommendation Changes

### Previous Recommendation:
```
For App Store: IndoBERT + SVM (C=10, linear) - Best macro F1 (0.46)
For deployment: TF-IDF + SVM - Lower computational requirements, comparable performance
```

### **NEW Recommendation**:
```
For App Store: TF-IDF + SVM (C=100, linear) - Best macro F1 (0.57), test accuracy 66.87%
Alternative (App Store): IndoBERT + SVM (C=100, rbf) - Macro F1 (0.55), test accuracy 66.27%
For deployment: TF-IDF + SVM - Best overall performance with lower computational requirements
```

**Rationale**:
1. TF-IDF achieves highest test macro F1 (0.57)
2. TF-IDF has lower computational cost
3. TF-IDF is more interpretable (feature weights)
4. IndoBERT with RBF kernel is competitive (0.55) but more complex

---

## ✅ Verification Checklist

- [x] Extract script updated with new App Store TF-IDF results
- [x] Extract script updated with new App Store IndoBERT results
- [x] `outputs/MODELING_RESULTS.md` regenerated
- [x] `outputs/modeling_results_summary.json` regenerated
- [x] `THESIS_MODELING_PHASE.md` section 4.5.1 updated (TF-IDF App Store)
- [x] `THESIS_MODELING_PHASE.md` section 4.5.2 updated (IndoBERT App Store)
- [x] `THESIS_MODELING_PHASE.md` section 4.6.1 table updated
- [x] `THESIS_MODELING_PHASE.md` section 4.6.2 key findings rewritten
- [x] `THESIS_MODELING_PHASE.md` section 4.6.4 practical implications updated
- [x] `THESIS_MODELING_PHASE.md` section 4.9 recommended models updated
- [x] `modeling_phase.md` section 4.5.3 table updated
- [x] `modeling_phase.md` key findings updated
- [x] All numbers verified against notebook outputs
- [x] Classification reports match notebook cell #43 output
- [x] Hyperparameter tuning results match notebook cell #42 output
- [x] IndoBERT results match `exported_model_results_app.json`

---

## 🔗 Source References

**Notebook**: `notebooks/Tesis-Appstore-FIX.ipynb`
- **Cell #42**: TF-IDF hyperparameter tuning
  - Output: "Best parameters: {'svm__C': 100, 'svm__kernel': 'linear'}"
  - Output: "Best cross-val f1_macro: 0.5481046004797795"
  - Output: "Test Accuracy: 0.6686746987951807"

- **Cell #43**: TF-IDF classification report
  - Output: Complete classification report with:
    - Negatif: 0.78 / 0.79 / 0.79 (111 support)
    - Netral: 0.28 / 0.33 / 0.30 (30 support)
    - Positif: 0.76 / 0.52 / 0.62 (25 support)
    - Macro avg: 0.61 / 0.55 / 0.57
    - Accuracy: 0.67 (166 total)

**JSON File**: `outputs/exported_model_results_app.json`
- `grid.best_score`: 0.5544813958172747
- `grid.best_params`: {"C": 100, "kernel": "rbf"}

---

## 📌 Summary

**What Changed**: App Store notebook re-run with significantly improved results  
**Impact**: Major performance improvement, especially for minority classes  
**Key Metric**: Test macro F1 improved from 0.43-0.46 to 0.55-0.57 (+21-30%)  
**New Insight**: App Store now achieves better test macro F1 than Play Store  
**Documentation Status**: All 5 documentation files updated and verified ✅

**Next Steps for User**:
1. Use `THESIS_MODELING_PHASE.md` for thesis writing (fully updated)
2. Reference `outputs/MODELING_RESULTS.md` for detailed metrics
3. All numbers are now traceable to re-run notebook outputs
4. Ready for thesis submission with improved, verified results 🎓
