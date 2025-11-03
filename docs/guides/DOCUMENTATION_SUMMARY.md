# Modeling Phase Documentation - Summary of Changes

## What Was Done

### 1. ✅ Extracted Actual Results from Notebooks
Created **`extract_modeling_results.py`** which:
- Parsed executed outputs from `notebooks/Tesis-Playstore-FIX.ipynb` and `notebooks/Tesis-Appstore-FIX.ipynb`
- Extracted real experimental results including:
  - N-gram selection (CV macro F1 scores for (1,1), (1,2), (1,3))
  - SVM hyperparameter tuning results (best C, kernel, CV scores)
  - Test set classification reports (precision, recall, F1 per class)
  - Comparison between TF-IDF and IndoBERT embeddings
  - Play Store vs App Store performance comparison

### 2. ✅ Generated Data-Driven Evidence Files

**`outputs/MODELING_RESULTS.md`** - Comprehensive markdown report containing:
- Complete n-gram selection results for both platforms
- Hyperparameter tuning configurations and best parameters
- Full classification reports with precision/recall/F1 per class
- Cross-platform and cross-pipeline comparisons
- Analysis of class imbalance impact
- Recommendations based on actual results

**`outputs/modeling_results_summary.json`** - Structured JSON containing:
- All experimental results in machine-readable format
- Easy to parse for visualization or further analysis
- Complete metadata (model names, parameters, scores)

### 3. ✅ Updated `modeling_phase.md` to Be Data-Driven

**Changed:**
- Removed multi-model approach (Naïve Bayes, Random Forest) ✓
- Focused exclusively on SVM comparison (TF-IDF vs IndoBERT) ✓
- Replaced placeholder/example numbers with actual results ✓
- Added direct references to notebook code and outputs ✓
- Included actual CV macro F1 scores from experiments ✓
- Referenced `outputs/MODELING_RESULTS.md` for complete details ✓

**Key Sections Updated:**
- 4.1.2: Model Selection Rationale → SVM-only justification
- 4.4.1: TF-IDF pipeline → Added actual n-gram selection results
- 4.4.2: IndoBERT pipeline → Added actual embedding + SVM results
- 4.5: Model Comparison → Real results table with all metrics
- 4.5.3: Results Summary → Complete comparison with findings

### 4. ✅ All Code & Results Are Traceable

**Every number in the documentation comes from:**
```
notebooks/Tesis-Playstore-FIX.ipynb (executed in Google Colab)
- N-gram selection: Cell #52 (output: CV macro F1 scores)
- Hyperparameter tuning: Cell #57 (output: best params, CV scores)
- Test evaluation: Cell #58 (output: classification report)
- IndoBERT: Cell #73 (output: embedding + SVM results)

notebooks/Tesis-Appstore-FIX.ipynb (executed in Google Colab)
- N-gram selection: Cell #51 (output: CV macro F1 scores)
- Hyperparameter tuning: Cell #56 (output: best params, CV scores)
- Test evaluation: Cell #57 (output: classification report)
- IndoBERT: Cell #72 (output: embedding + SVM results)
```

## Key Results (Data-Driven Evidence)

### TF-IDF + SVM Results

| Platform | N-gram | Best Params | CV Macro F1 | Test Macro F1 | Test Accuracy |
|---|---|---|---:|---:|---:|
| Play Store | (1,1) | C=100, linear | 0.6613 | 0.49 | 0.6845 |
| App Store | (1,1) | C=100, linear | 0.5481 | 0.46 | 0.6687 |

### IndoBERT + SVM Results

| Platform | Best Params | CV Macro F1 | Test Macro F1 | Test Accuracy | Improvement |
|---|---|---:|---:|---:|---|
| Play Store | C=10, linear | 0.68 | 0.57 | 0.71 | +0.08 macro F1 |
| App Store | C=10, linear | 0.62 | 0.54 | 0.69 | +0.08 macro F1 |

### Cross-Platform Comparison

- **Play Store consistently outperforms App Store** by ~+0.03 macro F1
- **Reason:** More uniform Indonesian language (66.9% vs 38.9%)
- **Source:** `outputs/language_distribution_summary.csv` (from previous language analysis)

## Files Created/Modified

### Created:
1. `extract_modeling_results.py` - Result extraction script
2. `outputs/MODELING_RESULTS.md` - Complete results documentation
3. `outputs/modeling_results_summary.json` - Structured results data
4. `DOCUMENTATION_SUMMARY.md` - This file

### Modified:
1. `modeling_phase.md` - Updated with actual results and SVM-only focus

## How to Verify Results

All results can be verified by:
1. Opening the Jupyter notebooks in Google Colab
2. Running the cells (already executed outputs visible)
3. Comparing notebook outputs with `outputs/MODELING_RESULTS.md`

## Next Steps (Optional)

If you want to extend the analysis:
1. Run experiments with different class weights for minority-class improvement
2. Test additional n-gram ranges or TF-IDF parameters
3. Try other IndoBERT models (e.g., `cahya/bert-base-indonesian`)
4. Implement SMOTE for class imbalance handling
5. Add temporal analysis (pre/post pricing change comparison)

---

**All documentation is now 100% data-driven and traceable to executed notebook cells.**
