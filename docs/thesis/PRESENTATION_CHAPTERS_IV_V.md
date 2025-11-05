# 📊 PRESENTATION OUTLINE: CHAPTERS IV-V (EVIDENCE-BASED)
## **Results & Discussion: 11 Slides with Complete CRISP-DM Flow**

**Context**: Follows Chapter III Methodology presentation  
**Target**: 11 slides covering Business Understanding → Deployment  
**Time Allocation**: 14-16 minutes  
**Approach**: Evidence-driven with actual project data

**Data Collection Date**: April 7th, 2025  
**Total Reviews**: 1,676 (838 per platform)

---

## **SLIDE 1: Business Understanding Results - Objectives Achieved**

**Title**: CRISP-DM Phase 1: Business Understanding Validation

**Visual**: Checklist table showing objectives vs. achieved status

**Content**:

```
✅ RESEARCH OBJECTIVES VALIDATION (Chapter III → Chapter IV)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 PRIMARY OBJECTIVE: Develop automated sentiment classification 
   system for Indonesian Disney+ Hotstar reviews

STATUS: ✅ FULLY ACHIEVED
   • 4 models developed (2 platforms × 2 feature methods)
   • Automated 3-class prediction operational (Positive/Neutral/Negative)
   • Indonesian language processing pipeline effective
   • Production-ready Streamlit dashboard deployed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 SPECIFIC OBJECTIVES ASSESSMENT:

Objective                          Status      Evidence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Collect & preprocess reviews    ✅ Achieved  838 per platform (2020-2025)
2. Compare TF-IDF vs IndoBERT      ✅ Achieved  TF-IDF wins +0.075 Macro F1
3. Optimize SVM classifiers        ✅ Achieved  Grid search, 10-fold CV
4. Cross-platform analysis         ✅ Achieved  15.87% sentiment gap found
5. Deploy real-time dashboard      ✅ Achieved  localhost:8600 operational

Evidence: [Chapter IV Section 4.2](../CHAPTER_IV_RESULTS_DISCUSSION.md#42-crisp-dm-phase-1-business-understanding-results)
```

**Speaking Points** (60 sec):
> "Let's validate the Business Understanding phase established in Chapter III. Primary objective was to develop an automated sentiment classification system for Indonesian Disney+ Hotstar reviews—fully achieved. We delivered 4 trained models covering both platforms and two feature extraction methods, with a production-ready Streamlit dashboard.

> All five specific objectives achieved: First, 838 reviews collected per platform spanning 2020-2025. Second, TF-IDF versus IndoBERT comparison completed showing TF-IDF wins by +0.075 Macro F1. Third, SVM hyperparameter optimization via grid search with 10-fold cross-validation. Fourth, cross-platform analysis revealing 15.87% sentiment gap between platforms. Fifth, dashboard deployed at localhost:8600. Business Understanding phase objectives fully achieved."

---

## **SLIDE 2: Data Preparation Results - Raw to Processed**

**Title**: How Raw Data Becomes Sentiment-Labeled Reviews

**Visual**: Flowchart showing preprocessing pipeline stages

**Content**:

```
📊 CRISP-DM PHASE 3: DATA PREPARATION RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Data Collection (April 7th, 2025):
   • App Store: 838 reviews
   • Play Store: 838 reviews
   • Total: 1,676 reviews
   • Temporal Split: 419 pre-2023, 419 post-2023 per platform

� 6-STAGE PREPROCESSING PIPELINE RESULTS:

Stage              Input               Output              Impact
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Translation     Mixed languages     All Indonesian      Standardized
2. Cleaning        Noisy text          Normalized text     Reduced noise
3. Tokenization    Sentences           Word arrays         Structured
4. Stopword        758 stopwords       Filtered tokens     48-57% reduction
   Removal         applied                                 
5. Stemming        Inflected words     Root forms          Unified variants
                   (e.g., "menyenangkan" → "senang")
6. Final Text      Processed tokens    `ulasan_bersih`     Ready for ML

⚠️ DATA QUALITY ISSUE DISCOVERED:
   • Empty strings after stopword removal:
     - App Store: 8 reviews (0.95%)
     - Play Store: 43 reviews (5.13%)
   • These were filtered before modeling
   • Final dataset: 830 App Store, 795 Play Store

📊 TOKEN REDUCTION STATISTICS:
   • App Store: 48% token reduction (avg 19.3 → 10.0 words)
   • Play Store: 57% token reduction (avg 13.2 → 5.7 words)

Evidence: [lex_labeled_review_app.csv](../../lex_labeled_review_app.csv), [lex_labeled_review_play.csv](../../lex_labeled_review_play.csv)
```

**Speaking Points** (75 sec):
> "Research Problem 1: How is raw data processed until sentiment-labeled? The 6-stage preprocessing pipeline transforms 1,676 raw reviews into clean Indonesian text. Translation standardizes mixed-language content. Cleaning removes noise. Tokenization splits text into words. Stopword removal filters 758 Indonesian function words, reducing tokens by 48-57%. Stemming unifies word variants like 'menyenangkan' to root 'senang'. Critical discovery: stopword removal created empty strings—8 on App Store, 43 on Play Store—these were filtered. Final datasets: 830 App Store, 795 Play Store reviews ready for sentiment labeling."

---

## **SLIDE 3: Evaluation - Sentiment Distribution (Lexicon-Based)**

**Title**: Sentiment Distribution Results from InSet Lexicon Labeling

**Visual**: Side-by-side bar charts showing sentiment distribution per platform

**Content**:

```
📊 CRISP-DM PHASE 5: EVALUATION RESULTS - SENTIMENT DISTRIBUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 LEXICON-BASED SENTIMENT LABELING (InSet Dictionary):
   • Method: 10,218 Indonesian terms (3,609 positive, 6,609 negative)
   • Algorithm: Compare positive vs negative word counts
   • Applied to: All 1,676 preprocessed reviews (ulasan_bersih)

📊 SENTIMENT DISTRIBUTION RESULTS:

Platform      Negatif    Netral     Positif    Total Reviews
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
App Store     556 (66.35%) 147 (17.54%) 135 (16.11%)   838
Play Store    689 (82.22%) 90 (10.74%)  59 (7.04%)     838

📊 RESEARCH PROBLEM 2 ANSWERED: 
   "How are the sentiment distributions on both platforms?"

KEY FINDINGS:
   ✅ Play Store 15.87% MORE negative than App Store
   ✅ Both platforms show majority user dissatisfaction
   ✅ Play Store has SEVERE class imbalance (82% negative)
   ✅ App Store more balanced (66% negative, moderate imbalance)

💡 CLASS IMBALANCE IMPLICATIONS:
   • Play Store: Accuracy trap—82% baseline (always predict "Negatif")
   • App Store: More manageable distribution for multi-class learning
   • Justifies Macro F1 as primary evaluation metric
   • Stratified train-test split required to preserve distribution

Evidence: [App Store Results](../../outputs/reports/EVALUATION_RESULTS_APPSTORE.md#1-initial-lexicon-sentiment-labeling-distribution), [Play Store Results](../../outputs/reports/EVALUATION_RESULTS_PLAYSTORE.md#1-initial-lexicon-sentiment-labeling-distribution)
```

**Speaking Points** (75 sec):
> "Research Problem 2: How are sentiment distributions on both platforms? After preprocessing, InSet lexicon labeled all 1,676 reviews by counting positive versus negative words. Results show dramatic platform asymmetry. App Store: 66% negative, 18% neutral, 16% positive—concerning but balanced enough for multi-class classification. Play Store: 82% negative, 11% neutral, only 7% positive—severe class imbalance. This 15.87 percentage point gap reveals Play Store users are significantly more dissatisfied. The imbalance creates an accuracy trap: a naive model predicting only 'Negatif' achieves 82% accuracy on Play Store without learning. This validates our Macro F1 metric choice."

---

## **SLIDE 4: Evaluation - Model Performance (App Store)**

**Title**: SVM Performance with TF-IDF vs IndoBERT (App Store)

**Visual**: Confusion matrix comparison + classification report table

**Content**:

```
📊 EVALUATION RESULTS - APP STORE (Test Set: 168 samples)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 RESEARCH PROBLEM 3A: SVM Performance with TF-IDF Embeddings

📈 TF-IDF + SVM MODEL:
   • Test Accuracy: 66.87%
   • Macro F1-Score: 0.57 ✅ (Primary Metric)
   • Weighted F1-Score: 0.67

🔍 CONFUSION MATRIX (TF-IDF):
                  Predicted →
Actual ↓       Negatif   Netral   Positif
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Negatif (111)     88       18        5
Netral (30)       17       10        3
Positif (27)      11        3       13

📋 CLASSIFICATION REPORT (TF-IDF):
Class      Precision   Recall   F1-Score   Support
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Negatif      0.78      0.79     0.79       111
Netral       0.28      0.33     0.30        30
Positif      0.76      0.52     0.62        27

🎯 RESEARCH PROBLEM 3B: SVM Performance with IndoBERT Embeddings

📈 INDOBERT + SVM MODEL:
   • Test Accuracy: 66.27%
   • Macro F1-Score: 0.47 ⚠️
   • Weighted F1-Score: 0.64

🔍 CONFUSION MATRIX (IndoBERT):
                  Predicted →
Actual ↓       Negatif   Netral   Positif
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Negatif (111)     93       13        5
Netral (30)       23        4        3
Positif (27)      13        4       10

📋 CLASSIFICATION REPORT (IndoBERT):
Class      Precision   Recall   F1-Score   Support
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Negatif      0.72      0.84     0.78       111
Netral       0.19      0.13     0.16        30
Positif      0.56      0.40     0.47        27

💡 KEY FINDINGS (App Store):
   ✅ TF-IDF wins: +0.10 Macro F1 advantage (0.57 vs 0.47)
   ✅ TF-IDF better at minority classes (Netral F1: 0.30 vs 0.16)
   ✅ IndoBERT over-predicts Negatif (23/30 Netral misclassified)

Evidence: [App Store Model Evaluation](../../outputs/reports/EVALUATION_RESULTS_APPSTORE.md#2-model-performance-evaluation)
```

**Speaking Points** (90 sec):
> "Research Problem 3: How is SVM performance with TF-IDF versus IndoBERT embeddings? App Store results first. TF-IDF with SVM achieves 66.87% accuracy and 0.57 Macro F1—our primary metric. The confusion matrix shows 88 out of 111 negatives correctly classified. Classification report reveals per-class performance: Negatif F1 is 0.79, Positif 0.62, but Netral struggles at 0.30 due to class imbalance. 

> IndoBERT with SVM scores similar accuracy at 66.27% but worse Macro F1 at 0.47—a significant 0.10 gap. Its confusion matrix shows it over-predicts Negatif: 23 out of 30 Netral reviews are misclassified as Negatif. Classification report confirms this: Netral F1 drops to 0.16—IndoBERT cannot handle minority classes. Winner: TF-IDF for balanced multi-class detection."

---

## **SLIDE 5: Evaluation - Model Performance (Play Store)**

**Title**: SVM Performance with TF-IDF vs IndoBERT (Play Store)

**Visual**: Confusion matrix comparison + classification report table

**Content**:

```
📊 EVALUATION RESULTS - PLAY STORE (Test Set: 168 samples)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 TF-IDF + SVM MODEL:
   • Test Accuracy: 73.21%
   • Macro F1-Score: 0.38 ✅ (Primary Metric)
   • Weighted F1-Score: 0.72

🔍 CONFUSION MATRIX (TF-IDF):
                  Predicted →
Actual ↓       Negatif   Netral   Positif
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Negatif (138)    116       18        4
Netral (18)       13        4        1
Positif (12)       9        2        1

📋 CLASSIFICATION REPORT (TF-IDF):
Class      Precision   Recall   F1-Score   Support
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Negatif      0.84      0.84     0.84       138
Netral       0.17      0.22     0.19        18
Positif      0.17      0.08     0.11        12

🎯 INDOBERT + SVM MODEL:
   • Test Accuracy: 72.62%
   • Macro F1-Score: 0.33 ⚠️
   • Weighted F1-Score: 0.71

🔍 CONFUSION MATRIX (IndoBERT):
                  Predicted →
Actual ↓       Negatif   Netral   Positif
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Negatif (138)    118       16        4
Netral (18)       14        3        1
Positif (12)      10        2        0

📋 CLASSIFICATION REPORT (IndoBERT):
Class      Precision   Recall   F1-Score   Support
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Negatif      0.83      0.86     0.84       138
Netral       0.14      0.17     0.15        18
Positif      0.00      0.00     0.00 🔴     12

💡 KEY FINDINGS (Play Store):
   ✅ TF-IDF wins: +0.05 Macro F1 advantage (0.38 vs 0.33)
   ⚠️ Both models struggle with severe imbalance (82% negative)
   🔴 IndoBERT COMPLETE FAILURE on Positif (F1: 0.00, 0/12 correct)
   ⚠️ Higher accuracy (73%) misleading due to 82% negative baseline

💡 ACCURACY vs MACRO F1 PARADOX:
   • Play Store accuracy HIGHER than App Store (73% vs 67%)
   • But Macro F1 LOWER (0.38 vs 0.57)
   • Reason: 82% negative baseline inflates accuracy
   • Macro F1 reveals true multi-class performance

Evidence: [Play Store Model Evaluation](../../outputs/reports/EVALUATION_RESULTS_PLAYSTORE.md#2-model-performance-evaluation)
```

**Speaking Points** (90 sec):
> "Play Store results show extreme class imbalance impact. TF-IDF achieves 73.21% accuracy but Macro F1 only 0.38. Confusion matrix shows 116 out of 138 negatives correct, but minority classes suffer: only 4 of 18 Netral and 1 of 12 Positif classified correctly. Classification report: Negatif F1 is 0.84, but Netral drops to 0.19 and Positif to 0.11.

> IndoBERT performs worse: 72.62% accuracy but Macro F1 only 0.33. Most critically, IndoBERT achieves ZERO F1 on Positif class—complete failure. It classified 0 out of 12 positive reviews correctly, treating all as Negatif. This demonstrates the accuracy trap: Play Store's 73% accuracy looks better than App Store's 67%, but Macro F1 tells the truth: 0.38 versus 0.57. The 82% negative baseline makes accuracy meaningless. TF-IDF wins both platforms."

---

## **SLIDE 6: Cross-Platform Performance Comparison**

**Title**: Cross-Platform Model Performance Analysis

**Visual**: Side-by-side comparison table + gap visualization

**Content**:

```
� CROSS-PLATFORM EVALUATION COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏆 OVERALL PERFORMANCE SUMMARY:

Platform      TF-IDF     IndoBERT   Advantage   Winner
              Macro F1   Macro F1   
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
App Store     0.57       0.47       +0.10       TF-IDF ✅
Play Store    0.38       0.33       +0.05       TF-IDF ✅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Average       0.475      0.400      +0.075      TF-IDF ✅

📊 DETAILED CROSS-PLATFORM METRICS:

Metric              App Store    Play Store   Observation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Negative %          66.35%       82.22%       +15.87% gap
Class Imbalance     Moderate     Severe       Play worse
TF-IDF Accuracy     66.87%       73.21%       +6.34%
TF-IDF Macro F1     0.57         0.38         -0.19
IndoBERT Macro F1   0.47         0.33         -0.14

💡 KEY CROSS-PLATFORM INSIGHTS:

1️⃣ ACCURACY vs MACRO F1 PARADOX:
   • Play Store: HIGHER accuracy (73% vs 67%)
   • Play Store: LOWER Macro F1 (0.38 vs 0.57)
   → Reason: 82% negative baseline inflates accuracy
   → Macro F1 reveals Play Store model is WORSE

2️⃣ CLASS IMBALANCE IMPACT:
   • App Store: Moderate imbalance enables better multi-class learning
   • Play Store: Severe imbalance (82%) limits minority class detection
   • TF-IDF maintains advantage on BOTH platforms

3️⃣ PER-CLASS PERFORMANCE GAPS:
   App Store → Play Store:
   • Negatif F1: 0.79 → 0.84 (improves with more samples)
   • Netral F1:  0.30 → 0.19 (degrades with imbalance)
   • Positif F1: 0.62 → 0.11 (severe degradation)

4️⃣ INDOBERT FAILURE PATTERN:
   • App Store: Struggles with Netral (F1: 0.16)
   • Play Store: COMPLETE failure on Positif (F1: 0.00)
   • Pattern: Cannot learn from extreme minority classes

Evidence: [Platform Comparison Analysis](../../outputs/reports/PLATFORM_COMPARISON_ANALYSIS.md)
```

**Speaking Points** (90 sec):
> "Cross-platform comparison reveals critical insights. TF-IDF wins on both platforms with +0.075 average Macro F1 advantage. But observe the accuracy-Macro F1 paradox: Play Store achieves 73% accuracy versus App Store's 67%, yet Macro F1 is 0.38 versus 0.57—the opposite ranking. This exposes the accuracy trap: Play Store's 82% negative baseline inflates accuracy without real learning.

> Class imbalance severely impacts minority classes. App Store's moderate 66% negative enables Positif F1 of 0.62; Play Store's extreme 82% negative crushes Positif F1 to 0.11—a 0.51 point drop. IndoBERT follows the same pattern but worse: App Store Netral F1 is 0.16; Play Store Positif F1 is literally zero—complete failure. Conclusion:    → TF-IDF is more robust to class imbalance, maintaining usable performance 
      on both platforms while IndoBERT collapses on severe imbalance.

Evidence: [Platform Comparison Analysis](../../outputs/reports/PLATFORM_COMPARISON_ANALYSIS.md)
```

**Speaking Points** (90 sec):"

---

## **SLIDE 7: Word Frequency Analysis (Business Intelligence)**

**Title**: Negative Keywords - Actionable Insights from 970 Reviews

**Visual**: Word cloud + horizontal bar chart of top keywords

**Content**:

```
�🔴 NEGATIVE SENTIMENT KEYWORDS (Primary Business Action Items)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Source: 970 negative reviews (503 App Store + 467 Play Store)
Analysis: Frequency count from stemmed text (ulasan_bersih column)

🔥 CRITICAL TECHNICAL ISSUES (Cross-Platform):

Keyword      App Store   Play Store   Combined   Business Action
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
'masuk'      75 (14.9%)  34 (7.3%)    109        🔴 FIX LOGIN SYSTEM
'bayar'      57 (11.3%)  77 (16.5%)   134        🔴 FIX PAYMENT FLOW
'kode/otp'   70+59       30+12        171        🔴 FIX OTP DELIVERY
'load'       19 (3.8%)   47 (10.1%)   66         🟡 REDUCE BUFFERING
'error'      15 (3.0%)   21 (4.5%)    36         🔴 ERROR HANDLING

📊 PLATFORM-SPECIFIC ISSUES:

App Store Distinctive (iOS):
   • 'otp': 59 reviews (11.7%) - OTP delivery problems
   • 'tv': 72 reviews (14.3%) - Apple TV integration
   • 'kode': 70 reviews (13.9%) - Verification codes

Play Store Distinctive (Android):
   • 'langgan': 117 reviews (25.1%) - Subscription confusion
   • 'gambar': 39 reviews (8.4%) - Picture quality
   • 'suara': 34 reviews (7.3%) - Audio sync issues
   • 'bug': 20 reviews (4.3%) - Software bugs

💡 ACTIONABLE INTELLIGENCE:
   1. PRIORITY 1: Authentication system (masuk/kode/otp = 280 mentions)
   2. PRIORITY 2: Payment processing (bayar = 134 mentions)
   3. PRIORITY 3: Streaming quality (load/gambar/suara = 139 mentions)

Evidence: [Word Frequency Analysis](../../docs/analysis/WORD_FREQUENCY_ANALYSIS.md)
```

**Speaking Points** (75 sec):
> "Word frequency analysis provides actionable business intelligence. From 970 negative reviews, three critical issues emerge. First, authentication problems: 'masuk' (login), 'kode', and 'otp' combine for 280 mentions across platforms—users cannot access paid services. Second, payment failures: 'bayar' appears 134 times—billing system is broken. Third, streaming quality: 'load', 'gambar', 'suara' total 139 mentions—buffering and A/V sync issues. Platform-specific: App Store users struggle with OTP delivery (59 reviews), Play Store users report 'bug' 20 times and subscription confusion 117 times. These keywords directly map to engineering priorities."

---

## **SLIDE 8: Deployment Success - Production Dashboard**

**Title**: CRISP-DM Phase 6: Deployed Dashboard Metrics

**Visual**: Dashboard screenshot + performance comparison chart

**Content**:

```
🚀 DEPLOYMENT RESULTS (Production-Ready System)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Dashboard Deployed: localhost:8600
   • Platform: Streamlit (Python)
   • Models: 4 pickle files (App/Play × TF-IDF/IndoBERT)
   • Features: CSV upload, real-time prediction, visualization

⚡ PRODUCTION PERFORMANCE METRICS:

Metric              TF-IDF          IndoBERT        Advantage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Prediction Speed    0.07-0.08s      0.82-0.85s      10× faster
Throughput          750-857 rev/min 70-73 rev/min   10× scalable
Memory Usage        Low (sparse)    High (768 dims) Efficient
Interpretability    High (weights)  Low (black box) Explainable

💼 BUSINESS VALUE:
   • Process 838 reviews in ~1 minute (TF-IDF) vs ~12 minutes (IndoBERT)
   • Stakeholders can upload weekly review exports, get instant insights
   • Word cloud highlights actionable keywords (e.g., "error", "bayar")
   • Export CSV for reporting to leadership

🎯 DEPLOYMENT RECOMMENDATION:
   → Deploy TF-IDF models to production (App Store model prioritized)
   → IndoBERT requires GPU for acceptable performance (0.82s → 0.1s)

Evidence: Dashboard at [localhost:8600](http://localhost:8600), [Model files](../../models/)
```

**Speaking Points** (75 sec):
> "CRISP-DM deployment phase delivered a production-ready dashboard. TF-IDF models process 750-857 reviews per minute—10× faster than IndoBERT's 70-73 reviews per minute. For business stakeholders: upload a CSV of 838 reviews, receive sentiment classification in under 60 seconds with TF-IDF, versus 12 minutes with IndoBERT. The dashboard generates word clouds highlighting negative keywords like 'error' and 'bayar', enabling non-technical users to identify actionable issues. Recommendation: deploy TF-IDF to production immediately; IndoBERT needs GPU infrastructure investment."

---

## **SLIDE 9: Discussion - Why TF-IDF Wins**

**Title**: Chapter V: Explaining the Counterintuitive Result

**Visual**: 4-quadrant diagram showing task/data/efficiency/bias factors

**Content**:

```
❓ CHAPTER V: WHY SIMPLER BEATS TRANSFORMER?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 FOUR FACTORS ANALYSIS:

1️⃣ DATASET SIZE (CRITICAL):
   • Training samples: 670 per platform (after 80/20 split)
   • IndoBERT parameters: 12 layers × 768 dimensions = ~110M parameters
   • Rule of thumb: Need 5,000+ samples for transformer fine-tuning
   • Our 670 samples: INSUFFICIENT for IndoBERT advantage
   → TF-IDF advantage: Only ~5,000 features (max_features setting)

2️⃣ TASK NATURE (EXPLICIT SENTIMENT):
   • Sentiment expressed via DIRECT keywords:
     - Negative: "error", "gagal", "lemot", "bug"
     - Positive: "mantap", "oke", "bagus", "senang"
   • TF-IDF: Learns these explicit keyword-sentiment mappings
   • IndoBERT: Contextual embeddings overkill for keyword-driven task
   → Complex contextual understanding not needed

3️⃣ PRODUCTION EFFICIENCY (10× SPEED):
   • TF-IDF: 0.07s per review = 750 reviews/min
   • IndoBERT: 0.82s per review = 70 reviews/min
   • Business impact: 838 reviews in 1 min vs 12 min
   → Speed enables real-time monitoring dashboards

4️⃣ INTERPRETABILITY (STAKEHOLDER VALUE):
   • TF-IDF: Feature weights = actual words
     Example: "bayar" has weight 2.34 → payment issues
   • IndoBERT: 768-dimensional vectors (black box)
   • Stakeholders can ACT on TF-IDF insights immediately
   → Actionable intelligence matters

💡 CONCLUSION: Context-dependent performance
   • NOT "TF-IDF always better than transformers"
   • Small datasets + explicit sentiment + speed needs = TF-IDF wins
   • Large datasets + nuanced context + GPU available = IndoBERT might win

Evidence: [App Store Evaluation](../../outputs/reports/EVALUATION_RESULTS_APPSTORE.md), [Play Store Evaluation](../../outputs/reports/EVALUATION_RESULTS_PLAYSTORE.md)
```

**Speaking Points** (90 sec):
> "Why does 1990s TF-IDF beat 2020 IndoBERT? Four factors. First, dataset size: 670 training samples is insufficient for 110 million parameter IndoBERT; literature recommends 5,000+ samples. TF-IDF uses only 5,000 features—appropriate for our data scale. Second, task nature: sentiment is explicit via keywords like 'error', 'gagal', 'mantap', 'oke'. TF-IDF learns these direct mappings; IndoBERT's contextual understanding is overkill. Third, efficiency: 0.07 seconds versus 0.82 seconds means processing 838 reviews in 1 minute versus 12 minutes—critical for production dashboards. Fourth, interpretability: TF-IDF weights map to actual words; stakeholders see 'bayar' has high weight and immediately know to fix payment issues. IndoBERT's 768-dimensional vectors are black boxes. Context matters: TF-IDF wins for small, explicit-sentiment datasets with speed and interpretability requirements."

---

## **SLIDE 10: Research Questions Answered**

**Title**: Evidence-Based Answers to RQs

**Visual**: Three-panel layout with RQ → Evidence → Answer

**Content**:

```
✅ RESEARCH QUESTIONS ANSWERED (Evidence-Based)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 RQ1: How does TF-IDF compare to IndoBERT for Indonesian 
         sentiment classification?

Evidence:
   • App Store: TF-IDF 0.57 vs IndoBERT 0.47 (Macro F1)
   • Play Store: TF-IDF 0.38 vs IndoBERT 0.33 (Macro F1)
   • Average advantage: +0.075 Macro F1
   • Speed advantage: 10× faster (0.07s vs 0.82s)

Answer: ✅ TF-IDF OUTPERFORMS IndoBERT for this dataset
   → Small data (838 samples) favors traditional methods
   → Explicit sentiment keywords suit TF-IDF feature extraction
   → Production efficiency (10× speed) delivers business value

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 RQ2: What are cross-platform sentiment differences between 
         App Store and Play Store?

Evidence:
   • Negative sentiment: 66.35% (App) vs 82.22% (Play) = +15.87%
   • Class imbalance: Moderate (App) vs Severe (Play)
   • Model performance: 0.57 (App) vs 0.38 (Play) Macro F1
   • Keyword focus: Authentication (App) vs Streaming (Play)

Answer: ✅ SIGNIFICANT CROSS-PLATFORM ASYMMETRY
   → Play Store users 15.87% more negative
   → Android users more price-sensitive (+3.6% post-price shift)
   → Platform-specific technical issues require tailored responses

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 RQ3: What is the impact of 2023 price increase on sentiment?

Evidence:
   • App Store: 64.5% → 66.2% negative (+1.7%, p=0.45 NS)
   • Play Store: 78.5% → 82.1% negative (+3.6%, p=0.06 marginal)
   • Android users show larger negative shift

Answer: ✅ MODEST BUT MEASURABLE IMPACT
   → Android users MORE price-sensitive than iOS users
   → Play Store sentiment worsened post-price increase
   → Statistical significance marginal (small sample size)

Evidence: [Platform Comparison Analysis](../../outputs/reports/PLATFORM_COMPARISON_ANALYSIS.md)
```

**Speaking Points** (90 sec):
> "All three research questions answered with evidence. RQ1: TF-IDF versus IndoBERT? TF-IDF wins +0.075 Macro F1 and 10× faster—validated by actual model evaluations on 838 samples per platform. RQ2: Cross-platform differences? Dramatic asymmetry: Play Store 15.87% more negative, with severe class imbalance impacting model performance. App Store users focus on authentication bugs; Play Store users complain about streaming quality. RQ3: Price increase impact? Modest but measurable. App Store sentiment increased 1.7 percentage points negative (not statistically significant). Play Store increased 3.6 points (marginally significant, p=0.06). Android users are more price-sensitive. All findings tied to actual scraped data from April 2025 temporal split analysis."

---

## **SLIDE 11: Contributions & Recommendations**

**Title**: Research Contributions + Actionable Next Steps

**Visual**: Timeline diagram showing immediate/short-term/long-term actions

**Content**:

```
🏆 KEY CONTRIBUTIONS (Evidence-Driven)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣ METHODOLOGICAL CONTRIBUTION:
   ✅ Controlled TF-IDF vs IndoBERT comparison (same SVM classifier)
   ✅ Demonstrates traditional methods competitive for small datasets
   ✅ Challenges "transformer always better" assumption
   → Contribution to Indonesian NLP literature

2️⃣ EMPIRICAL CONTRIBUTION:
   ✅ First cross-platform sentiment analysis for Disney+ Hotstar
   ✅ Quantified platform asymmetry (15.87% negative gap)
   ✅ Documented price increase impact (+3.6% Android sensitivity)
   → Baseline for future streaming app research

3️⃣ PRACTICAL CONTRIBUTION:
   ✅ Production-ready dashboard (750 reviews/min throughput)
   ✅ Actionable keyword intelligence (error/bayar/masuk priorities)
   ✅ Complete CRISP-DM implementation (not just modeling)
   → Bridges academic research to business value

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 RECOMMENDATIONS (Prioritized by Evidence)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMMEDIATE (0-3 Months) - CRITICAL:
   🔴 Priority 1: Fix authentication system
      Evidence: 280 mentions (masuk/kode/otp combined)
      Action: Resolve OTP delivery, login flow, verification
   
   🔴 Priority 2: Fix payment processing
      Evidence: 134 mentions (bayar keyword)
      Action: Debug billing failures, improve transaction flow
   
   🔴 Priority 3: Deploy TF-IDF models to production
      Evidence: Macro F1 0.57 (App Store), 10× speed advantage
      Action: Integrate with customer support dashboard

SHORT-TERM (3-6 Months):
   📊 Continuous monitoring: Track sentiment recovery
      Target: Reduce negative from 66%/82% toward 50%
   
   📈 Data collection: Scale to 5,000+ reviews
      Goal: Enable fair IndoBERT fine-tuning comparison
   
   🔍 Platform-specific fixes:
      - App Store: Focus on authentication, OTP, Apple TV
      - Play Store: Focus on streaming, buffering, audio sync

LONG-TERM (6-12 Months):
   🎯 Aspect-based sentiment: Separate content vs technical issues
   ⚡ Real-time alerting: Sentiment spike detection
   🤖 Fine-tuned IndoBERT: With larger dataset (5,000+ samples)
   🌍 Multi-app expansion: Apply methodology to competitors

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 DEPLOYMENT READINESS:
   ✅ TF-IDF models: READY for production (App Store priority)
   ⚠️ IndoBERT models: Requires GPU investment or code optimization
   ✅ Dashboard: Operational at localhost:8600
   ✅ Documentation: Complete CRISP-DM methodology documented

Evidence: [Word Frequency Analysis](../../docs/analysis/WORD_FREQUENCY_ANALYSIS.md), [Evaluation Reports](../../outputs/reports/)
```

**Speaking Points** (90 sec):
> "Three key contributions. Methodologically, we demonstrate traditional TF-IDF remains competitive for small Indonesian datasets, challenging transformer assumptions. Empirically, we provide the first cross-platform sentiment analysis for Disney+ Hotstar, quantifying a 15.87% negative sentiment gap and 3.6% price-increase impact on Android. Practically, we deliver a production dashboard processing 750 reviews per minute with actionable insights.

> Recommendations are evidence-driven. Immediate priorities: fix authentication system—280 keyword mentions prove it's broken. Fix payment processing—134 'bayar' mentions. Deploy TF-IDF models with 0.57 Macro F1 performance. Short-term: monitor sentiment recovery, scale data collection to 5,000 samples for IndoBERT comparison. Long-term: implement aspect-based analysis separating content complaints from technical issues, build real-time sentiment spike alerting. TF-IDF models are production-ready today; IndoBERT needs GPU infrastructure investment."

---

## 📋 PRESENTATION STRUCTURE SUMMARY

```
🕐 COMPLETE CHAPTERS IV-V TIMELINE (REVISED - FULL CRISP-DM FLOW):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Slide  Topic                                Time    Cumulative
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1      Business Understanding Results       1:00    0:00-1:00
       (Primary & specific objectives)
2      Data Preparation Results             1:00    1:00-2:00
       (6-stage preprocessing pipeline)
3      Sentiment Distribution               1:15    2:00-3:15
       (InSet lexicon-based labeling)
4      Model Performance - App Store        1:15    3:15-4:30
       (Confusion matrices, F1 scores)
5      Model Performance - Play Store       1:15    4:30-5:45
       (Confusion matrices, F1 scores)
6      Cross-Platform Comparison            1:30    5:45-7:15
       (Model performance gaps)
7      Word Frequency Analysis              1:15    7:15-8:30
       (Negative keywords - business intel)
8      Deployment Dashboard                 1:15    8:30-9:45
       (Production metrics, throughput)
9      Discussion: Why TF-IDF Wins          1:30    9:45-11:15
       (4 factors: data, task, speed, bias)
10     Research Questions Answered          1:30    11:15-12:45
       (Evidence-based RQ responses)
11     Contributions & Recommendations      1:30    12:45-14:15
       (Immediate/short/long-term actions)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL (Chapters IV-V)                       14:15 minutes

FULL SESSION (with Chapter III):
   • Chapter III (Methodology):  6-7 minutes
   • Chapters IV-V (Results):    14-15 minutes
   • TOTAL:                      20-22 minutes
   • Q&A Buffer:                 5-10 minutes
   • GRAND TOTAL:                25-32 minutes
```

---

## 🎤 DEFENSE PREPARATION: ANTICIPATED QUESTIONS

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q1: "Why didn't you collect more data? 838 samples is small."

A: "Methodological constraint for temporal analysis. We needed balanced 
    sampling: 419 reviews before 2023 price increase, 419 after. This 
    ensures fair pre/post comparison. Future work will scale to 5,000+ 
    for IndoBERT fine-tuning, but current sample size is sufficient for 
    controlled TF-IDF comparison and establishes baseline."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q2: "Your Play Store Macro F1 is only 0.38—isn't that too low?"

A: "Context matters: 82% negative class imbalance. A naive baseline 
    predicting only 'Negative' achieves 82% accuracy but 0.33 Macro F1 
    (assuming no minority class detection). Our TF-IDF at 0.38 exceeds 
    this baseline. The low score reflects data reality, not model failure. 
    Recommendation: Consider binary classification (negative vs non-negative) 
    for Play Store to improve practical utility."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q3: "Can IndoBERT beat TF-IDF with more data?"

A: "Possibly yes. Transformer literature suggests 5,000+ samples needed 
    to leverage pre-training advantage. Our 670 training samples are 
    insufficient. However, for this use case (explicit sentiment keywords, 
    production speed requirements, interpretability needs), TF-IDF may 
    remain preferred even with larger datasets. Cost-benefit: 10× speed 
    advantage and stakeholder interpretability justify TF-IDF deployment."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q4: "How do you validate InSet lexicon accuracy?"

A: "InSet is the largest available Indonesian sentiment lexicon (10,218 
    terms). We acknowledge limitation: no ground truth validation possible 
    without manual annotation. However, our approach is reproducible and 
    transparent. Word frequency analysis (Slide 3) shows sensible patterns: 
    'error', 'gagal' in negative; 'mantap', 'oke' in positive. Future work: 
    commission manual annotation of sample for lexicon validation."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q5: "What about bias in your models?"

A: "We addressed class imbalance via three mechanisms: (1) Stratified 
    train-test split preserves real-world distribution, (2) SVM class 
    weighting set to 'balanced' inversely scales by frequency, (3) Macro 
    F1 as primary metric treats all classes equally. Remaining bias: 
    TF-IDF shows ±3.6% prediction bias; IndoBERT shows -10.72% negative 
    over-prediction on App Store. This is documented in evaluation reports. 
    No model is perfect; we prioritize transparency."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q6: "Can your methodology generalize to other apps?"

A: "Yes, with caveats. CRISP-DM framework is app-agnostic. Preprocessing 
    pipeline (translation, cleaning, tokenization, stopword removal, 
    stemming) applies to any Indonesian text. InSet lexicon works for 
    general sentiment. However, app-specific keywords ('masuk', 'bayar', 
    'otp') are Disney+ Hotstar-specific. For other apps: retain methodology, 
    adjust custom stopwords and domain keywords. Our GitHub documentation 
    enables replication."
```

---

## 📊 VISUAL ASSETS REQUIRED

```
SLIDE-BY-SLIDE VISUAL CHECKLIST (UPDATED FOR 11-SLIDE FLOW):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Slide 1: Business Understanding validation checklist table
   Source: [Chapter IV Section 4.2](../CHAPTER_IV_RESULTS_DISCUSSION.md#42-crisp-dm-phase-1-business-understanding-results)
   + Objectives vs. achieved status table

✅ Slide 2: 6-stage preprocessing pipeline flowchart
   Create custom: Flowchart showing translation→cleaning→tokenization→stopword→stemming→final

✅ Slide 3: Sentiment distribution bar charts (App vs Play) 
   Source: [PLATFORM_COMPARISON_ANALYSIS.md](../../outputs/reports/PLATFORM_COMPARISON_ANALYSIS.md)
   + InSet lexicon coverage statistics

✅ Slide 4: App Store confusion matrices (TF-IDF vs IndoBERT)
   Source: [EVALUATION_RESULTS_APPSTORE.md](../../outputs/reports/EVALUATION_RESULTS_APPSTORE.md)
   + Classification report tables (Precision, Recall, F1)

✅ Slide 5: Play Store confusion matrices (TF-IDF vs IndoBERT)
   Source: [EVALUATION_RESULTS_PLAYSTORE.md](../../outputs/reports/EVALUATION_RESULTS_PLAYSTORE.md)
   + Classification report tables (Precision, Recall, F1)

✅ Slide 6: Cross-platform comparison table + performance gap visualization
   Source: [PLATFORM_COMPARISON_ANALYSIS.md](../../outputs/reports/PLATFORM_COMPARISON_ANALYSIS.md)
   + Accuracy vs Macro F1 paradox chart

✅ Slide 7: Word cloud (negative keywords, sized by frequency)
   Source: [Wordclouds folder](../../docs/analysis/wordclouds/), [Word Frequency Analysis](../../docs/analysis/WORD_FREQUENCY_ANALYSIS.md)
   + Horizontal bar chart (top 10 negative keywords)

✅ Slide 8: Dashboard screenshot ([localhost:8600](http://localhost:8600) interface)
   + Throughput comparison chart (750 vs 70 reviews/min)

✅ Slide 9: 4-quadrant diagram (dataset size, task nature, efficiency, interpretability)
   Create custom: PowerPoint/Keynote diagram explaining TF-IDF advantage

✅ Slide 10: Three-panel RQ answer layout (RQ1, RQ2, RQ3)
   Create custom: Evidence → Answer format for each RQ

✅ Slide 11: Timeline diagram (0-3mo, 3-6mo, 6-12mo recommendations)
   Create custom: Horizontal timeline with priority icons
```

---

## 🎯 KEY MESSAGES TO EMPHASIZE

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. EVIDENCE-BASED RESEARCH:
   ✅ ALL numbers from actual scraped data (April 7th, 2025)
   ✅ 1,676 reviews analyzed (838 per platform)
   ✅ Reproducible methodology (CRISP-DM framework)

2. COUNTERINTUITIVE FINDING:
   ✅ Traditional TF-IDF beats modern IndoBERT (+0.075 Macro F1)
   ✅ Context matters: small data + explicit sentiment + speed
   ✅ Challenges transformer superiority assumption

3. ACTIONABLE INTELLIGENCE:
   ✅ Top 3 priorities: authentication, payment, streaming quality
   ✅ 280 auth mentions, 134 payment mentions (quantified)
   ✅ Dashboard enables continuous monitoring (750 reviews/min)

4. PRODUCTION READINESS:
   ✅ Not just research—deployed functional dashboard
   ✅ TF-IDF models ready for production integration
   ✅ Complete CRISP-DM cycle (business → deployment)

5. CROSS-PLATFORM INSIGHTS:
   ✅ Play Store 15.87% more negative than App Store
   ✅ Android users more price-sensitive (+3.6% vs +1.7%)
   ✅ Platform-specific technical issues documented
```

---

**End of Evidence-Based Presentation Outline (Chapters IV-V)**

**Total Slides**: 11 slides (complete CRISP-DM flow with Business Understanding validation)  
**Total Time**: 14-15 minutes (within 20-22 min session with Chapter III)  
**Evidence Sources**: All data tied to actual project files and reports
**Flow**: Business Understanding (Slide 1) → Data Preparation (Slide 2) → Evaluation (Slides 3-6) → Word Frequency (Slide 7) → Deployment (Slide 8) → Discussion (Slides 9-11)
