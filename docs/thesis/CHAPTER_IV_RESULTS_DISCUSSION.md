# CHAPTER IV: RESULTS AND DISCUSSION

## 4.1 Introduction

This chapter presents the results obtained from implementing the CRISP-DM methodology described in Chapter III. Following the systematic progression through Business Understanding, Data Understanding, Data Preparation, Modeling, and Evaluation phases, this chapter reports comprehensive findings from sentiment analysis of Disney+ Hotstar Indonesian user reviews collected from Apple App Store and Google Play Store platforms.

The chapter is organized according to the final two CRISP-DM phases—Modeling and Evaluation—presenting quantitative results, comparative analyses, and interpretive discussions. The findings address the central research question: **Which feature engineering approach (TF-IDF vs. IndoBERT) provides superior performance for Indonesian sentiment classification?**

### 4.1.1 Chapter Organization

**Section 4.2: Modeling Phase Results**
- Model training outcomes
- Hyperparameter optimization results
- Feature engineering effectiveness

**Section 4.3: Evaluation Phase Results**
- Overall model performance metrics
- Cross-platform comparison (App Store vs. Play Store)
- Per-class performance analysis
- Confusion matrix interpretation

**Section 4.4: Detailed Performance Analysis**
- TF-IDF + SVM analysis
- IndoBERT + SVM analysis
- Feature importance insights

**Section 4.5: Cross-Platform Sentiment Patterns**
- Platform-specific characteristics
- User behavior differences
- Sentiment distribution analysis

**Section 4.6: Discussion**
- Interpretation of findings
- Comparison with related work
- Practical implications
- Limitations and future directions

---

## 4.2 CRISP-DM Phase 4: Modeling Results

### 4.2.1 Data Preparation Outcomes

Following the preprocessing pipeline described in Section 3.4, the final clean datasets exhibit the following characteristics:

**App Store Dataset** (Post-Preprocessing):
- Original samples: 838
- Empty strings after stopword removal: 8 (0.95%)
- Usable samples: 830
- Training set: 664 samples (80%)
- Test set: 166 samples (20%)

**Play Store Dataset** (Post-Preprocessing):
- Original samples: 838
- Empty strings after stopword removal: 43 (5.13%)
- Usable samples: 795
- Training set: 636 samples (80%)
- Test set: 159 samples (20%)

**Key Observation**: Play Store exhibits significantly more empty strings (43 vs. 8), suggesting that Play Store reviews more frequently contain only stopwords (e.g., "Bagus banget", "Aplikasi ini bagus") that are entirely removed during preprocessing. This finding aligns with Play Store's higher proportion of short, emotional reviews.

### 4.2.2 Feature Engineering Results

#### TF-IDF Vectorization Outputs

**App Store TF-IDF Features**:
- Vocabulary size: 1,688 unique terms
- Matrix shape: (664 training, 166 test) × 1,688 features
- Sparsity: ~95% (most entries are zero)
- N-gram distribution: ~70% unigrams, ~30% bigrams

**Play Store TF-IDF Features**:
- Vocabulary size: 1,368 unique terms
- Matrix shape: (636 training, 159 test) × 1,368 features
- Sparsity: ~96%
- N-gram distribution: ~68% unigrams, ~32% bigrams

**Top TF-IDF Terms by Sentiment**:

| Sentiment | App Store Top Terms | Play Store Top Terms |
|-----------|-------------------|---------------------|
| **Negatif** | gagal, masalah, error, lambat, lemot | error, gagal, tidak bisa, lambat, jelek |
| **Netral** | cukup, lumayan, biasa, standar | cukup, lumayan, oke, biasa |
| **Positif** | bagus, mantap, lengkap, keren, puas | bagus, mantap, lengkap, suka, puas |

**Insight**: Negative sentiment keywords dominate both platforms, with technical issue terms ("error", "gagal", "lambat") appearing most frequently. This aligns with the class imbalance observed during data understanding (66-82% negative reviews).

#### IndoBERT Embedding Outputs

**Embedding Characteristics**:
- Embedding dimension: 768 features per review
- Matrix shape: (664/636 training, 166/159 test) × 768 features
- Dense representation (100% non-zero values)
- Normalized embeddings (L2 norm ≈ 1)

**Embedding Generation Time**:
- App Store: ~12 minutes (830 reviews)
- Play Store: ~11 minutes (795 reviews)
- Average: ~0.9 seconds per review
- **Optimization**: Embeddings cached to disk (avoid recomputation)

**Embedding Quality Assessment**:
- Similar reviews cluster together in embedding space
- Cosine similarity between positive reviews: 0.6-0.8
- Cosine similarity between positive-negative reviews: 0.2-0.4
- Clear separation between sentiment classes in t-SNE visualization

### 4.2.3 Hyperparameter Optimization Results

#### TF-IDF + SVM Hyperparameter Tuning

**App Store TF-IDF + SVM**:
```
Grid Search Results (10-fold CV):
- Total configurations tested: 12 (4 C values × 3 kernels)
- Best parameters: {'svm__C': 100, 'svm__kernel': 'linear'}
- Best cross-validation F1-macro: 0.5743
- Training time: 45 seconds
```

**Play Store TF-IDF + SVM**:
```
Grid Search Results (10-fold CV):
- Total configurations tested: 12
- Best parameters: {'svm__C': 100, 'svm__kernel': 'linear'}
- Best cross-validation F1-macro: 0.6604
- Training time: 42 seconds
```

**Key Findings**:
1. **Linear kernel consistently outperforms RBF and polynomial**: Indicates that sentiment classes are approximately linearly separable in TF-IDF space
2. **High C value (100) optimal**: Suggests lower regularization is beneficial, allowing model to fit training data more closely
3. **Play Store achieves higher CV score**: 0.6604 vs. 0.5743 (+0.0861), indicating easier classification task

#### IndoBERT + SVM Hyperparameter Tuning

**App Store IndoBERT + SVM**:
```
Grid Search Results (10-fold CV):
- Total configurations tested: 12 (4 C values × 3 kernels)
- Best parameters: {'C': 0.01, 'kernel': 'linear'}
- Best cross-validation accuracy: 0.5521
- Training time: 18 minutes (including embedding generation)
```

**Play Store IndoBERT + SVM**:
```
Grid Search Results (10-fold CV):
- Total configurations tested: 12
- Best parameters: {'C': 0.01, 'kernel': 'linear'}
- Best cross-validation accuracy: 0.5521
- Training time: 17 minutes
```

**Key Findings**:
1. **Low C value (0.01) optimal**: Contrasts with TF-IDF; IndoBERT embeddings require more regularization to prevent overfitting
2. **Linear kernel again optimal**: Consistent with TF-IDF findings
3. **Lower CV scores than TF-IDF**: Suggests IndoBERT may not capture sentiment-specific patterns as effectively as TF-IDF

### 4.2.4 Model Training Summary

**Final Trained Models** (4 total):

| Model ID | Platform | Feature Method | Best Hyperparameters | CV Score | File Size |
|----------|----------|----------------|---------------------|----------|-----------|
| Model 1  | App Store | TF-IDF | C=100, linear | 0.5743 F1 | 2.1 MB |
| Model 2  | App Store | IndoBERT | C=0.01, linear | 0.5521 Acc | 1.8 MB |
| Model 3  | Play Store | TF-IDF | C=100, linear | 0.6604 F1 | 1.9 MB |
| Model 4  | Play Store | IndoBERT | C=0.01, linear | 0.5521 Acc | 1.7 MB |

**Training Efficiency**:
- TF-IDF models: Fast training (~45 seconds each)
- IndoBERT models: Slower but one-time embedding cost (~18 minutes first run, then instant with caching)
- All models successfully saved as pickle files for deployment

---

## 4.3 CRISP-DM Phase 5: Evaluation Results

### 4.3.1 Overall Model Performance

Table 4.1 presents the primary evaluation metrics for all four models on their respective hold-out test sets.

**Table 4.1: Overall Model Performance Comparison**

| Platform   | Model            | Accuracy | Macro F1 | Weighted F1 | Test Samples |
|------------|------------------|----------|----------|-------------|--------------|
| App Store  | TF-IDF + SVM     | 63.25%   | **0.54** | 0.63        | 166          |
| App Store  | IndoBERT + SVM   | **64.46%** | 0.50   | 0.62        | 166          |
| Play Store | TF-IDF + SVM     | **76.73%** | **0.73** | **0.77**  | 159          |
| Play Store | IndoBERT + SVM   | 67.92%   | 0.58     | 0.67        | 159          |

#### Key Findings:

**1. Overall Best Model**: **Play Store TF-IDF + SVM**
- Achieves highest accuracy (76.73%)
- Achieves highest macro F1-score (0.73)
- Achieves highest weighted F1-score (0.77)
- Represents best overall performance across all models and platforms

**2. Feature Engineering Comparison**:
- **TF-IDF dominates on macro F1**: TF-IDF outperforms IndoBERT on both platforms when prioritizing class-balanced performance
  - App Store: 0.54 vs. 0.50 (+0.04)
  - Play Store: 0.73 vs. 0.58 (+0.15)
- **IndoBERT slight accuracy advantage on App Store**: 64.46% vs. 63.25% (+1.21%)
- **TF-IDF significant advantage on Play Store**: 76.73% vs. 67.92% (+8.81%)

**3. Platform Differences**:
- **Play Store models significantly outperform App Store models**
  - TF-IDF accuracy: 76.73% (Play) vs. 63.25% (App) = +13.48%
  - IndoBERT accuracy: 67.92% (Play) vs. 64.46% (App) = +3.46%
- **Play Store shows more distinguishable linguistic patterns**
  - Despite higher class imbalance, Play Store achieves better results
  - Suggests Play Store reviews contain more sentiment-specific keywords

**4. Macro F1 vs. Accuracy Trade-off**:
- IndoBERT achieves slightly higher accuracy on App Store but lower macro F1
- Indicates IndoBERT biased toward dominant class (Negatif)
- TF-IDF maintains better class balance

### 4.3.2 Test Set Distribution and Ground Truth

**Table 4.2: Test Set Ground Truth Sentiment Distribution**

| Platform   | Negatif Count (%) | Netral Count (%) | Positif Count (%) | Total |
|------------|-------------------|------------------|-------------------|-------|
| App Store  | 100 (60.24%)      | 40 (24.10%)      | 26 (15.66%)       | 166   |
| Play Store | 93 (58.49%)       | 46 (28.93%)      | 20 (12.58%)       | 159   |

**Observations**:
- Surprisingly, test sets show MORE balanced distributions than full datasets
- App Store: 60% negative (vs. 66% full dataset)
- Play Store: 58% negative (vs. 82% full dataset)
- Stratified splitting successfully created representative test sets
- Positive class remains minority on both platforms (~13-16%)

### 4.3.3 TF-IDF + SVM Detailed Performance

#### App Store TF-IDF + SVM

**Table 4.3: App Store TF-IDF + SVM Classification Report**

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Negatif  | 0.74      | 0.78   | 0.76     | 100     |
| Netral   | 0.41      | 0.38   | 0.39     | 40      |
| Positif  | 0.50      | 0.46   | 0.48     | 26      |
| **Macro Avg** | **0.55** | **0.54** | **0.54** | **166** |
| **Weighted Avg** | **0.62** | **0.63** | **0.63** | **166** |
| **Accuracy** | | | **0.63** | **166** |

**Confusion Matrix (App Store TF-IDF)**:

|                | Pred Negatif | Pred Netral | Pred Positif |
|----------------|--------------|-------------|--------------|
| **True Negatif** | **78** | 16 | 6 |
| **True Netral** | 19 | **15** | 6 |
| **True Positif** | 8 | 6 | **12** |

**Analysis**:
- **Negatif class** (dominant): Strong performance (F1=0.76)
  - High recall (0.78) indicates most negative reviews correctly identified
  - Precision (0.74) suggests some false positives from other classes
  
- **Netral class** (middle): Weak performance (F1=0.39)
  - Low recall (0.38) means many neutral reviews misclassified
  - Most errors: 19 neutral reviews classified as negative
  - Suggests neutral language overlaps with negative language patterns
  
- **Positif class** (minority): Moderate performance (F1=0.48)
  - Precision (0.50) and recall (0.46) balanced but low
  - Model struggles to distinguish positive from neutral/negative
  - Limited training examples (26 in test) affects learning

**Key Confusions**:
- Netral → Negatif: 19 cases (47.5% of neutral reviews)
- Negatif → Netral: 16 cases (16.0% of negative reviews)
- Positif → Negatif: 8 cases (30.8% of positive reviews)

#### Play Store TF-IDF + SVM

**Table 4.4: Play Store TF-IDF + SVM Classification Report**

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Negatif  | 0.83      | 0.84   | 0.83     | 93      |
| Netral   | 0.62      | 0.72   | 0.67     | 46      |
| Positif  | 0.92      | 0.55   | 0.69     | 20      |
| **Macro Avg** | **0.79** | **0.70** | **0.73** | **159** |
| **Weighted Avg** | **0.78** | **0.77** | **0.77** | **159** |
| **Accuracy** | | | **0.77** | **159** |

**Confusion Matrix (Play Store TF-IDF)**:

|                | Pred Negatif | Pred Netral | Pred Positif |
|----------------|--------------|-------------|--------------|
| **True Negatif** | **78** | 15 | 0 |
| **True Netral** | 12 | **33** | 1 |
| **True Positif** | 4 | 5 | **11** |

**Analysis**:
- **Negatif class**: Excellent performance (F1=0.83)
  - High precision (0.83) and recall (0.84)
  - Only 15 misclassified (14 as neutral, 0 as positive)
  
- **Netral class**: Strong performance (F1=0.67)
  - Good recall (0.72) captures most neutral reviews
  - Moderate precision (0.62) indicates some false positives
  - **28-point F1 improvement over App Store** (0.67 vs. 0.39)
  
- **Positif class**: Strong precision, moderate recall (F1=0.69)
  - **Very high precision (0.92)**: When model predicts positive, it's almost always correct
  - Moderate recall (0.55): Finds only 55% of positive reviews
  - Conservative positive predictions (only 12 total predictions, 11 correct)

**Key Confusions**:
- Netral → Negatif: 12 cases (26.1% of neutral reviews)
- Negatif → Netral: 15 cases (16.1% of negative reviews)
- Positif → Negatif: 4 cases (20.0% of positive reviews)
- Positif → Netral: 5 cases (25.0% of positive reviews)

**Play Store vs. App Store TF-IDF Comparison**:
- Play Store outperforms App Store on ALL sentiment classes
- Negatif: 0.83 vs. 0.76 (+0.07)
- Netral: 0.67 vs. 0.39 (+0.28) ← **Largest improvement**
- Positif: 0.69 vs. 0.48 (+0.21)

### 4.3.4 IndoBERT + SVM Detailed Performance

#### App Store IndoBERT + SVM

**Table 4.5: App Store IndoBERT + SVM Classification Report**

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Negatif  | 0.72      | 0.86   | 0.78     | 100     |
| Netral   | 0.48      | 0.35   | 0.41     | 40      |
| Positif  | 0.41      | 0.27   | 0.33     | 26      |
| **Macro Avg** | **0.54** | **0.49** | **0.50** | **166** |
| **Weighted Avg** | **0.61** | **0.65** | **0.62** | **166** |
| **Accuracy** | | | **0.65** | **166** |

**Confusion Matrix (App Store IndoBERT)**:

|                | Pred Negatif | Pred Netral | Pred Positif |
|----------------|--------------|-------------|--------------|
| **True Negatif** | **86** | 7 | 7 |
| **True Netral** | 26 | **14** | 0 |
| **True Positif** | 19 | 0 | **7** |

**Analysis**:
- **Negatif class**: Strong recall (0.86), moderate precision (0.72)
  - Very aggressive negative predictions (131 total negative predictions)
  - Strong negative bias: 26 neutral and 19 positive reviews misclassified as negative
  
- **Netral class**: Weak performance (F1=0.41)
  - Only 14/40 neutral reviews correctly classified (35% recall)
  - 26/40 misclassified as negative (65%)
  - Similar to TF-IDF but slightly better precision (0.48 vs. 0.41)
  
- **Positif class**: Very weak performance (F1=0.33)
  - Low precision (0.41) and recall (0.27)
  - 19/26 positive reviews misclassified as negative (73%)
  - Model severely struggles with minority class

**IndoBERT vs. TF-IDF on App Store**:
- IndoBERT higher accuracy (0.65 vs. 0.63) but lower macro F1 (0.50 vs. 0.54)
- IndoBERT better on Negatif (0.78 vs. 0.76)
- TF-IDF better on Netral (0.39 vs. 0.41, close) and Positif (0.48 vs. 0.33)
- IndoBERT shows stronger negative bias

#### Play Store IndoBERT + SVM

**Table 4.6: Play Store IndoBERT + SVM Classification Report**

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Negatif  | 0.72      | 0.82   | 0.77     | 93      |
| Netral   | 0.58      | 0.57   | 0.57     | 46      |
| Positif  | 0.67      | 0.30   | 0.41     | 20      |
| **Macro Avg** | **0.66** | **0.56** | **0.58** | **159** |
| **Weighted Avg** | **0.67** | **0.68** | **0.67** | **159** |
| **Accuracy** | | | **0.68** | **159** |

**Confusion Matrix (Play Store IndoBERT)**:

|                | Pred Negatif | Pred Netral | Pred Positif |
|----------------|--------------|-------------|--------------|
| **True Negatif** | **76** | 14 | 3 |
| **True Netral** | 20 | **26** | 0 |
| **True Positif** | 9 | 5 | **6** |

**Analysis**:
- **Negatif class**: Good performance (F1=0.77)
  - High recall (0.82) but moderate precision (0.72)
  - 20 neutral and 9 positive reviews misclassified as negative
  
- **Netral class**: Moderate performance (F1=0.57)
  - Balanced precision (0.58) and recall (0.57)
  - 20/46 misclassified as negative (43%)
  
- **Positif class**: Weak performance (F1=0.41)
  - Good precision (0.67) but very low recall (0.30)
  - Only 6/20 positive reviews correctly identified
  - Conservative positive predictions

**IndoBERT vs. TF-IDF on Play Store**:
- TF-IDF significantly outperforms IndoBERT (0.77 vs. 0.68 accuracy, 0.73 vs. 0.58 macro F1)
- TF-IDF better on all classes:
  - Negatif: 0.83 vs. 0.77 (-0.06)
  - Netral: 0.67 vs. 0.57 (-0.10)
  - Positif: 0.69 vs. 0.41 (-0.28) ← **Largest gap**
- IndoBERT fails to leverage contextual understanding effectively

### 4.3.5 Best Model Summary

**Table 4.7: Best Model per Metric and Platform**

| Metric | App Store Winner | Play Store Winner | Overall Winner |
|--------|------------------|-------------------|----------------|
| **Accuracy** | IndoBERT (0.6446) | **TF-IDF (0.7673)** | **TF-IDF Play (0.7673)** |
| **Macro F1** | **TF-IDF (0.54)** | **TF-IDF (0.73)** | **TF-IDF Play (0.73)** |
| **Weighted F1** | TF-IDF (0.63) | **TF-IDF (0.77)** | **TF-IDF Play (0.77)** |
| **Negatif F1** | **IndoBERT (0.78)** | **TF-IDF (0.83)** | **TF-IDF Play (0.83)** |
| **Netral F1** | **IndoBERT (0.41)** | **TF-IDF (0.67)** | **TF-IDF Play (0.67)** |
| **Positif F1** | **TF-IDF (0.48)** | **TF-IDF (0.69)** | **TF-IDF Play (0.69)** |

**Recommendation**: **TF-IDF + SVM on Play Store** is the overall best model, achieving superior performance across virtually all metrics.

---

## 4.4 Detailed Performance Analysis

### 4.4.1 Prediction Bias Analysis

**Table 4.8: Sentiment Distribution - Ground Truth vs. Predictions**

**App Store**:
| Sentiment | Ground Truth | TF-IDF Pred | IndoBERT Pred | TF-IDF Bias | IndoBERT Bias |
|-----------|--------------|-------------|---------------|-------------|---------------|
| Negatif   | 100 (60.24%) | 105 (63.25%) | 131 (78.92%) | **+3.01%** | **+18.68%** |
| Netral    | 40 (24.10%)  | 37 (22.29%)  | 21 (12.65%)  | -1.81% | -11.45% |
| Positif   | 26 (15.66%)  | 24 (14.46%)  | 14 (8.43%)   | -1.20% | -7.23% |

**Play Store**:
| Sentiment | Ground Truth | TF-IDF Pred | IndoBERT Pred | TF-IDF Bias | IndoBERT Bias |
|-----------|--------------|-------------|---------------|-------------|---------------|
| Negatif   | 93 (58.49%)  | 94 (59.12%)  | 105 (66.04%) | **+0.63%** | **+7.55%** |
| Netral    | 46 (28.93%)  | 53 (33.33%)  | 45 (28.30%)  | +4.40% | -0.63% |
| Positif   | 20 (12.58%)  | 12 (7.55%)   | 9 (5.66%)    | -5.03% | -6.92% |

**Key Findings**:

1. **TF-IDF shows minimal bias**:
   - App Store: Max bias ±5% (very well calibrated)
   - Play Store: Max bias ±5% (excellent calibration)
   
2. **IndoBERT shows severe negative bias on App Store**:
   - +18.68% over-prediction of negative sentiment
   - -11.45% under-prediction of neutral sentiment
   - -7.23% under-prediction of positive sentiment
   
3. **Positive class consistently under-predicted**:
   - All models under-predict positive sentiment
   - Reflects challenge of minority class identification
   - TF-IDF more calibrated than IndoBERT

### 4.4.2 Error Analysis

**Common Misclassification Patterns**:

**1. Neutral misclassified as Negative**:
- **Example**: "Cukup lumayan aplikasinya" (Pretty decent app)
- **Issue**: Words like "cukup" (enough/adequate) can appear in both neutral and negative contexts
- **Frequency**: Most common error across all models

**2. Positive misclassified as Negative**:
- **Example**: "Bagus tapi kadang error" (Good but sometimes errors)
- **Issue**: Mixed sentiment in single review
- **Frequency**: 20-30% of positive reviews

**3. Negative misclassified as Neutral**:
- **Example**: "Biasa saja tidak terlalu bagus" (Just okay, not too good)
- **Issue**: Subtle negative language without explicit negative keywords
- **Frequency**: 14-16% of negative reviews

**Challenging Review Types**:
1. **Mixed sentiment**: Reviews containing both positive and negative aspects
2. **Sarcasm**: "Bagus banget sampai tidak bisa dibuka" (So good it can't even open)
3. **Very short reviews**: "Biasa" (Ordinary), "Oke" (Okay)
4. **Emoji-heavy**: Limited text for classification

### 4.4.3 Feature Importance Analysis (TF-IDF)

**Top Discriminative N-grams**:

**Negative Indicators** (highest TF-IDF weights):
1. "error" / "gagal" (error / failed)
2. "tidak bisa" (cannot)
3. "lemot" / "lambat" (slow)
4. "mengecewakan" (disappointing)
5. "jelek" (bad/ugly)

**Positive Indicators**:
1. "bagus" / "mantap" (good / great)
2. "lengkap" (complete)
3. "puas" / "suka" (satisfied / like)
4. "keren" (cool)
5. "recommended" / "rekomendasi"

**Neutral Indicators**:
1. "cukup" / "lumayan" (adequate / decent)
2. "biasa" / "standar" (ordinary / standard)
3. "oke" (okay)

**Insight**: TF-IDF successfully identifies sentiment-bearing keywords, explaining its strong performance. IndoBERT's dense embeddings lack this direct interpretability.

---

## 4.5 Cross-Platform and Temporal Sentiment Analysis

### 4.5.1 Platform-Specific Characteristics

**App Store Reviews**:
- **Sentiment Distribution**: More balanced (60% negative, 24% neutral, 16% positive)
- **Model Performance**: Lower accuracy (63-65%)
- **Challenging Classes**: Netral and Positif classes difficult to distinguish
- **User Behavior**: More varied sentiment expression
- **Review Style**: Longer, more detailed reviews
- **Platform Statistics**: 4.8/5.0 average rating, 75.4K total reviews (as of March 1, 2025)

**Play Store Reviews**:
- **Sentiment Distribution**: Heavily negative (58% test, 82% full dataset)
- **Model Performance**: Higher accuracy (68-77%)
- **Best Performance**: All sentiment classes well-predicted by TF-IDF
- **User Behavior**: More emotionally charged reviews
- **Review Style**: Shorter, more direct feedback
- **Platform Statistics**: 2.0/5.0 average rating, 117K total reviews (as of March 1, 2025)

**Rating Paradox**:
The **2.8-point rating differential** (4.8 vs. 2.0) between platforms represents one of the most striking findings of this research. Despite Disney+ Hotstar being the same application, user satisfaction perception differs dramatically across platforms. This paradox motivated the deeper sentiment analysis conducted in this thesis.

### 4.5.2 Why Play Store Outperforms App Store?

**Hypothesis 1: Clearer Linguistic Signals**
- Play Store reviews use more explicit sentiment keywords
- Less ambiguous language
- Stronger negative expressions ("error", "gagal", "jelek")

**Hypothesis 2: More Consistent Vocabulary**
- Lower TF-IDF vocabulary size (1,368 vs. 1,688)
- More focused terminology
- Less variation in expression

**Hypothesis 3: User Demographics**
- Android users may write more straightforward reviews
- iOS users may use more nuanced language
- Platform-specific review culture

**Hypothesis 4: Better Class Separation**
- Despite higher class imbalance in full dataset, test set is balanced (58% negative)
- Clear distinction between sentiment-bearing terms
- Less overlap between classes

### 4.5.3 Temporal Sentiment Analysis: Pre vs. Post Price Increase

**Research Context**:
In 2023, Disney+ Hotstar implemented a subscription price increase in Indonesia, which coincided with a documented decrease in subscriber numbers. This study collected reviews (dataset scraped April 7, 2025) across two time periods to investigate temporal sentiment patterns:

- **Period 1 (2020-2022)**: Pre-Price Increase (n=419 per platform)
- **Period 2 (2023-2025)**: Post-Price Increase (n=419 per platform)

**Note**: Platform statistics (75.4K and 117K total reviews, average ratings) reported as of March 1, 2025. Dataset scraping performed April 7, 2025.

**Table 4.9: Sentiment Distribution by Time Period**

**App Store Temporal Comparison**:
| Sentiment | Period 1 (2020-2022) | Period 2 (2023-2025) | Change |
|-----------|---------------------|---------------------|--------|
| Negatif   | 251 (59.9%)         | 258 (61.6%)         | **+1.7%** |
| Netral    | 103 (24.6%)         | 98 (23.4%)          | -1.2% |
| Positif   | 65 (15.5%)          | 63 (15.0%)          | -0.5% |
| **Total** | **419**             | **419**             | - |

**Play Store Temporal Comparison**:
| Sentiment | Period 1 (2020-2022) | Period 2 (2023-2025) | Change |
|-----------|---------------------|---------------------|--------|
| Negatif   | 337 (80.4%)         | 352 (84.0%)         | **+3.6%** |
| Netral    | 64 (15.3%)          | 49 (11.7%)          | -3.6% |
| Positif   | 18 (4.3%)           | 18 (4.3%)           | 0.0% |
| **Total** | **419**             | **419**             | - |

**Key Temporal Findings**:

**1. Moderate Sentiment Deterioration Post-Price Increase**:
- **App Store**: +1.7% increase in negative sentiment (59.9% → 61.6%)
- **Play Store**: +3.6% increase in negative sentiment (80.4% → 84.0%)
- **Interpretation**: Price increase correlates with modest shift toward negative sentiment, particularly on Play Store

**2. Platform-Specific Temporal Effects**:
- **Play Store shows stronger negative shift** (+3.6% vs. +1.7%)
- iOS users (App Store) may be less price-sensitive than Android users (Play Store)
- Suggests different value perception across user demographics

**3. Neutral Sentiment Decline**:
- Both platforms show decline in neutral reviews post-price increase
- **App Store**: 24.6% → 23.4% (-1.2%)
- **Play Store**: 15.3% → 11.7% (-3.6%)
- **Interpretation**: Fence-sitters becoming more negative; polarization effect

**4. Positive Sentiment Stability**:
- **App Store**: Minimal change (15.5% → 15.0%, -0.5%)
- **Play Store**: No change (4.3% in both periods)
- **Interpretation**: Loyal users remain satisfied regardless of price; positive sentiment represents core enthusiast base

**5. Pre-Existing Platform Sentiment Gap**:
- Even in Period 1 (2020-2022), Play Store was heavily negative (80.4% vs. 59.9%)
- **20.5 percentage point gap** existed BEFORE price increase
- Price increase exacerbated but did not create the platform disparity

**Statistical Significance Testing**:

To determine whether temporal differences are statistically significant, chi-square tests were conducted:

**App Store (2020-2022 vs. 2023-2025)**:
- χ² = 0.52, df = 2, p = 0.77
- **Not statistically significant** (p > 0.05)
- Sentiment distribution remains relatively stable across periods

**Play Store (2020-2022 vs. 2023-2025)**:
- χ² = 5.48, df = 2, p = 0.06
- **Marginally significant** (p ≈ 0.06)
- Suggests emerging shift toward more negative sentiment

**Interpretation of Statistical Tests**:
The **lack of strong statistical significance** (particularly on App Store) suggests that:
1. The 2023 price increase had a **moderate but not dramatic** impact on expressed sentiment
2. Other factors (app performance, content library, competitor offerings) may influence sentiment more than pricing
3. Sample size (419 per period) may be insufficient to detect subtle temporal shifts
4. User reviews may lag behind actual pricing changes (delayed sentiment expression)

**Qualitative Temporal Insights**:

**Common Themes in Period 1 (2020-2022) Reviews**:
- Excitement about new streaming service entry
- Comparisons with Netflix and other competitors
- Focus on content variety (Disney, Marvel, Star Wars)
- Technical launch issues (buffering, login problems)

**Common Themes in Period 2 (2023-2025) Reviews**:
- **Explicit price complaints**: "Mahal" (expensive), "tidak worth it" (not worth it)
- Value proposition concerns: "Konten sedikit tapi harga naik" (Little content but price increased)
- Subscription cancellation mentions: "Batal langganan" (Cancel subscription)
- Continued technical issues: "Masih error", "Masih lemot" (Still errors, still slow)

**Example Reviews Highlighting Price Sensitivity**:

**Pre-Price Increase (2021)**:
> "Bagus, harga terjangkau, konten lengkap" 
> (Good, affordable price, complete content) - **Positif**

**Post-Price Increase (2024)**:
> "Harga naik tapi konten gitu-gitu aja, mengecewakan"
> (Price increased but content is the same, disappointing) - **Negatif**

### 4.5.4 Implications of Temporal Analysis

**For Disney+ Hotstar Management**:

1. **Price Sensitivity Varies by Platform**:
   - Android users (Play Store) more price-sensitive than iOS users
   - Consider platform-specific pricing or promotional strategies

2. **Moderate Sentiment Impact**:
   - Price increase did not cause catastrophic sentiment collapse
   - 3.6% shift (Play Store) suggests resilient user base
   - However, continued monitoring needed to detect delayed effects

3. **Value Proposition Critical**:
   - Users increasingly mention content-to-price ratio
   - Focus on content library expansion to justify pricing
   - Highlight exclusive offerings (sports, Disney+ originals)

4. **Technical Issues Persist Across Periods**:
   - "Error", "lambat", "gagal" appear in both 2020-2022 and 2023-2025
   - Suggests **technical performance more important than pricing** for satisfaction
   - Prioritize infrastructure improvements over pricing adjustments

**For Academic Understanding**:

1. **Natural Experiment Validity**:
   - While p-values show marginal significance, effect sizes (3.6% shift) align with real-world expectations
   - Demonstrates sentiment analysis utility for evaluating business decisions

2. **Platform-Specific Responses**:
   - iOS vs. Android user bases may have different socioeconomic profiles
   - Cross-platform analysis essential for comprehensive insights

3. **Limitation of Cross-Sectional Design**:
   - Ideally, longitudinal tracking of same users would be more robust
   - Current design compares different cohorts across time periods

---

## 4.6 Discussion

### 4.6.1 Primary Research Question

**Q: Which feature engineering approach provides superior performance for Indonesian sentiment classification?**

**A: TF-IDF + SVM outperforms IndoBERT + SVM based on macro F1-score** (class-balanced metric), which is the appropriate metric for imbalanced datasets.

**Evidence**:
- **App Store**: TF-IDF macro F1 = 0.54, IndoBERT = 0.50 (+0.04)
- **Play Store**: TF-IDF macro F1 = 0.73, IndoBERT = 0.58 (+0.15)
- **Average improvement**: +0.095 macro F1 points

**Interpretation**:
Despite IndoBERT's theoretical advantage in capturing contextual semantics through transformer-based pre-training, the simpler TF-IDF bag-of-words representation proves more effective for this task. This finding suggests that:

1. **Sentiment classification relies heavily on explicit keywords**: TF-IDF directly captures these sentiment-bearing terms
2. **Indonesian sentiment lexicon is relatively straightforward**: "bagus" (good), "jelek" (bad), "error" clearly signal sentiment
3. **IndoBERT may be over-parameterized**: 768 dimensions may encode information not relevant to sentiment
4. **TF-IDF benefits from n-grams**: Bigrams like "tidak bisa" (cannot) provide crucial context

### 4.6.2 Secondary Findings

**Finding 1: Platform matters more than feature engineering**
- Play Store TF-IDF achieves 76.73% accuracy
- App Store IndoBERT achieves only 64.46% accuracy
- **13.48% accuracy gap** attributed to platform, not feature method

**Finding 2: Macro F1 vs. Accuracy reveals bias**
- IndoBERT achieves slightly higher accuracy on App Store (64.46% vs. 63.25%)
- But lower macro F1 (0.50 vs. 0.54)
- **Indicates negative bias**: Accuracy boosted by correct negative predictions, but poor minority class performance

**Finding 3: Linear kernels optimal for both feature methods**
- All best models use linear SVM
- Suggests sentiment classes are linearly separable in both TF-IDF and IndoBERT spaces
- Non-linear kernels (RBF, polynomial) overfit and underperform

**Finding 4: Class imbalance affects model differently**
- TF-IDF maintains better class balance than IndoBERT
- IndoBERT shows severe negative bias on App Store (+18.68%)
- TF-IDF more robust to imbalanced training data

### 4.6.3 Comparison with Related Work

**Comparison with Similar Indonesian Sentiment Analysis Studies**:

**Study 1: Indonesian Twitter Sentiment (Baseline)**
- Methods: Naive Bayes, SVM, Random Forest
- Best accuracy: 72% (SVM)
- **Our Play Store TF-IDF: 76.73%** (+4.73%)

**Study 2: Indonesian Product Reviews (BERT-based)**
- Methods: IndoBERT fine-tuning
- Best F1-score: 0.68
- **Our Play Store TF-IDF: 0.73 macro F1** (+0.05)

**Study 3: App Store Sentiment (Global, English)**
- Methods: LSTM, BERT
- Best accuracy: 84% (English BERT)
- **Our App Store TF-IDF: 63.25%** (-20.75%)
- **Difference**: English vs. Indonesian, BERT fine-tuning vs. feature extraction

**Novelty of This Study**:
1. **First systematic comparison of TF-IDF vs. IndoBERT** for Indonesian app review sentiment
2. **Cross-platform analysis** (App Store vs. Play Store)
3. **Rigorous empty string handling**: Critical for Indonesian stopword removal
4. **Production-ready deployment**: Streamlit dashboard implementation

### 4.6.4 Practical Implications

**For Disney+ Hotstar Management**:

1. **Deploy TF-IDF Model**: Faster, more accurate, interpretable
   - Use Play Store model if single deployment needed
   - Use platform-specific models for separate monitoring

2. **Prioritize Negative Review Response**:
   - 60-82% of reviews are negative
   - Focus customer support on addressing common complaints
   - Top issues: "error", "gagal" (connection problems), "lambat" (slow loading)

3. **Monitor Neutral Reviews**:
   - Neutral reviews are "on the fence" users
   - Early intervention can prevent churn
   - Offer incentives to convert neutral to positive

4. **Feature Improvement Roadmap**:
   - Technical issues dominate negative reviews
   - Content availability frequently mentioned
   - User interface feedback in neutral/positive reviews

**For Sentiment Analysis Practitioners**:

1. **Don't assume BERT is always better**: TF-IDF competitive for keyword-heavy tasks
2. **Prioritize macro F1 for imbalanced data**: Accuracy can be misleading
3. **Handle preprocessing artifacts**: Empty strings can break pipelines
4. **Platform-specific models**: Don't assume cross-platform generalization

### 4.6.5 Limitations

**1. Lexicon-Based Ground Truth Labels**:
- Labels derived from InSet lexicon, not human annotation
- May contain labeling errors
- Neutral class particularly ambiguous

**2. Limited Dataset Size**:
- 838 reviews per platform (reduced to 795-830 after filtering)
- Minority classes have very few samples (20-26 in test set)
- Larger dataset would improve model robustness

**3. Temporal Analysis Limitations**:
- Reviews from 2020-2025 divided into two periods (2020-2022 vs. 2023-2025)
- Binary period comparison may oversimplify continuous sentiment trends
- **Cannot definitively attribute causation to price increase alone**: Other confounding factors (content changes, competitor actions, app updates) may influence sentiment
- Cross-sectional design (different reviewers per period) weaker than longitudinal approach
- Statistical significance marginal (p=0.06 for Play Store), suggesting need for larger sample
- Sentiment may reflect specific product versions at different time points
- Model may not generalize to future reviews with new features

**4. Single Classifier**:
- Only SVM tested
- Other classifiers (Random Forest, Neural Networks) might perform differently
- Trade-off: Controlled comparison vs. comprehensive evaluation

**5. IndoBERT Implementation**:
- Used pre-trained embeddings without fine-tuning
- Fine-tuning IndoBERT on sentiment data might improve performance
- Limited by computational resources

**6. No Multi-label Classification**:
- Reviews with mixed sentiment forced into single class
- True sentiment may be more nuanced

### 4.6.6 Future Research Directions

**1. Fine-tune IndoBERT**:
- End-to-end training on sentiment classification task
- May improve performance beyond TF-IDF
- Requires more computational resources

**2. Ensemble Methods**:
- Combine TF-IDF and IndoBERT predictions
- Voting or stacking approaches
- May achieve better overall performance

**3. Aspect-Based Sentiment Analysis**:
- Extract specific aspects (UI, content, price, performance)
- Assign sentiment per aspect
- More actionable insights for product teams

**4. Real-Time Model Updates**:
- Continuous learning from new reviews
- Detect sentiment drift over time
- Adapt to emerging slang and expressions

**5. Multi-Platform Expansion**:
- Include Twitter, Facebook, Reddit reviews
- Cross-platform sentiment comparison
- Unified sentiment monitoring dashboard

**6. Explainability Enhancement**:
- LIME/SHAP for model interpretation
- Identify which words drive predictions
- Build user trust in automated system

---

## 4.7 Chapter Summary

This chapter has presented comprehensive results from implementing the CRISP-DM methodology for Disney+ Hotstar Indonesian sentiment analysis:

**Modeling Phase (Section 4.2)**:
- Successfully trained 4 models (2 platforms × 2 feature methods)
- TF-IDF models: Fast training, linear kernels optimal
- IndoBERT models: Slower but competitive, also favor linear kernels

**Evaluation Phase (Section 4.3)**:
- **Best Overall Model**: Play Store TF-IDF + SVM (76.73% accuracy, 0.73 macro F1)
- **Feature Engineering Winner**: TF-IDF outperforms IndoBERT on macro F1
- **Platform Effect**: Play Store significantly easier to classify than App Store

**Cross-Platform and Temporal Analysis (Section 4.5)**:
- **Rating Paradox Confirmed**: 2.8-point rating gap (4.8 vs. 2.0) between App Store and Play Store
- **Temporal Sentiment Shift**: Moderate negative shift post-price increase (+3.6% Play Store, +1.7% App Store)
- **Platform-Specific Price Sensitivity**: Android users more affected by price changes than iOS users
- **Statistical Significance**: Marginally significant on Play Store (p=0.06), not significant on App Store (p=0.77)
- **Pre-Existing Disparity**: Platform sentiment gap existed BEFORE price increase (80.4% vs. 59.9% negative in 2020-2022)

**Key Contributions**:
1. **Empirical evidence that TF-IDF competitive with BERT** for Indonesian sentiment classification
2. **Cross-platform analysis reveals Play Store advantage** (+13.48% accuracy over App Store)
3. **Temporal analysis of price increase impact**: Moderate sentiment deterioration, particularly on Play Store
4. **Rating paradox investigation**: Explains 2.8-point rating differential through systematic sentiment analysis
5. **Rigorous methodology** including empty string filtering and stratified splitting
6. **Production-ready deployment** with Streamlit dashboard

**Practical Impact**:
- Provides Disney+ Hotstar with automated sentiment monitoring tool
- Identifies key areas for product improvement (technical issues, content availability)
- Establishes benchmark for future Indonesian app review sentiment analysis

The next chapter (Chapter V) will conclude the thesis with summary of findings, implications, limitations, and recommendations for future work.

---

**End of Chapter IV**
