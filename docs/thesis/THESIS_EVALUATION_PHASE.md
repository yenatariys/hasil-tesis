# Chapter: Evaluation Phase - Model Assessment and Cross-Platform Analysis

## 5.1 Introduction

This chapter presents the evaluation phase of the Disney+ Hotstar sentiment analysis study, following the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology. The evaluation phase systematically assesses the performance of trained models, analyzes sentiment distributions, examines the relationship between user ratings and sentiment scores, and conducts linguistic analysis through word cloud visualizations.

Building upon the modeling phase (Chapter 4), which developed TF-IDF + SVM and IndoBERT + SVM models for both App Store and Play Store platforms, this chapter evaluates their effectiveness in classifying Indonesian language reviews into three sentiment categories: Positif, Netral, and Negatif. The evaluation employs a comprehensive cross-platform comparative approach to identify platform-specific patterns, model strengths and weaknesses, and provide actionable recommendations for deployment.

### 5.1.1 Evaluation Objectives

The primary objectives of this evaluation phase are:

1. **Performance Assessment**: Quantify model accuracy, precision, recall, and F1-scores across all sentiment classes
2. **Cross-Platform Comparison**: Identify performance differences between App Store and Play Store reviews
3. **Distribution Analysis**: Compare initial lexicon-based labeling with machine learning predictions
4. **Rating Correlation**: Examine the relationship between numerical ratings and text-based sentiment scores
5. **Linguistic Insights**: Extract dominant keywords and themes for each sentiment category
6. **Model Selection**: Recommend the optimal model for deployment based on comprehensive evaluation metrics

### 5.1.2 Evaluation Methodology

The evaluation employs multiple complementary approaches:

- **Quantitative Metrics**: Confusion matrices, classification reports, accuracy, macro/weighted F1-scores
- **Comparative Analysis**: Side-by-side platform comparison to identify systematic differences
- **Correlation Analysis**: Pearson and Spearman correlations between ratings and sentiment scores
- **Visual Analytics**: Word clouds to reveal linguistic patterns and user concerns
- **Bias Analysis**: Assessment of prediction bias toward dominant classes

All evaluations are conducted on stratified test sets (20% of data) that were held out during the training phase to ensure unbiased performance assessment.

> **Note on Test Set Size:** After text preprocessing (stemming and cleaning), some reviews become empty strings and must be filtered before modeling. App Store has 8 empty strings (test set: 166 samples), and Play Store has 41 empty strings (test set: ~160 samples). See `docs/technical/DATA_FILTERING_NOTE.md` for details.

---

## 5.2 Dataset Distribution and Class Imbalance

### 5.2.1 Initial Lexicon-Based Sentiment Distribution

Before examining machine learning model performance, it is crucial to understand the baseline sentiment distribution produced by the lexicon-based labeling approach (described in Chapter 3). Table 5.1 presents the sentiment distribution across the entire dataset for both platforms.

**Table 5.1: Initial Lexicon Sentiment Distribution (Full Dataset)**

| Sentiment | App Store Count | App Store (%) | Play Store Count | Play Store (%) | Difference |
|-----------|-----------------|---------------|------------------|----------------|------------|
| Negatif   | 556             | 66.35%        | 689              | 82.22%         | +15.87%    |
| Netral    | 147             | 17.54%        | 90               | 10.74%         | -6.80%     |
| Positif   | 135             | 16.11%        | 59               | 7.04%          | -9.07%     |
| **Total** | **838**         | **100.00%**   | **838**          | **100.00%**    | -          |

#### Key Observations:

**Platform-Specific Patterns:**
- Play Store exhibits significantly higher negative sentiment (82.22%) compared to App Store (66.35%), representing a 15.87 percentage point difference
- App Store demonstrates a more balanced sentiment distribution with 17.54% Netral and 16.11% Positif reviews
- Play Store shows severe class imbalance with Negatif sentiment dominating over 82% of all reviews

**Class Imbalance Implications:**
- The severe imbalance on Play Store (82:11:7 ratio) creates a challenging learning environment where models may develop strong bias toward the dominant Negatif class
- App Store's moderate imbalance (66:18:16 ratio) provides better representation of minority classes, potentially enabling more balanced model learning
- Both platforms have limited Positif class samples, which may affect model ability to accurately identify positive sentiment

**User Behavior Differences:**
- The stark platform difference suggests distinct user populations or review motivations
- Play Store users appear more inclined to leave negative reviews, possibly due to different demographic profiles, technical issues, or review culture
- App Store users demonstrate more varied sentiment expression, with nearly one-third of reviews expressing neutral or positive opinions

### 5.2.2 Test Set Distribution

The stratified train-test split (80:20 ratio) preserves the class distribution in the test set, ensuring evaluation reflects real-world data characteristics. Table 5.2 shows the test set distribution used for all model evaluations.

**Table 5.2: Test Set Ground Truth Distribution**

| Sentiment | App Store Count | App Store (%) | Play Store Count | Play Store (%) |
|-----------|-----------------|---------------|------------------|----------------|
| Negatif   | 100             | 60.24%        | 93               | 58.49%         |
| Netral    | 40              | 24.10%        | 46               | 28.93%         |
| Positif   | 26              | 15.66%        | 20               | 12.58%         |
| **Total** | **166**         | **100.00%**   | **159**          | **100.00%**    |

**Note:** The test set sizes reflect filtering of empty strings after preprocessing. App Store filtered 8 empty strings (830 samples → 664 train, 166 test), and Play Store filtered 43 empty strings (795 samples → 636 train, 159 test). The test set accurately reflects the filtered dataset distribution, confirming that stratified sampling successfully maintained class proportions. This consistency ensures that evaluation results are representative of the overall data characteristics.

---

## 5.3 Model Performance Evaluation

This section presents comprehensive quantitative evaluation of both TF-IDF + SVM and IndoBERT + SVM models across both platforms. Performance is assessed using multiple metrics to provide a holistic view of model effectiveness.

### 5.3.1 Overall Performance Metrics

Table 5.3 summarizes the primary performance indicators for all models across both platforms.

**Table 5.3: Overall Model Performance Comparison**

| Platform   | Model            | Accuracy | Macro F1 | Weighted F1 | Test Samples |
|------------|------------------|----------|----------|-------------|--------------|
| App Store  | TF-IDF + SVM     | 66.87%   | **0.57** | 0.67        | 168          |
| App Store  | IndoBERT + SVM   | 66.27%   | 0.47     | 0.64        | 168          |
| Play Store | TF-IDF + SVM     | **73.21%** | 0.38   | **0.72**  | 168          |
| Play Store | IndoBERT + SVM   | 72.62%   | 0.33     | 0.71        | 168          |

#### Key Findings:

**Accuracy Analysis:**
- Play Store models post higher raw accuracy (73.21% TF-IDF, 72.62% IndoBERT) compared to App Store models (66.87% TF-IDF, 66.27% IndoBERT).
- TF-IDF achieves the highest measured accuracy overall (73.21% on Play Store), though this advantage must be interpreted alongside balanced-performance metrics.
- On App Store, TF-IDF and IndoBERT record nearly identical accuracy (66.87% vs. 66.27%), indicating feature-method differences manifest more strongly on balanced metrics.
- Play Store gains +6.34 percentage points accuracy (73.21% vs. 66.87% for TF-IDF), but at the cost of severely degraded minority-class performance (see macro F1 discussion below).

**Macro F1-Score Analysis:**
- TF-IDF consistently outperforms IndoBERT on macro F1 across both platforms (App: 0.57 vs. 0.47; Play: 0.38 vs. 0.33).
- **App Store TF-IDF delivers the best balanced performance** (macro F1 = 0.57), enabling reasonable detection of Netral and Positif classes despite imbalance.
- **Play Store exhibits severe macro F1 collapse** (0.38 for TF-IDF, 0.33 for IndoBERT), indicating that high accuracy stems from Negatif class dominance rather than true balanced capability.
- The accuracy-vs.-macro-F1 trade-off underscores the risk of relying solely on accuracy for imbalanced datasets; Play Store models may meet top-line metrics while failing to surface minority-class feedback.

**Weighted F1-Score Analysis:**
- Weighted F1-scores favor Play Store TF-IDF model (0.77), followed by App Store models (0.63 and 0.62)
- Weighted F1, which accounts for class imbalance by weighting each class by support, aligns closely with accuracy
- The convergence of weighted F1 and accuracy for all models confirms consistent performance across evaluation metrics

**Model Comparison:**
- TF-IDF + SVM outperforms IndoBERT + SVM when considering macro F1 (class-balanced metric) on both platforms
- The performance gap is most pronounced on Play Store: TF-IDF achieves 0.38 macro F1 vs IndoBERT's 0.33 (0.05 difference), though both models exhibit severe minority-class degradation
- Despite IndoBERT's theoretical advantage in capturing contextual semantics, TF-IDF's simpler bag-of-words representation proves more effective for this Indonesian sentiment classification task
- IndoBERT shows slightly better accuracy on App Store but fails to maintain balanced performance across minority classes

### 5.3.2 TF-IDF + SVM Performance Analysis

#### 5.3.2.1 Confusion Matrix Analysis

The confusion matrix provides detailed insight into classification patterns, revealing which sentiment classes are most frequently confused. Figure 5.1 presents the confusion matrices for TF-IDF + SVM on both platforms.

**Table 5.4: TF-IDF + SVM Confusion Matrix - App Store**

|                      | Predicted Negatif | Predicted Netral | Predicted Positif |
|----------------------|-------------------|------------------|-------------------|
| **Actual Negatif**   | 88                | 18               | 5                 |
| **Actual Netral**    | 17                | 10               | 3                 |
| **Actual Positif**   | 11                | 3                | 13                |

**Table 5.5: TF-IDF + SVM Confusion Matrix - Play Store**

|                      | Predicted Negatif | Predicted Netral | Predicted Positif |
|----------------------|-------------------|------------------|-------------------|
| **Actual Negatif**   | 116               | 18               | 4                 |
| **Actual Netral**    | 13                | 4                | 1                 |
| **Actual Positif**   | 9                 | 2                | 1                 |

#### Confusion Matrix Insights:

**App Store Patterns:**
- Negatif class achieves 88/111 (79.3%) correct predictions, showing strong recall on the dominant class
- Netral class delivers 10/30 (33.3%) correct predictions, reflecting substantial difficulty with neutral sentiment boundary
- Positif class reaches 13/27 (48.1%) correct predictions, better than expected for a minority class
- Most common error: Misclassifying Netral as Negatif (17 cases) and Positif as Negatif (11 cases), indicating a bias toward negative sentiment
- Total 28 minority-class samples misclassified as Negatif, highlighting the challenge in distinguishing subtle linguistic cues

**Play Store Patterns:**
- **Critical imbalance effects visible:** Negatif class achieves 116/138 (84.1%) correct predictions, dominating outcomes
- **Netral class severely degraded:** Only 4/18 (22.2%) correct, indicating the model fails to recognize neutral expressions in an imbalanced scenario
- **Positif class collapses:** Only 1/12 (8.3%) correct, representing near-total failure to identify positive sentiment
- Total 22 minority-class samples misclassified as Negatif (13 Netral + 9 Positif), demonstrating extreme majority-class bias
- Despite high raw accuracy (73.21%), the model operates primarily as a Negatif detector, missing actionable feedback from other classes

**Cross-Platform Comparison:**
- Play Store posts higher raw accuracy (84.1% Negatif recall vs. 79.3% App Store), but this comes at catastrophic cost to minority classes
- App Store maintains moderate Netral/Positif recall (33.3%/48.1%) vs. Play Store's collapse (22.2%/8.3%)
- **App Store demonstrates superior balanced performance** despite lower headline accuracy, aligning with macro F1 findings (0.57 vs. 0.38)
- The confusion matrices reveal that Play Store's accuracy advantage is misleading; it stems from Negatif dominance rather than robust multiclass capability

#### 5.3.2.2 Per-Class Performance Metrics

Table 5.6 provides detailed per-class metrics for TF-IDF + SVM, enabling granular performance comparison.

**Table 5.6: TF-IDF + SVM Per-Class Metrics**

| Platform   | Class    | Precision | Recall | F1-Score | Support |
|------------|----------|-----------|--------|----------|---------|
| App Store  | Negatif  | 0.76      | 0.79   | 0.77     | 111     |
| App Store  | Netral   | 0.32      | 0.33   | 0.33     | 30      |
| App Store  | Positif  | 0.62      | 0.48   | 0.54     | 27      |
| Play Store | Negatif  | 0.84      | 0.84   | 0.84     | 138     |
| Play Store | Netral   | 0.17      | 0.22   | 0.19     | 18      |
| Play Store | Positif  | 0.17      | 0.08   | 0.11     | 12      |

#### Detailed Class Analysis:

**Negatif Class (Dominant Class):**
- Play Store: 0.84 F1-score with balanced precision (0.84) and recall (0.84), representing strong majority-class performance
- App Store: 0.77 F1-score with balanced precision (0.76) and recall (0.79), showing solid dominant-class performance
- High support (111–138 samples) enables robust learning for the Negatif category
- Play Store achieves a +0.07 F1 advantage (0.84 vs. 0.77), but this reflects dataset composition rather than superior model capability

**Netral Class (Middle Class):**
- **App Store: 0.33 F1-score** with low precision (0.32) and recall (0.33), indicating moderate difficulty distinguishing neutral sentiment
- **Play Store: 0.19 F1-score** with very low precision (0.17) and recall (0.22), revealing severe degradation due to imbalance
- Play Store suffers a −0.14 F1 disadvantage (0.19 vs. 0.33), underperforming App Store by 42% relative
- Despite having 18 Netral test samples, Play Store model correctly identifies only 4 (22.2% recall), missing most neutral feedback
- App Store maintains usable Netral detection (33.3% recall), though still challenging

**Positif Class (Minority Class):**
- **App Store: 0.54 F1-score** with precision (0.62) and recall (0.48), demonstrating reasonable minority-class handling
- **Play Store: 0.11 F1-score** with very low precision (0.17) and recall (0.08), representing near-total failure
- Play Store suffers a −0.43 F1 disadvantage (0.11 vs. 0.54), underperforming App Store by 80% relative
- With only 1 of 12 Positif samples correctly classified (8.3% recall), Play Store model essentially ignores positive sentiment
- App Store correctly identifies 13 of 27 Positif samples (48.1% recall), enabling meaningful positive feedback capture

**Performance Winners:**
- Negatif: Play Store (+0.07 F1) — misleading advantage due to class dominance
- Netral: **App Store (+0.14 F1)** — superior balanced performance
- Positif: **App Store (+0.43 F1)** — superior minority-class handling

**Conclusion:** App Store TF-IDF + SVM delivers superior balanced multiclass performance, with usable Netral/Positif detection (F1 = 0.33/0.54) compared to Play Store's collapse (F1 = 0.19/0.11). Play Store's higher accuracy (73.21% vs. 66.87%) is a statistical artifact of Negatif dominance (82.1% of test samples), not robust sentiment analysis capability. For production sentiment systems requiring actionable feedback across all classes, **App Store configuration is strongly preferred** (macro F1 = 0.57 vs. 0.38).

### 5.3.3 IndoBERT + SVM Performance Analysis

#### 5.3.3.1 Confusion Matrix Analysis

**Table 5.7: IndoBERT + SVM Confusion Matrix - App Store**

|                      | Predicted Negatif | Predicted Netral | Predicted Positif |
|----------------------|-------------------|------------------|-------------------|
| **Actual Negatif**   | 93                | 13               | 5                 |
| **Actual Netral**    | 23                | 4                | 3                 |
| **Actual Positif**   | 13                | 4                | 10                |

**Table 5.8: IndoBERT + SVM Confusion Matrix - Play Store**

|                      | Predicted Negatif | Predicted Netral | Predicted Positif |
|----------------------|-------------------|------------------|-------------------|
| **Actual Negatif**   | 118               | 16               | 4                 |
| **Actual Netral**    | 14                | 3                | 1                 |
| **Actual Positif**   | 10                | 2                | 0                 |

#### Confusion Matrix Insights:

**App Store Patterns:**
- Negatif class achieves 93/111 (83.8%) correct predictions, higher than TF-IDF (79.3%), indicating strong recall on the dominant class
- Netral class shows weak performance with only 4/30 (13.3%) correct, significantly worse than TF-IDF (33.3%)
- Positif class achieves 10/27 (37.0%) correct, worse than TF-IDF (48.1%)
- Strong negative bias: 23 Netral and 13 Positif reviews misclassified as Negatif (total 36 minority samples lost)
- IndoBERT's contextual embeddings fail to distinguish neutral/positive sentiment from negative, despite theoretical semantic advantages

**Play Store Patterns:**
- Strong Negatif performance: 118/138 (85.5%) correct, slightly higher than TF-IDF (84.1%)
- **Catastrophic Netral performance: 3/18 (16.7%) correct**, far worse than TF-IDF's already-poor 22.2%
- **Complete Positif failure: 0/12 (0.0%) correct**, representing total inability to detect positive sentiment (worse than TF-IDF's 8.3%)
- Extreme negative bias: 14 Netral and 10 Positif reviews misclassified as Negatif (total 24 minority samples lost)
- IndoBERT exacerbates Play Store's imbalance problem, turning the model into a pure Negatif classifier

**IndoBERT vs TF-IDF Comparison:**
- IndoBERT achieves marginally higher Negatif recall on both platforms (App: 83.8% vs 79.3%; Play: 85.5% vs 84.1%) but at catastrophic cost to minority classes
- **On App Store:** IndoBERT Netral recall drops to 13.3% (vs TF-IDF 33.3%, −60% relative); Positif recall drops to 37.0% (vs TF-IDF 48.1%, −23% relative)
- **On Play Store:** IndoBERT Netral recall collapses to 16.7% (vs TF-IDF 22.2%, −25% relative); Positif recall reaches 0.0% (vs TF-IDF 8.3%, total failure)
- Despite theoretical advantages of contextual embeddings, **IndoBERT shows stronger majority-class bias** than TF-IDF across both platforms
- TF-IDF's simpler bag-of-words representation proves substantially more effective for balanced multiclass classification under imbalance

#### 5.3.3.2 Per-Class Performance Metrics

**Table 5.9: IndoBERT + SVM Per-Class Metrics**

| Platform   | Class    | Precision | Recall | F1-Score | Support |
|------------|----------|-----------|--------|----------|---------|
| App Store  | Negatif  | 0.72      | 0.84   | 0.78     | 111     |
| App Store  | Netral   | 0.19      | 0.13   | 0.16     | 30      |
| App Store  | Positif  | 0.56      | 0.37   | 0.44     | 27      |
| Play Store | Negatif  | 0.83      | 0.86   | 0.84     | 138     |
| Play Store | Netral   | 0.14      | 0.17   | 0.15     | 18      |
| Play Store | Positif  | 0.00      | 0.00   | 0.00     | 12      |

#### Detailed Class Analysis:

**Negatif Class:**
- Play Store: 0.84 F1-score (identical to TF-IDF's 0.84), demonstrating equivalent dominant-class performance
- App Store: 0.78 F1-score (slightly higher than TF-IDF's 0.77, +0.01 difference)
- IndoBERT achieves comparable performance to TF-IDF on the dominant class, with balanced precision-recall
- Higher recall on App Store (0.84 vs TF-IDF 0.79) but lower precision (0.72 vs TF-IDF 0.76), indicating slightly more aggressive Negatif predictions
- On Play Store, IndoBERT matches TF-IDF on Negatif F1 (0.84), but this comes at the expense of minority-class performance

**Netral Class:**
- **App Store: 0.16 F1-score** (far worse than TF-IDF's 0.33, −0.17 difference, −52% relative)
- **Play Store: 0.15 F1-score** (far worse than TF-IDF's 0.19, −0.04 difference, −21% relative)
- IndoBERT shows severe degradation on Netral class across both platforms
- App Store achieves only 13.3% recall (4 of 30 samples), with low precision (0.19) indicating frequent false positives
- Play Store achieves only 16.7% recall (3 of 18 samples), with even lower precision (0.14)
- **IndoBERT's contextual embeddings fail to leverage semantic understanding** for the challenging neutral sentiment boundary

**Positif Class:**
- **App Store: 0.44 F1-score** (worse than TF-IDF's 0.54, −0.10 difference, −19% relative)
- **Play Store: 0.00 F1-score** (complete failure vs TF-IDF's 0.11; IndoBERT detects zero Positif samples)
- IndoBERT severely underperforms TF-IDF on Positif class across both platforms
- Play Store exhibits **total Positif class collapse**: precision, recall, and F1 all = 0.00, indicating the model never predicts Positif sentiment
- App Store shows moderate precision (0.56) but poor recall (0.37), correctly identifying only 10 of 27 Positif samples

**IndoBERT Performance Summary:**
- Achieves parity with TF-IDF on dominant class (Negatif) but at catastrophic cost to minority classes
- **Severely underperforms TF-IDF on both Netral (−52% App, −21% Play) and Positif (−19% App, total failure Play)**
- Despite theoretical advantages of contextual embeddings and pre-training on Indonesian text, IndoBERT shows **stronger majority-class bias** than TF-IDF
- Play Store IndoBERT represents the worst-performing configuration, with macro F1 = 0.33 (vs TF-IDF 0.38) and zero Positif detection
- **Conclusion:** For this imbalanced Indonesian sentiment task, TF-IDF's simpler bag-of-words representation with explicit n-gram features proves substantially more effective than IndoBERT's dense contextual embeddings
- Shows moderate performance on Netral class on both platforms
- Despite deeper semantic understanding, fails to fully leverage contextual information for balanced multiclass classification
- The largest performance gap is on Play Store Positif class (-0.28 F1 compared to TF-IDF)

### 5.3.4 Model Performance Summary

**Table 5.10: Best Model per Platform and Class**

| Platform   | Overall Winner (Macro F1) | Negatif Winner | Netral Winner | Positif Winner |
|------------|---------------------------|----------------|---------------|----------------|
| App Store  | **TF-IDF** (0.57)         | **IndoBERT** (0.78 F1) | **TF-IDF** (0.33 F1) | **TF-IDF** (0.62 F1) |
| Play Store | **TF-IDF** (0.38)         | TF-IDF / IndoBERT (0.84 F1 tie) | **TF-IDF** (0.19 F1) | **TF-IDF** (0.11 F1) |

**Note:** Play Store "winners" achieve poor absolute F1 scores (Netral 0.19, Positif 0.11), indicating severe imbalance effects. App Store TF-IDF is the only configuration delivering usable balanced performance.

**Key Evaluation Conclusions:**
1. **TF-IDF + SVM is the recommended model** based on macro F1-score (class-balanced metric): TF-IDF achieves 0.57 (App Store) and 0.38 (Play Store) vs IndoBERT's 0.47 and 0.33
2. TF-IDF handles minority classes significantly better, particularly on Play Store
3. IndoBERT achieves nearly identical accuracy to TF-IDF on App Store (66.27% vs 66.87%) but severely underperforms on balanced metrics (macro F1: 0.47 vs 0.57)
4. **App Store TF-IDF demonstrates superior balanced performance** (macro F1 = 0.57), maintaining usable minority-class detection (Netral F1 = 0.33, Positif F1 = 0.54)
5. TF-IDF's simpler bag-of-words representation proves more effective than IndoBERT's contextual embeddings for this Indonesian sentiment classification task

---

## 5.4 Sentiment Distribution Analysis

This section analyzes the predicted sentiment distribution on the test set and compares it to the ground truth distribution to identify prediction biases.

### 5.4.1 Ground Truth vs Predictions

**Table 5.11: Sentiment Distribution Comparison (Test Set)**

| Platform   | Sentiment | Ground Truth | GT %   | TF-IDF Pred | TF-IDF % | IndoBERT Pred | IndoBERT % |
|------------|-----------|--------------|--------|-------------|----------|---------------|------------|
| App Store  | Negatif   | 111          | 66.07% | 116         | 69.05%   | 129           | 76.79%     |
| App Store  | Netral    | 30           | 17.86% | 31          | 18.45%   | 21            | 12.50%     |
| App Store  | Positif   | 27           | 16.07% | 21          | 12.50%   | 18            | 10.71%     |
| Play Store | Negatif   | 138          | 82.14% | 138         | 82.14%   | 142           | 84.52%     |
| Play Store | Netral    | 18           | 10.71% | 24          | 14.29%   | 21            | 12.50%     |
| Play Store | Positif   | 12           | 7.14%  | 6           | 3.57%    | 5             | 2.98%      |

### 5.4.2 Prediction Bias Analysis

**Table 5.12: Prediction Bias (Predicted % - Ground Truth %)**

| Platform   | Model          | Negatif Bias | Netral Bias | Positif Bias |
|------------|----------------|--------------|-------------|--------------|
| App Store  | TF-IDF + SVM   | +2.98%       | +0.59%      | -3.57%       |
| App Store  | IndoBERT + SVM | +10.72%      | -5.36%      | -5.36%       |
| Play Store | TF-IDF + SVM   | 0.00%        | +3.58%      | -3.57%       |
| Play Store | IndoBERT + SVM | +2.38%       | +1.79%      | -4.16%       |

#### Key Bias Observations:

**App Store Bias Patterns:**
- TF-IDF shows minimal bias (+2.98% Negatif, +0.59% Netral, -3.57% Positif), with slight under-prediction of positive sentiment
- IndoBERT shows strong negative bias (+10.72% Negatif), over-predicting negative sentiment
- IndoBERT significantly under-predicts both Netral (-5.36%) and Positif (-5.36%) sentiment
- IndoBERT's negative bias is 3.6× stronger than TF-IDF (10.72% vs 2.98%), indicating a tendency to default to the dominant class

**Play Store Bias Patterns:**
- TF-IDF shows perfect Negatif calibration (0.00% bias), demonstrating exceptional alignment with ground truth
- TF-IDF slightly over-predicts Netral (+3.58%) and under-predicts Positif (-3.57%)
- IndoBERT shows minimal negative bias (+2.38%) and under-predicts Positif (-4.16%)
- Both models under-predict Positif class, reflecting the extreme imbalance challenge (only 7.14% of test samples)

**Cross-Platform Comparison:**
- **App Store IndoBERT shows the strongest negative bias** (+10.72%), indicating moderate but significant tendency toward the dominant class
- **Play Store TF-IDF achieves perfect Negatif calibration** (0.00% bias), matching ground truth distribution exactly
- **Positif class is consistently under-predicted** across all models and platforms (-3.57% to -5.36%), reflecting the inherent challenges of minority class identification under severe imbalance
- **TF-IDF maintains better distributional balance** than IndoBERT on App Store (max bias: 3.57% vs 10.72%); both models show similar patterns on Play Store

### 5.4.3 Implications for Deployment

The prediction bias analysis reveals important considerations for model deployment:

1. **TF-IDF provides better calibrated predictions** with minimal distribution shift from ground truth (max bias: 3.58% vs IndoBERT: 10.72%)
2. **IndoBERT's negative bias on App Store** (+10.72%) may lead to moderate over-reporting of negative sentiment in production
3. **Positif sentiment under-prediction** across all models (−3.57% to −5.36%) means systematic underestimation of customer satisfaction
4. **Play Store deployment** must account for extreme negative skew (82.14% Negatif), where minority-class detection is severely compromised
5. **App Store deployment is strongly preferred** due to superior balanced performance (macro F1 = 0.57) and more usable minority-class detection (Netral F1 = 0.33, Positif F1 = 0.62)

---

## 5.5 Rating vs Lexicon Score Correlation Analysis

This section examines the relationship between numerical user ratings (1-5 stars) and lexicon-based sentiment scores to assess the validity of the lexicon labeling approach and understand factors influencing user ratings.

### 5.5.1 Correlation Metrics

**Table 5.13: Rating-Lexicon Correlation Metrics**

| Metric                    | App Store | Play Store | Better Correlation |
|---------------------------|-----------|------------|-------------------|
| **MAE (Mean Absolute Error)** | 1.2387    | 1.4672     | App Store         |
| **RMSE (Root Mean Square Error)** | 1.6231    | 1.8453     | App Store         |
| **Pearson Correlation**   | 0.4896    | 0.3824     | App Store (+0.11) |
| **Spearman Correlation**  | 0.4854    | 0.3791     | App Store (+0.11) |

### 5.5.2 Correlation Interpretation

#### App Store Analysis:
- **Moderate positive correlation** (Pearson: 0.49, Spearman: 0.49) between user ratings and lexicon sentiment scores
- **MAE of 1.24** indicates that, on average, the lexicon score differs from the actual rating by approximately 1.24 stars
- **RMSE of 1.62** shows that larger errors exist, with some reviews having substantial rating-sentiment misalignment
- **Pearson ≈ Spearman** suggests a relatively linear relationship without strong outlier influence

#### Play Store Analysis:
- **Low-moderate positive correlation** (Pearson: 0.38, Spearman: 0.38) indicates weaker alignment between ratings and text sentiment
- **MAE of 1.47** (19% higher than App Store) shows larger average discrepancies
- **RMSE of 1.85** (14% higher than App Store) indicates more frequent large misalignments
- **Pearson ≈ Spearman** again suggests linearity but with weaker overall relationship

#### Cross-Platform Comparison:
- **App Store demonstrates 11-point stronger correlation** (0.49 vs 0.38), indicating more consistent alignment between user ratings and review text sentiment
- **Play Store exhibits higher prediction errors** (MAE: 1.47 vs 1.24, RMSE: 1.85 vs 1.62)
- **App Store lexicon labeling is more reliable** as a proxy for user satisfaction
- The moderate correlations on both platforms (< 0.5) suggest that user ratings are influenced by factors beyond text sentiment

### 5.5.3 Factors Contributing to Rating-Sentiment Misalignment

The imperfect correlations can be attributed to several factors:

**Technical vs Sentiment Factors:**
- Users may rate based on app functionality (loading speed, crashes, bugs) while writing about content quality
- Technical issues often receive low ratings regardless of sentiment toward content
- Example: "Great content but the app crashes" → Low rating, mixed sentiment

**Expectation Mismatches:**
- Users with high expectations may express negative sentiment despite adequate service
- Platform-specific pricing or feature availability influences ratings
- Comparison to competitors affects rating scales

**Lexicon Limitations:**
- InSet lexicon may not capture domain-specific sentiment (e.g., "buffering" is negative for streaming but neutral in lexicon)
- Sarcasm, irony, and complex emotions are difficult for lexicon-based approaches
- Colloquial language and spelling variations reduce lexicon matching accuracy

**Platform-Specific Behavior:**
- Play Store users may be more review-averse, only reviewing during extreme satisfaction or dissatisfaction
- App Store users may provide more balanced feedback across the rating spectrum
- Different demographic profiles and technical literacy levels affect review styles

**Cultural and Linguistic Factors:**
- Indonesian language nuances (e.g., understated complaints) may not align with lexicon intensity
- Indirect criticism common in Indonesian culture may be classified as neutral despite negative intent
- Code-switching between Indonesian and English affects lexicon coverage

### 5.5.4 Implications for Model Evaluation

The moderate rating-sentiment correlations validate several research decisions:

1. **Lexicon labeling is a reasonable but imperfect proxy** for ground truth sentiment
2. **Machine learning models can potentially improve upon lexicon-based labels** by learning context-specific sentiment patterns
3. **Cross-platform differences justify separate model evaluation** rather than assuming platform-agnostic performance
4. **Multi-dimensional evaluation** (including rating correlation) provides more comprehensive model assessment than accuracy alone
5. **Domain adaptation** of the InSet lexicon for streaming service reviews could improve initial labeling accuracy

---

## 5.6 Word Cloud Analysis: Linguistic Patterns

This section presents word cloud analysis to visualize dominant keywords and themes within each sentiment category across both platforms. The analysis reveals user concerns, satisfaction drivers, and platform-specific linguistic patterns.

### 5.6.1 Negatif Sentiment Keywords

**Table 5.14: Top Keywords in Negatif Sentiment Reviews**

| Platform   | Review Count | Top 15 Keywords |
|------------|--------------|-----------------|
| **App Store**  | 556 (66%)    | aplikasi, lag, error, jelek, kecewa, buruk, lemot, crash, tidak, gagal, loading, bug, lambat, payah, buffering |
| **Play Store** | 689 (82%)    | aplikasi, tidak, error, jelek, buruk, ga, lemot, lag, crash, kecewa, loading, bug, payah, eror, sering |

#### Negatif Sentiment Insights:

**Common Technical Complaints (Both Platforms):**
- **Performance issues**: lag, lemot (slow), lambat (slow), loading
- **Stability problems**: crash, error, bug
- **Quality descriptors**: jelek (bad), buruk (poor), kecewa (disappointed), payah (terrible)

**Platform-Specific Terms:**
- **App Store emphasis**: "buffering" (streaming-specific), "gagal" (failed)
- **Play Store emphasis**: "ga" (informal contraction of "tidak"), "sering" (often/frequently), "eror" (spelling variant of error)

**Linguistic Observations:**
- Technical English terms (lag, crash, error, bug, loading) dominate both platforms, indicating these are standardized vocabulary for technical complaints
- "aplikasi" (application) is the most frequent term, as users reference the app itself
- Play Store shows more informal language use ("ga" vs "tidak")
- Spelling variations ("error" vs "eror") are more common on Play Store

**User Frustration Themes:**
1. **Performance degradation**: Multiple terms related to slowness and lag indicate this is a primary pain point
2. **Reliability issues**: Crashes and errors are frequent complaints across both platforms
3. **Streaming quality**: Buffering and loading problems directly affect user experience
4. **Emotional response**: Words like "kecewa" (disappointed) reveal user frustration beyond technical descriptions

### 5.6.2 Netral Sentiment Keywords

**Table 5.15: Top Keywords in Netral Sentiment Reviews**

| Platform   | Review Count | Top 15 Keywords |
|------------|--------------|-----------------|
| **App Store**  | 147 (18%)    | aplikasi, disney, hotstar, nonton, film, drama, konten, streaming, coba, paket, biasa, ok, lumayan, standar, bisa |
| **Play Store** | 90 (11%)     | aplikasi, disney, hotstar, nonton, film, konten, drama, bagus, coba, subscribe, biasa, ok, kurang, ada, tapi |

#### Netral Sentiment Insights:

**Common Descriptive Terms (Both Platforms):**
- **Brand references**: disney, hotstar (users identify the service)
- **Content terms**: nonton (watch), film (movie), drama, konten (content)
- **Usage verbs**: coba (try), streaming
- **Mild evaluations**: biasa (ordinary), ok

**Platform-Specific Terms:**
- **App Store emphasis**: "paket" (package/subscription), "lumayan" (decent/not bad), "standar" (standard)
- **Play Store emphasis**: "subscribe" (English term for subscription), "kurang" (lacking), "tapi" (but), "ada" (there is), "bagus" (good)

**Linguistic Observations:**
- Netral reviews focus heavily on content and functionality rather than strong emotions
- Terms like "biasa" and "ok" indicate moderate, non-committal sentiment
- App Store uses more Indonesian terms while Play Store mixes Indonesian and English
- The presence of "bagus" (good) in Play Store neutral reviews suggests positive elements mixed with criticisms

**User Behavior Themes:**
1. **Exploratory reviews**: Words like "coba" (try) indicate users testing the service
2. **Content-focused**: Film, drama, and content are central to neutral discussions
3. **Qualified opinions**: Terms like "lumayan" (decent), "standar" (standard), and "kurang" (lacking) show balanced views
4. **Subscription discussions**: References to packages and subscribing indicate cost-benefit considerations

### 5.6.3 Positif Sentiment Keywords

**Table 5.16: Top Keywords in Positif Sentiment Reviews**

| Platform   | Review Count | Top 15 Keywords |
|------------|--------------|-----------------|
| **App Store**  | 135 (16%)    | bagus, mantap, keren, suka, rekomendasi, terbaik, lancar, puas, lengkap, sempurna, aplikasi, film, drama, konten, disney |
| **Play Store** | 59 (7%)      | bagus, mantap, keren, lancar, aplikasi, suka, puas, rekomendasi, terbaik, film, disney, lengkap, konten, sempurna, top |

#### Positif Sentiment Insights:

**Common Praise Terms (Both Platforms):**
- **Quality adjectives**: bagus (good), mantap (excellent), keren (cool), terbaik (best), sempurna (perfect)
- **Satisfaction**: suka (like), puas (satisfied), rekomendasi (recommendation)
- **Performance**: lancar (smooth/seamless)
- **Content**: lengkap (complete/comprehensive)

**Platform-Specific Terms:**
- **App Store**: More diverse positive vocabulary
- **Play Store**: Adds "top" (English slang for excellent)

**Linguistic Observations:**
- Positive reviews use strong Indonesian adjectives (mantap, keren, sempurna)
- "Lancar" (smooth/seamless) indicates satisfaction with technical performance
- Recommendation language ("rekomendasi") suggests word-of-mouth potential
- Both platforms praise content completeness ("lengkap")

**User Satisfaction Themes:**
1. **Performance satisfaction**: "Lancar" (smooth) indicates the app works well when functioning properly
2. **Content quality**: References to film, drama, and content completeness drive positive sentiment
3. **Recommendation willingness**: Presence of "rekomendasi" suggests satisfied users promote the service
4. **Enthusiastic language**: Terms like "mantap", "keren", "sempurna" show genuine satisfaction

### 5.6.4 Cross-Platform Linguistic Comparison

**Table 5.17: Key Linguistic Differences**

| Aspect | App Store | Play Store |
|--------|-----------|------------|
| **Formality** | More formal Indonesian | More informal, colloquial |
| **Code-switching** | Limited English terms | More English-Indonesian mixing |
| **Spelling** | Standard spelling | More variants (eror, ga) |
| **Vocabulary diversity** | Higher (especially positive) | Lower |
| **Technical terms** | English technical terms | English technical + variants |

#### Cross-Platform Insights:

1. **Platform demographics**: Play Store appears to have younger or more casual users based on informal language use
2. **Education levels**: App Store users may have higher formal education based on standard spelling and vocabulary
3. **Cultural factors**: Both platforms use Indonesian sentiment expressions, validating the use of InSet lexicon
4. **Domain language**: Technical English terms (lag, crash, error) are universal across platforms, suggesting these are standard vocabulary

### 5.6.5 Implications for Sentiment Analysis

The word cloud analysis provides several important insights for sentiment classification:

**Model Training Considerations:**
1. **Technical English terms** should be preserved during preprocessing (lag, crash, error, loading)
2. **Spelling variations** (eror vs error, ga vs tidak) suggest the need for normalization or variant handling
3. **Domain-specific terms** (buffering, streaming, loading) carry strong negative sentiment in this context
4. **Colloquial language** on Play Store may challenge models trained on formal text

**Feature Engineering Implications:**
1. TF-IDF's success may be attributed to capturing key discriminative terms (lag, crash for negative; bagus, mantap for positive)
2. IndoBERT's underperformance suggests pre-training corpus may not cover streaming app domain or colloquial language
3. N-gram features capture multi-word expressions important for sentiment (e.g., "sangat bagus" - very good)

**Lexicon Enhancement Opportunities:**
1. Add streaming-specific negative terms: buffering, loading, crash, lag
2. Recognize informal variants: ga (tidak), eror (error)
3. Include English technical terms with appropriate sentiment weights
4. Incorporate domain-specific quality descriptors: lemot, lancar

---

## 5.7 Cross-Platform Key Findings

This section synthesizes evaluation results into actionable insights comparing App Store and Play Store performance.

### 5.7.1 Model Performance Findings

**Finding 1: Play Store Achieves Higher Overall Accuracy**
- Play Store TF-IDF: 73.21%, IndoBERT: 72.62%
- App Store TF-IDF: 66.87%, IndoBERT: 66.27%
- **Conclusion**: The 6.3-6.4 percentage point advantage is primarily driven by the dominant Negatif class (82% vs 66%)

**Finding 2: App Store Achieves Better Macro F1-Scores**
- App Store TF-IDF: 0.57, IndoBERT: 0.47
- Play Store TF-IDF: 0.38, IndoBERT: 0.33
- **Conclusion**: App Store models better handle all classes equally, indicating superior balanced performance

**Finding 3: Play Store Accuracy is Concentrated in Dominant Class**
- Weighted F1-scores (0.71-0.72) closely match accuracy (72-73%)
- **Conclusion**: High accuracy reflects strong Negatif class performance rather than balanced multi-class competence

**Finding 4: TF-IDF Consistently Outperforms IndoBERT**
- TF-IDF macro F1 advantage: 0.10 (App Store), 0.05 (Play Store)
- **Conclusion**: Simpler bag-of-words representation proves more effective than contextual embeddings for this task

### 5.7.2 Class-Specific Performance Findings

**Finding 5: Negatif Class Performs Best on Both Platforms**
- F1-scores range from 0.78-0.84 across models and platforms
- **Conclusion**: Abundant training data (66-82% of dataset) enables robust negative sentiment identification

**Finding 6: Netral Class is Most Challenging**
- F1-scores range from 0.15-0.30, lowest across all classes
- **Conclusion**: Limited samples (11-18% of dataset) and ambiguous language make neutral sentiment difficult to distinguish

**Finding 7: Positif Class Shows Dramatic Platform Difference**
- App Store F1: 0.47-0.62 (TF-IDF: 0.62)
- Play Store F1: 0.00-0.11 (IndoBERT: 0.00)
- **Conclusion**: Severe class imbalance on Play Store (7% Positif) leads to model inability to identify positive sentiment

**Finding 8: App Store Models Handle Minority Classes Better**
- Despite lower overall accuracy, App Store achieves higher F1 for Netral (+0.11-0.15) and Positif (+0.36-0.51)
- **Conclusion**: More balanced training distribution (66:18:16) enables learning of minority class patterns

### 5.7.3 Distribution and Bias Findings

**Finding 9: Initial Lexicon Distribution Differs Significantly**
- Play Store: 82% Negatif vs App Store: 66% Negatif (15.87 point difference)
- **Conclusion**: Play Store user base is substantially more negative or has different review behavior

**Finding 10: Play Store Shows Severe Class Imbalance**
- Play Store ratio: 82:11:7 (Negatif:Netral:Positif)
- App Store ratio: 66:18:16
- **Conclusion**: Extreme imbalance challenges model generalization to minority classes

**Finding 11: Both Models Show Negative Bias**
- App Store IndoBERT: +10.72% Negatif bias (strongest)
- Play Store TF-IDF: Perfect Negatif alignment (0% bias)
- **Conclusion**: Models tend to over-predict dominant class, but TF-IDF maintains better calibration

**Finding 12: Positif Class Consistently Under-Predicted**
- All models under-predict Positif by 3.57-5.36 percentage points
- **Conclusion**: Minority class under-representation leads to systematic under-prediction

### 5.7.4 Rating-Sentiment Correlation Findings

**Finding 13: App Store Shows Stronger Rating-Sentiment Correlation**
- App Store: 0.49 vs Play Store: 0.38 (Pearson correlation)
- **Conclusion**: App Store lexicon labeling more accurately reflects user satisfaction

**Finding 14: Play Store Has Higher Prediction Errors**
- MAE: 1.47 vs 1.24, RMSE: 1.85 vs 1.62
- **Conclusion**: Greater misalignment between Play Store ratings and text sentiment

**Finding 15: Moderate Correlations Suggest Multi-Factor Ratings**
- Both platforms show correlations below 0.5
- **Conclusion**: User ratings influenced by factors beyond text sentiment (technical issues, expectations, pricing)

### 5.7.5 Linguistic Pattern Findings

**Finding 16: Technical Issues Dominate Negative Sentiment**
- Common terms: lag, error, crash, loading, lemot (both platforms)
- **Conclusion**: Performance and stability are primary drivers of dissatisfaction

**Finding 17: Play Store Uses More Informal Language**
- "ga" vs "tidak", "eror" vs "error", spelling variations
- **Conclusion**: Reflects different user demographics or cultural norms

**Finding 18: Content Quality Drives Positive Sentiment**
- Film, drama, konten, lengkap appear in positive reviews
- **Conclusion**: Content completeness and quality are key satisfaction factors

---

## 5.8 Model Selection and Deployment Recommendations

Based on comprehensive evaluation across quantitative metrics, distribution analysis, correlation assessment, and linguistic patterns, this section provides data-driven recommendations for model deployment.

### 5.8.1 Recommended Models

**Table 5.18: Model Selection Recommendations**

| Platform   | Recommended Model | Justification |
|------------|-------------------|---------------|
| **App Store**  | **TF-IDF + SVM**  | • Highest macro F1 (0.57 vs 0.47)<br>• Best minority class performance (Netral: 0.30, Positif: 0.62)<br>• Minimal prediction bias (+2.98% Negatif)<br>• Accuracy: 66.87% |
| **Play Store** | **TF-IDF + SVM**  | • Highest macro F1 (0.38 vs 0.33)<br>• Better minority class performance (Netral: 0.19, Positif: 0.11 vs 0.00)<br>• Perfect Negatif alignment (0% bias)<br>• Accuracy: 73.21% |

**Rationale for TF-IDF Selection:**
1. **Macro F1-score prioritized** over raw accuracy to account for class imbalance
2. **Consistent superiority** across both platforms for minority class identification
3. **Better calibration** with minimal distributional shift from ground truth
4. **Simpler implementation** with lower computational requirements
5. **Interpretability** through feature importance analysis of discriminative terms

### 5.8.2 Model Strengths and Limitations

**TF-IDF + SVM Strengths:**
- Captures key discriminative terms (lag, crash, error for negative; bagus, mantap for positive)
- Handles high-dimensional sparse features effectively
- Robust to spelling variations through subword patterns in n-grams
- Fast training and prediction suitable for production deployment
- Transparent feature weights enable explainable predictions

**TF-IDF + SVM Limitations:**
- Struggles with Netral class identification (F1: 0.19-0.30)
- Limited Positif class performance on Play Store (F1: 0.11) due to severe class imbalance
- Cannot capture word order or contextual nuances
- Requires domain-specific feature engineering for optimal performance
- Sensitive to vocabulary drift over time

**IndoBERT + SVM Weaknesses:**
- Lower macro F1-scores despite theoretical semantic advantages
- Stronger bias toward dominant class
- Complete failure on Play Store Positif class (F1: 0.00)
- Higher computational cost without performance benefit
- Pre-training corpus may not cover streaming app domain or colloquial Indonesian

### 5.8.3 Deployment Considerations

**Pre-Deployment Steps:**
1. **Class Imbalance Mitigation**: Implement class weighting or SMOTE for minority classes
2. **Threshold Tuning**: Adjust classification thresholds to balance precision-recall trade-offs
3. **Confidence Calibration**: Apply Platt scaling or isotonic regression to improve probability estimates
4. **Monitoring Pipeline**: Establish continuous evaluation to detect distribution drift

**Platform-Specific Strategies:**

**App Store Deployment:**
- Leverage relatively balanced distribution for fair multi-class predictions
- Focus on maintaining Positif class performance (F1: 0.62)
- Monitor for negative bias drift (currently minimal at +2.98%)
- Implement confidence thresholds for Netral class (low precision: 0.28)

**Play Store Deployment:**
- Acknowledge severe class imbalance (82% Negatif) in deployment design
- Implement cost-sensitive learning to improve minority class detection
- Consider separate binary classifiers (Negatif vs non-Negatif, then Netral vs Positif)
- Accept that Positif identification will be challenging (baseline F1: 0.11)
- Prioritize high precision for Negatif class (currently 0.84)

### 5.8.4 Future Improvement Strategies

**Short-Term Improvements:**
1. **Data Augmentation**: Generate synthetic minority class samples using back-translation or paraphrasing
2. **Ensemble Methods**: Combine TF-IDF predictions with rule-based sentiment for edge cases
3. **Feature Engineering**: Add domain-specific lexicon features (streaming-specific terms)
4. **Hyperparameter Tuning**: Optimize class weights and SVM C parameter per platform

**Long-Term Improvements:**
1. **Active Learning**: Prioritize manual labeling of Netral and Positif samples
2. **Domain Adaptation**: Fine-tune IndoBERT on streaming app reviews before feature extraction
3. **Multi-Task Learning**: Train models jointly on sentiment classification and rating prediction
4. **Cross-Platform Transfer**: Leverage App Store's balanced data to improve Play Store minority class performance
5. **Lexicon Enhancement**: Expand InSet with streaming-specific terms and colloquialisms

### 5.8.5 Business Impact Considerations

**Sentiment Analysis Use Cases:**
1. **Customer Satisfaction Monitoring**: Track Negatif sentiment trends to identify emerging issues
2. **Feature Prioritization**: Extract Negatif sentiment keywords to prioritize bug fixes (lag, crash, loading)
3. **Content Strategy**: Analyze Positif sentiment to understand what content resonates with users
4. **Platform Comparison**: Use cross-platform analysis to understand demographic differences
5. **Competitive Intelligence**: Compare Disney+ Hotstar sentiment to competitors

**Model Limitations for Business Decisions:**
- **Positif sentiment under-prediction**: Will systematically underestimate customer satisfaction
- **Netral class confusion**: Ambiguous reviews may be misclassified as Negatif
- **Platform bias**: Play Store analysis will be heavily weighted toward negative sentiment
- **Temporal drift**: Model trained on 2020-2024 data may degrade as language evolves

**Risk Mitigation:**
- Implement human-in-the-loop review for high-stakes decisions
- Provide confidence scores alongside predictions
- Establish regular model retraining schedules (quarterly or bi-annually)
- Maintain manual QA sampling to detect systematic errors

---

## 5.9 Conclusions

This evaluation phase comprehensively assessed TF-IDF + SVM and IndoBERT + SVM models for Indonesian sentiment classification of Disney+ Hotstar reviews across App Store and Play Store platforms. The key conclusions are:

**Model Performance:**
- TF-IDF + SVM emerges as the superior approach on both platforms, achieving higher macro F1-scores (0.57 on App Store, 0.38 on Play Store)
- Despite IndoBERT's theoretical advantages in capturing contextual semantics, simpler bag-of-words representation proves more effective for this sentiment classification task
- Play Store achieves higher accuracy (73.21%) but App Store demonstrates better balanced performance across all classes

**Class-Specific Challenges:**
- Negatif class performs strongly on both platforms (F1: 0.78-0.84) due to abundant training data
- Netral class presents the greatest challenge (F1: 0.15-0.30) across all models and platforms
- Positif class shows dramatic platform differences: App Store achieves reasonable performance (F1: 0.62) while Play Store struggles severely (F1: 0.11, or 0.00 for IndoBERT)

**Platform Differences:**
- Play Store exhibits severe class imbalance (82% Negatif) compared to App Store's moderate imbalance (66% Negatif)
- App Store shows stronger correlation between ratings and sentiment (0.49 vs 0.38), indicating more reliable lexicon labeling
- Play Store uses more informal and colloquial language, reflecting different user demographics
- Technical complaints (lag, crash, error) dominate negative sentiment on both platforms

**Deployment Readiness:**
- TF-IDF + SVM is recommended for both platforms based on superior macro F1-scores and better minority class handling
- Models are production-ready with appropriate confidence thresholds and monitoring
- Platform-specific deployment strategies are necessary to address class imbalance differences
- Continuous evaluation and periodic retraining are essential to maintain performance

**Research Contributions:**
- First comprehensive cross-platform comparison of Indonesian sentiment analysis for streaming services
- Demonstrates that traditional TF-IDF features outperform modern transformer embeddings for this task
- Provides empirical evidence of platform-specific linguistic patterns and user behavior differences
- Establishes baseline performance metrics for future Indonesian sentiment analysis research in the streaming domain

The evaluation phase successfully validates the modeling approach while identifying clear areas for future improvement, particularly in minority class identification and handling severe class imbalance on Play Store.

---

**End of Chapter 5: Evaluation Phase**

---

## References

1. Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., & Wirth, R. (2000). CRISP-DM 1.0: Step-by-step data mining guide.
2. Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. Information Processing & Management, 45(4), 427-437.
3. He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263-1284.
4. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL-HLT 2019.
5. Koto, F., et al. (2020). IndoLEM and IndoBERT: A benchmark dataset and pre-trained language model for Indonesian NLP. COLING 2020.
6. Adiwijaya, M. S., et al. (2017). InSet Lexicon: An Indonesian sentiment lexicon for text analysis. Journal of ICT Research and Applications, 11(1), 24-44.

---

**Document Information**

- **Title**: Chapter 5 - Evaluation Phase: Model Assessment and Cross-Platform Analysis
- **Project**: Disney+ Hotstar Indonesian Sentiment Analysis
- **Methodology**: CRISP-DM (Cross-Industry Standard Process for Data Mining)
- **Author**: [Your Name]
- **Institution**: [Your Institution]
- **Date**: November 3, 2025
- **Version**: 1.0

**Data Sources:**
- Tesis-Appstore-FIX.ipynb (App Store analysis notebook)
- Tesis-Playstore-FIX.ipynb (Play Store analysis notebook)
- Total samples: 1,676 reviews (838 per platform)
- Test set: 336 reviews (168 per platform, 20% stratified split)
- Date range: 2020-2024

**Evaluation Metrics:**
- Classification: Accuracy, Precision, Recall, F1-Score (Macro & Weighted)
- Correlation: Pearson & Spearman coefficients, MAE, RMSE
- Visual: Confusion matrices, Word clouds

**Models Evaluated:**
1. TF-IDF + SVM (Recommended)
2. IndoBERT + SVM

**Recommended Model:** TF-IDF + SVM (both platforms)
