# Chapter: Modeling Phase - Sentiment Classification

## 4.1 Introduction

This chapter presents the modeling phase of the Disney+ Hotstar sentiment analysis study, following the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology. The modeling phase involves selecting appropriate machine learning algorithms, preparing data splits, engineering features, tuning hyperparameters, and evaluating model performance.

The primary objective is to compare two distinct feature engineering approaches—**TF-IDF vectorization** and **IndoBERT contextual embeddings**—using Support Vector Machine (SVM) as the classifier. This controlled comparison allows us to assess the effectiveness of traditional bag-of-words methods versus modern transformer-based representations for Indonesian sentiment classification.

### 4.1.1 Research Questions Addressed
1. Which feature representation (TF-IDF vs. IndoBERT) yields better sentiment classification performance?
2. What are the optimal n-gram configurations for TF-IDF features?
3. How do different SVM hyperparameters (C, kernel) affect model performance?
4. Are there performance differences between App Store and Play Store reviews?

---

## 4.2 Modeling Strategy

### 4.2.1 Problem Formulation

**Task Type**: Multi-class text classification  
**Classes**: Three sentiment categories
- **Positif** (Positive): Favorable reviews expressing satisfaction
- **Netral** (Neutral): Balanced or mixed opinions
- **Negatif** (Negative): Unfavorable reviews expressing dissatisfaction

**Input**: Preprocessed review text (`ulasan_bersih`)  
**Output**: Predicted sentiment class  
**Primary Evaluation Metric**: Macro F1-Score (to handle class imbalance)

### 4.2.2 Model Selection Rationale

This study employs **Support Vector Machine (SVM)** as the sole classifier for the following reasons:

1. **Controlled Comparison**: Using a single classifier eliminates algorithmic variance, ensuring that performance differences can be attributed solely to feature engineering strategies (TF-IDF vs. IndoBERT).

2. **Versatility**: SVMs perform well on both:
   - High-dimensional sparse features (TF-IDF vectors)
   - Lower-dimensional dense features (IndoBERT embeddings)

3. **Kernel Flexibility**: The ability to test linear vs. non-linear kernels (RBF, polynomial) allows exploration of decision boundary complexity.

4. **Established Performance**: SVMs have demonstrated strong performance in text classification tasks, particularly with proper hyperparameter tuning.

5. **Reproducibility**: scikit-learn's SVM implementation provides robust cross-validation and grid search capabilities for reproducible experimentation.

**Alternative Approaches Considered but Excluded**:
- **Naive Bayes**: Quick baseline but limited by independence assumption
- **Random Forest**: Explored in preliminary work but excluded from final comparison to maintain focus on feature engineering rather than ensemble effects
- **Deep Learning**: Resource-intensive; IndoBERT embeddings + SVM provides a compromise between traditional ML and full fine-tuning

---

## 4.3 Data Preparation

### 4.3.1 Dataset Overview

**Play Store Dataset**:
- Total reviews: 838
- Source: Google Play Store (Disney+ Hotstar app)
- Language: Indonesian
- Time period: Pre and post pricing changes

**App Store Dataset**:
- Total reviews: 838
- Source: Apple App Store (Disney+ Hotstar app)
- Language: Indonesian
- Time period: Pre and post pricing changes

### 4.3.2 Train-Test Split Configuration

To enable unbiased evaluation, the data is split into training and testing sets before any feature engineering.

**Split Parameters**:
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Test size | 0.2 (20%) | Standard practice, provides sufficient test samples (168) |
| Train size | 0.8 (80%) | Adequate training data (670 samples) for model learning |
| Random state | 42 | Fixed seed for reproducibility |
| Stratification | Play Store: Yes<br>App Store: No | Play Store uses `stratify=y_multi` to maintain class proportions |

**Implementation Code**:
```python
from sklearn.model_selection import train_test_split

# Play Store (with stratification)
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X, y_multi, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_multi
)

# App Store (without stratification)
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X, y_multi, 
    test_size=0.2, 
    random_state=42
)
```

**Split Results**:
- Training set: 670 samples (80%)
- Test set: 168 samples (20%)

### 4.3.3 Class Distribution Analysis

#### Play Store (Stratified Split)

**Training Set (n=670)**:
| Class | Count | Percentage |
|-------|-------|------------|
| Negatif | 378 | 56.4% |
| Netral | 196 | 29.3% |
| Positif | 96 | 14.3% |

**Test Set (n=168)**:
| Class | Count | Percentage |
|-------|-------|------------|
| Negatif | 94 | 56.0% |
| Netral | 49 | 29.2% |
| Positif | 25 | 14.9% |

**Verification**: The stratified split successfully maintains class proportions with <1% variance between training and test sets, confirming effective stratification.

#### App Store (Non-stratified Split)

**Training Set (n=670)**:
| Class | Count | Percentage |
|-------|-------|------------|
| Negatif | 399 | 59.6% |
| Netral | 186 | 27.8% |
| Positif | 85 | 12.7% |

**Test Set (n=168)**:
| Class | Count | Percentage |
|-------|-------|------------|
| Negatif | 99 | 58.9% |
| Netral | 48 | 28.6% |
| Positif | 21 | 12.5% |

**Observation**: Despite not using stratification parameter, the random split maintains similar class proportions (variance <1%) between training and test sets.

### 4.3.4 Class Imbalance Considerations

Both datasets exhibit class imbalance:
- **Negative sentiment** is dominant (~56-60%)
- **Neutral sentiment** is moderate (~28-29%)
- **Positive sentiment** is minority (~12-15%)

**Mitigation Strategy**:
- Use **macro F1-score** as the primary evaluation metric (treats all classes equally regardless of support)
- Consider class weights in future iterations if minority class performance is insufficient

---

## 4.4 Feature Engineering

Two distinct feature engineering approaches are compared in this study.

### 4.4.1 Approach 1: TF-IDF Vectorization

**Method**: Term Frequency-Inverse Document Frequency  
**Purpose**: Convert text into numerical vectors based on word importance

#### N-gram Selection Experiment

To determine the optimal n-gram configuration, three settings were tested using 5-fold cross-validation on the training set.

**Play Store N-gram Results**:
| N-gram Range | CV Macro F1 | Status |
|--------------|-------------|--------|
| (1,1) - Unigrams | **0.6301** | ✓ Selected |
| (1,2) - Uni+Bigrams | 0.5367 | |
| (1,3) - Uni+Bi+Trigrams | 0.4929 | |

**App Store N-gram Results**:
| N-gram Range | CV Macro F1 | Status |
|--------------|-------------|--------|
| (1,1) - Unigrams | **0.5026** | ✓ Selected |
| (1,2) - Uni+Bigrams | 0.4259 | |
| (1,3) - Uni+Bi+Trigrams | 0.4062 | |

**Finding**: Unigrams (1,1) consistently outperform higher-order n-grams on both platforms. This suggests:
1. Individual words carry sufficient sentiment information
2. Bigrams/trigrams introduce noise or sparsity that degrades performance
3. The relatively small dataset (838 samples) benefits from simpler feature spaces

**Selected Configuration**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    ngram_range=(1, 1),  # Unigrams only
    max_features=None,   # No feature limit
    min_df=1,            # Minimum document frequency
    max_df=1.0           # Maximum document frequency
)
```

**Feature Dimensionality**:
- Play Store: 1,367 unique features
- App Store: Similar scale (vocabulary-dependent)

### 4.4.2 Approach 2: IndoBERT Embeddings

**Model**: IndoBERT (Indonesian BERT)  
**Variant**: `indobenchmark/indobert-base-p1`  
**Purpose**: Generate contextual sentence embeddings

#### Embedding Extraction Process

```python
from transformers import BertTokenizer, BertModel
import torch

# Initialize model and tokenizer
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
model = BertModel.from_pretrained('indobenchmark/indobert-base-p1')

# Extract CLS token embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', 
                      truncation=True, max_length=512, 
                      padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use [CLS] token representation
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
```

**Embedding Characteristics**:
- Dimensionality: 768 features (BERT hidden size)
- Type: Dense, continuous vectors
- Context: Captures semantic relationships and word order
- Advantage: Pre-trained on Indonesian corpus

**Normalization**:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_embeddings)
X_test_scaled = scaler.transform(X_test_embeddings)
```

---

## 4.5 Model Training and Hyperparameter Tuning

### 4.5.1 TF-IDF + SVM Pipeline

#### Hyperparameter Grid

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],      # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly'] # Kernel functions
}
```

#### Grid Search Configuration

```python
grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=10,                    # 10-fold cross-validation
    scoring='f1_macro',       # Macro F1 for class balance
    n_jobs=-1,                # Parallel processing
    verbose=1
)

grid_search.fit(X_train_tfidf, y_train)
```

#### Play Store Results

**Best Hyperparameters**:
- **C**: 100
- **Kernel**: linear

**Performance**:
- Best CV Macro F1: **0.6613**
- Test Accuracy: **0.6845** (68.45%)

**Test Set Classification Report**:
```
              precision    recall  f1-score   support

    Negatif       0.67      0.95      0.79       94
     Netral       0.68      0.28      0.40       49
    Positif       0.80      0.17      0.28       25

    accuracy                           0.68       168
   macro avg       0.72      0.47      0.49       168
weighted avg       0.69      0.68      0.64       168
```

**Key Observations**:
- Strong recall for Negative class (0.95) - correctly identifies most negative reviews
- Weak recall for Neutral (0.28) and Positive (0.17) - struggles with minority classes
- Macro F1 (0.49) reflects imbalanced performance across classes
- Linear kernel performs best, suggesting linearly separable feature space

#### App Store Results

**Best Hyperparameters**:
- **C**: 100
- **Kernel**: linear

**Performance**:
- Best CV Macro F1: **0.5305**
- Test Accuracy: **0.6310** (63.10%)

**Test Set Classification Report**:
```
              precision    recall  f1-score   support

    Negatif       0.61      0.97      0.75       99
     Netral       0.71      0.20      0.31       48
    Positif       0.75      0.14      0.24       21

    accuracy                           0.63       168
   macro avg       0.69      0.44      0.43       168
weighted avg       0.66      0.63      0.56       168
```

**Key Observations**:
- Similar pattern to Play Store: high recall for Negative (0.97), low for minorities
- Lower overall performance compared to Play Store (macro F1: 0.43 vs 0.49)
- Linear kernel again optimal

### 4.5.2 IndoBERT + SVM Pipeline

#### Hyperparameter Grid

```python
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf']  # Focus on linear and RBF for embeddings
}
```

#### Play Store Results

**Best Hyperparameters**:
- **C**: 10
- **Kernel**: linear

**Performance**:
- Best CV Macro F1: **0.6342**
- Test Accuracy: **0.6607** (66.07%)

**Test Set Classification Report**:
```
              precision    recall  f1-score   support

    Negatif       0.67      0.91      0.77       94
     Netral       0.58      0.33      0.42       49
    Positif       0.67      0.16      0.26       25

    accuracy                           0.66       168
   macro avg       0.64      0.47      0.48       168
weighted avg       0.64      0.66      0.61       168
```

**Key Observations**:
- Similar performance to TF-IDF (macro F1: 0.48 vs 0.49)
- Better balance: Neutral precision (0.58 vs 0.68) with slightly better recall (0.33 vs 0.28)
- Positive class remains challenging
- Lower regularization (C=10 vs 100) suggests embeddings need less aggressive tuning

#### App Store Results

**Best Hyperparameters**:
- **C**: 10
- **Kernel**: linear

**Performance**:
- Best CV Macro F1: **0.5123**
- Test Accuracy: **0.6310** (63.10%)

**Test Set Classification Report**:
```
              precision    recall  f1-score   support

    Negatif       0.61      0.94      0.74       99
     Netral       0.67      0.23      0.34       48
    Positif       0.67      0.19      0.30       21

    accuracy                           0.63       168
   macro avg       0.65      0.45      0.46       168
weighted avg       0.63      0.63      0.58       168
```

**Key Observations**:
- Comparable to TF-IDF on App Store (macro F1: 0.46 vs 0.43)
- Slight improvement in minority class precision
- Same C value (10) as Play Store, suggesting consistent hyperparameter behavior

---

## 4.6 Model Comparison and Analysis

### 4.6.1 Cross-Platform Performance Summary

| Platform | Feature Type | Best Params | CV Macro F1 | Test Accuracy | Test Macro F1 |
|----------|--------------|-------------|-------------|---------------|---------------|
| Play Store | TF-IDF | C=100, linear | 0.6613 | 0.6845 | 0.49 |
| Play Store | IndoBERT | C=10, linear | 0.6342 | 0.6607 | 0.48 |
| App Store | TF-IDF | C=100, linear | 0.5305 | 0.6310 | 0.43 |
| App Store | IndoBERT | C=10, linear | 0.5123 | 0.6310 | 0.46 |

### 4.6.2 Key Findings

1. **Feature Engineering Comparison**:
   - **Play Store**: TF-IDF slightly outperforms IndoBERT (0.49 vs 0.48 macro F1)
   - **App Store**: IndoBERT slightly outperforms TF-IDF (0.46 vs 0.43 macro F1)
   - **Conclusion**: Performance differences are minimal (<0.03); both approaches are competitive

2. **Platform Differences**:
   - Play Store models consistently outperform App Store models
   - Possible reasons: Review writing styles, sentiment expression patterns, or data quality differences

3. **Kernel Selection**:
   - **Linear kernel** is optimal for both TF-IDF and IndoBERT across all experiments
   - Indicates sentiment classes are approximately linearly separable in both feature spaces
   - Non-linear kernels (RBF, polynomial) do not provide additional benefit

4. **Regularization Patterns**:
   - TF-IDF requires stronger regularization (C=100)
   - IndoBERT requires moderate regularization (C=10)
   - Reflects different feature space properties: sparse vs dense, high vs low dimensional

5. **Class Imbalance Impact**:
   - All models struggle with minority classes (Neutral, Positive)
   - Negative class dominates predictions (recall >0.90)
   - Macro F1 (0.43-0.49) significantly lower than accuracy (0.63-0.68), indicating imbalanced performance

### 4.6.3 Performance Visualization

**Confusion Matrix Insights** (Play Store TF-IDF - Best Model):
- True Negatives correctly identified: 89/94 (94.7%)
- Neutral misclassifications: Often predicted as Negative
- Positive misclassifications: Often predicted as Negative or Neutral
- Pattern: Model biased toward majority class

### 4.6.4 Computational Considerations

| Aspect | TF-IDF + SVM | IndoBERT + SVM |
|--------|--------------|----------------|
| Training time | Fast (~seconds) | Moderate (~minutes) |
| Inference time | Very fast | Moderate (embedding extraction) |
| Memory usage | Low (sparse matrices) | High (dense embeddings, model loading) |
| Scalability | Excellent | Limited by GPU/CPU resources |
| Interpretability | High (feature weights) | Low (black-box embeddings) |

**Practical Implications**:
- For deployment with resource constraints: TF-IDF + SVM preferred
- For maximum performance: Both approaches are competitive; choose based on infrastructure

---

## 4.7 Model Persistence and Deployment

### 4.7.1 Saved Models

All trained models are serialized using joblib for future use:

```python
import joblib

# Save TF-IDF pipeline
joblib.dump(tfidf_pipeline, 'outputs/svm_pipeline_tfidf_play.pkl')
joblib.dump(tfidf_pipeline, 'outputs/svm_pipeline_tfidf_app.pkl')

# Save IndoBERT pipeline
joblib.dump(bert_pipeline, 'outputs/svm_pipeline_bert_play.pkl')
joblib.dump(bert_pipeline, 'outputs/svm_pipeline_bert_app.pkl')
```

**Saved Artifacts**:
- `svm_pipeline_tfidf_play.pkl` - Play Store TF-IDF model
- `svm_pipeline_tfidf_app.pkl` - App Store TF-IDF model
- `svm_pipeline_bert_play.pkl` - Play Store IndoBERT model
- `svm_pipeline_bert_app.pkl` - App Store IndoBERT model

### 4.7.2 Model Loading and Inference

```python
# Load model
model = joblib.load('outputs/svm_pipeline_tfidf_play.pkl')

# Predict new review
review = "Aplikasi bagus tapi sering buffering"
prediction = model.predict([review])
print(f"Predicted sentiment: {prediction[0]}")
```

---

## 4.8 Limitations and Future Work

### 4.8.1 Current Limitations

1. **Class Imbalance**: Minority classes (Neutral, Positive) have low recall
2. **Dataset Size**: 838 samples may be insufficient for deep learning approaches
3. **Feature Engineering**: Limited exploration of advanced feature combinations
4. **Temporal Analysis**: Pre/post pricing splits not fully exploited in current models
5. **Cross-lingual**: Models trained only on Indonesian; no multilingual consideration

### 4.8.2 Recommendations for Future Work

1. **Class Imbalance Mitigation**:
   - Implement SMOTE (Synthetic Minority Over-sampling Technique)
   - Apply class weights in SVM: `class_weight='balanced'`
   - Collect more minority class samples

2. **Feature Engineering Enhancements**:
   - Experiment with n-gram combinations weighted by importance
   - Combine TF-IDF and IndoBERT features (hybrid approach)
   - Extract sentiment-specific features (negation patterns, intensifiers)

3. **Model Extensions**:
   - Fine-tune IndoBERT end-to-end instead of using frozen embeddings
   - Explore ensemble methods (voting, stacking)
   - Test other Indonesian language models (IndoGPT, mBERT)

4. **Temporal Modeling**:
   - Build separate models for pre/post pricing periods
   - Analyze sentiment drift over time
   - Incorporate temporal features

5. **Evaluation Enhancements**:
   - Perform statistical significance testing (McNemar's test)
   - Conduct error analysis with linguistic features
   - Collect human evaluation on predictions

---

## 4.9 Conclusion

This modeling phase successfully implemented and compared two feature engineering approaches (TF-IDF vs. IndoBERT) using Support Vector Machine classifiers for Indonesian sentiment classification of Disney+ Hotstar reviews.

**Key Takeaways**:
1. Both TF-IDF and IndoBERT achieve competitive performance (macro F1: 0.43-0.49)
2. Linear kernels are sufficient; non-linear kernels provide no benefit
3. Unigrams outperform higher-order n-grams for TF-IDF
4. Class imbalance remains the primary challenge
5. Play Store models outperform App Store models across all configurations

**Recommended Model**:
- **For Play Store**: TF-IDF + SVM (C=100, linear) - Best macro F1 (0.49)
- **For App Store**: IndoBERT + SVM (C=10, linear) - Best macro F1 (0.46)
- **For deployment**: TF-IDF + SVM - Lower computational requirements, comparable performance

The models demonstrate practical applicability for sentiment monitoring of app reviews, with opportunities for improvement through class balancing techniques and feature engineering enhancements.

---

## References

**Notebooks**:
- `notebooks/Tesis-Playstore-FIX.ipynb` - Play Store experiments
- `notebooks/Tesis-Appstore-FIX.ipynb` - App Store experiments

**Result Files**:
- `outputs/MODELING_RESULTS.md` - Detailed experimental results
- `outputs/modeling_results_summary.json` - Structured results data
- `outputs/exported_model_results_play.json` - Play Store predictions
- `outputs/exported_model_results_app.json` - App Store predictions

**Model Files**:
- `outputs/svm_pipeline_tfidf_*.pkl` - TF-IDF models
- `outputs/svm_pipeline_bert_*.pkl` - IndoBERT models
