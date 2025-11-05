# Section IV: Results and Analysis

## 1. Data Processing Pipeline
### Raw Dataset → Sentiment Labels

1. **Data Collection**
   - App Store: 838 reviews
   - Play Store: 838 reviews
   - Collection Date: April 7th, 2025

2. **Text Preprocessing Pipeline**
   ```
   Raw Review → Translation → Case Folding → 
   Cleaning → Stopword Removal → Stemming → Clean Text
   ```

3. **Lexicon-Based Sentiment Labeling**
   - Indonesian sentiment lexicon
   - Rule-based scoring system
   - Automated labeling process

## 2. Sentiment Distribution Analysis

### Platform Comparison

| Sentiment | App Store | Play Store | Difference |
|-----------|-----------|------------|------------|
| Negatif   | 66.35%   | 82.22%    | +15.87%    |
| Netral    | 17.54%   | 10.74%    | -6.80%     |
| Positif   | 16.11%   | 7.04%     | -9.07%     |

**Key Findings:**
- Play Store shows higher negative sentiment (+15.87%)
- App Store has more balanced distribution
- Both platforms indicate user dissatisfaction majority

## 3. SVM Performance Comparison

### Feature Extraction Methods

#### A. TF-IDF + SVM Results

| Metric | App Store | Play Store |
|--------|-----------|------------|
| Accuracy | 66.87% | 73.21% |
| Macro F1 | 0.57 | 0.38 |
| Weighted F1 | 0.67 | 0.72 |

#### B. IndoBERT + SVM Results

| Metric | App Store | Play Store |
|--------|-----------|------------|
| Accuracy | 66.27% | 72.62% |
| Macro F1 | 0.47 | 0.33 |
| Weighted F1 | 0.64 | 0.71 |

**Key Findings:**
1. TF-IDF slightly outperforms IndoBERT embeddings
2. Play Store shows higher accuracy but lower F1 scores in both methods
3. App Store results more balanced across sentiment classes

### Best Configuration Details

#### A. TF-IDF + SVM
- **N-gram Range:** (1,1) [Unigrams]
- **Vocabulary Size:** 1,367 features
- **Hyperparameters:**
  - C = 100.0
  - kernel = 'linear'
  - class_weight = 'balanced'
- **CV F1-macro:** 0.6613

#### B. IndoBERT + SVM
- **Embedding Size:** 768 features (fixed)
- **Hyperparameters:**
  - C = 1.0
  - kernel = 'linear'
  - class_weight = 'balanced'
- **CV F1-macro:** 0.4921

### Overall Best Performance
- **Feature Extraction:** TF-IDF
- **Classifier:** SVM with RBF kernel
- **Best Platform:** Play Store (73.21% accuracy)
- **Most Balanced:** App Store (0.57 macro F1)