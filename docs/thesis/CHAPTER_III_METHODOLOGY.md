# CHAPTER III: RESEARCH METHODOLOGY

## 3.1 Introduction

This chapter describes the research methodology employed in conducting sentiment analysis of Disney+ Hotstar Indonesian user reviews collected from the Apple App Store and Google Play Store. The study follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** framework, a widely-adopted industry standard for data science projects that provides a structured approach to planning and executing data mining tasks.

CRISP-DM consists of six iterative phases: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment. This methodology ensures systematic progression from problem definition through model development to practical implementation, while allowing flexibility to revisit earlier phases as insights emerge.

### 3.1.1 Research Framework Overview

The research is structured according to CRISP-DM phases, with specific adaptations for sentiment analysis:

1. **Business Understanding**: Define research objectives, success criteria, and stakeholder requirements for sentiment analysis
2. **Data Understanding**: Explore and characterize review datasets from both app stores
3. **Data Preparation**: Clean, preprocess, and transform raw Indonesian text data
4. **Modeling**: Develop and compare TF-IDF and IndoBERT-based classification models
5. **Evaluation**: Assess model performance using multiple metrics and validation approaches
6. **Deployment**: Implement dashboard for stakeholder consumption and monitoring

This methodology provides a comprehensive framework for addressing the research questions while ensuring reproducibility and practical applicability.

---

## 3.2 CRISP-DM Phase 1: Business Understanding

### 3.2.1 Research Background

Disney+ Hotstar, launched in Indonesia in September 2020, faces increasing competition in the streaming market. Understanding user sentiment expressed in app store reviews is critical for:
- Identifying service quality issues
- Prioritizing feature development
- Monitoring competitive positioning
- Improving user retention strategies

### 3.2.2 Research Objectives

**Primary Objective**: Develop an automated sentiment classification system for Indonesian-language Disney+ Hotstar reviews that can accurately categorize user opinions into Positive, Neutral, and Negative sentiment.

**Specific Objectives**:
1. Collect and preprocess Indonesian user reviews from App Store and Play Store (2020-2025)
2. Compare traditional TF-IDF features versus modern IndoBERT embeddings for sentiment classification
3. Develop Support Vector Machine (SVM) classifiers optimized for Indonesian text
4. Evaluate cross-platform sentiment patterns and differences
5. Implement a real-time dashboard for stakeholder monitoring

### 3.2.3 Success Criteria

**Success Criteria Categories**:

**1. Model Performance Metrics**:
- Classification accuracy 
- Macro F1-score for balanced class evaluation
- Processing time efficiency

**2. Business Value Metrics**:
- Model interpretability
- Cross-platform consistency
- Deployment feasibility

### 3.2.4 Data Mining Goals

Transform the business objectives into technical data mining goals:

1. **Classification Task**: Multi-class sentiment classification (Positive, Neutral, Negative)
2. **Feature Engineering**: Compare sparse TF-IDF vectors vs. dense IndoBERT embeddings
3. **Model Selection**: Optimize SVM hyperparameters for Indonesian text
4. **Evaluation Strategy**: Stratified cross-validation with hold-out test set
5. **Deployment**: Interactive Streamlit dashboard with real-time predictions

---

## 3.3 CRISP-DM Phase 2: Data Understanding

### 3.3.1 Data Collection

**Data Sources**: Two primary platforms
- **Apple App Store**: iOS user reviews
- **Google Play Store**: Android user reviews

**Target Collection Period**: 2020 - 2025

**Collection Method**: Web scraping using `google-play-scraper` and `app-store-scraper` libraries
- Python libraries for automated review extraction
- Retrieval of reviews with metadata (username, score, content, timestamp, helpful votes)

**Collected Attributes**:
| Attribute | Description | Data Type |
|-----------|-------------|-----------|
| `userName` | Reviewer username | String |
| `score` | Numerical rating (1-5 stars) | Integer |
| `content` | Review text | String (Indonesian) |
| `at` | Review timestamp | Datetime |
| `thumbsUpCount` | Helpful votes | Integer |
| `reviewId` | Unique identifier | String |

### 3.3.2 Initial Dataset Characteristics

**Temporal Scope Rationale**:
This study will divide the dataset into two distinct time periods to analyze sentiment evolution in relation to a critical business event:

- **Period 1 (2020-2022)**: Pre-Price Increase Era
  - September 2020: Disney+ Hotstar publicly launched in Indonesia
  - Represents baseline sentiment during initial market entry and growth phase
  - Pricing remained stable throughout this period

- **Period 2 (2023-2025)**: Post-Price Increase Era
  - 2023: Disney+ Hotstar implemented subscription price increase
  - Documented subscriber decline following price adjustment
  - Expected to capture sentiment potentially influenced by perceived value proposition changes
  - Data collection planned through April 2025

**Dataset Collection Strategy**:
To ensure fair temporal comparison, the dataset will maintain equal sample sizes per period:

**App Store Dataset (Planned)**:
- Target total reviews: 838 samples
- Period 1 (2020-2022): 419 reviews
- Period 2 (2023-2025): 419 reviews
- Output file: `combined_reviews_app.csv`
- Platform context: 4.8/5.0 average rating, 75.4K total reviews (as of March 1, 2025)

**Play Store Dataset (Planned)**:
- Target total reviews: 838 samples
- Period 1 (2020-2022): 419 reviews
- Period 2 (2023-2025): 419 reviews
- Output file: `combined_reviews_play.csv`
- Platform context: 2.0/5.0 average rating, 117K total reviews (as of March 1, 2025)

**Research Motivation for Temporal Analysis**:
The 2023 price increase serves as a **natural experiment** for sentiment analysis, enabling investigation of:
1. Whether user sentiment shifts between pre- and post-price increase periods
2. How pricing decisions impact user perception expressed in reviews
3. Whether negative feedback correlates temporally with the price adjustment
4. Cross-platform differences in temporal sentiment patterns

**Data Collection Timeline**:
The actual data scraping will be performed in April 2025, retrieving historical reviews spanning the entire period from September 2020 to April 2025.

### 3.3.3 Exploratory Data Analysis Plan

**Rating Distribution Analysis**:
Initial exploration will examine star ratings (1-5) to understand:
- Distribution of ratings across both platforms
- Platform-specific rating patterns

**Review Length Analysis**:
Planned analysis of review characteristics:
- Average and median review length
- Distribution of short vs. long reviews
- Presence of emoji-only or minimal text reviews

**Language Characteristics**:
Expected characteristics based on preliminary observation:
- Primary language: Indonesian (Bahasa Indonesia)
- Informal language patterns (social media style)
- Mixed Indonesian-English terms
- Colloquialisms and slang expressions

**Anticipated Data Quality Issues**:
Expected challenges to address during preprocessing:
1. Missing or null content fields
2. Non-Indonesian text (English reviews requiring filtering)
3. Spam and repetitive reviews
4. Emoji-only reviews
5. Special characters and formatting inconsistencies

### 3.3.4 Initial Sentiment Labeling Approach

**Labeling Strategy**: Lexicon-based approach using InSet lexicon
- **InSet Lexicon**: Indonesian sentiment lexicon containing:
  - Positive words: 3,609 terms
  - Negative words: 6,609 terms
  - Total coverage: 10,218 sentiment-bearing words

**Labeling Algorithm**:
```
For each review:
    1. Tokenize review text
    2. Count positive terms (pos_count)
    3. Count negative terms (neg_count)
    4. Calculate sentiment score:
       - If pos_count > neg_count: Label = "Positif"
       - If neg_count > pos_count: Label = "Negatif"
       - If pos_count == neg_count: Label = "Netral"
```

This lexicon-based labeling will serve as the ground truth for supervised machine learning model training.

**Expected Sentiment Distribution**:

| Platform | Positif | Netral | Negatif | Total |
|----------|---------|--------|---------|-------|
| App Store | 135 (16.11%) | 147 (17.54%) | 556 (66.35%) | 838 |
| Play Store | 59 (7.04%) | 90 (10.74%) | 689 (82.22%) | 838 |

**Key Observations**:
- Strong class imbalance favoring negative sentiment
- Play Store more negative than App Store (+15.87%)
- Limited positive sentiment representation (especially Play Store: 7.04%)

---

## 3.4 CRISP-DM Phase 3: Data Preparation

### 3.4.1 Data Cleaning Pipeline

The data preparation phase follows a systematic multi-stage pipeline to transform raw Indonesian text into clean, analysis-ready format.

#### Stage 1: Initial Cleaning

**Translation to Indonesian**:
- Method: Google Translate API via `googletrans` library
- Purpose: Standardize all reviews to Indonesian language
- Implementation: `translate_to_indonesian()` function

**Case Normalization**:
- Convert all text to lowercase
- Ensures consistency in pattern matching and reduces vocabulary size

**Special Character Removal**:
- Remove URLs, email addresses, HTML tags
- Strip special characters except spaces
- Preserve Indonesian diacritics where applicable

**Whitespace Normalization**:
- Remove extra spaces, tabs, newlines
- Standardize spacing between words

#### Stage 2: Tokenization

**Method**: Word-level tokenization using NLTK
- Split text into individual tokens (words)
- Preserve Indonesian word boundaries
- Handle punctuation separation

**Implementation**:
```python
from nltk.tokenize import word_tokenize
tokens = word_tokenize(cleaned_text, language='indonesian')
```

#### Stage 3: Stopword Removal

**Stopword List**: Indonesian stopwords from NLTK and Sastrawi
- Total stopwords: 758 words
- Includes common function words: "yang", "dan", "di", "ke", "dari"

**Process**:
- Filter out stopwords from token list
- Retain sentiment-bearing content words
- Preserve negation words (e.g., "tidak", "bukan")

**Impact of Stopword Removal**:
- **Empty String Issue**: Reviews containing ONLY stopwords become empty after removal
- **App Store**: 8 empty strings (0.95%)
- **Play Store**: 43 empty strings (5.13%)
- **Solution**: Filter empty strings before modeling (see Section 3.5.2)

**Example**:
- Original: "Aplikasi ini sangat bagus dan menyenangkan"
- After stopword removal: "aplikasi sangat bagus menyenangkan"

#### Stage 4: Stemming

**Stemmer**: Sastrawi (Indonesian stemmer)
- Reduces words to root form (base word)
- Handles Indonesian morphology (prefixes, suffixes, infixes)

**Implementation**:
```python
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
stemmer = StemmerFactory().create_stemmer()
stemmed_text = stemmer.stem(text)
```

**Examples**:
- "menyenangkan" → "senang"
- "berlangganan" → "langgan"
- "menggunakan" → "guna"

**Purpose**:
- Reduce vocabulary size
- Group morphological variants
- Improve feature generalization

#### Stage 5: Final Text Reconstruction

**Creation of `ulasan_bersih` Column**:
- Join stemmed tokens back into single string
- Space-separated format
- Represents final preprocessed text for modeling

### 3.4.2 Feature Engineering

Two distinct feature engineering approaches are implemented for comparison:

#### Approach 1: TF-IDF Vectorization

**Method**: Term Frequency-Inverse Document Frequency
- **TF (Term Frequency)**: How often a term appears in a document
- **IDF (Inverse Document Frequency)**: How rare/common a term is across all documents
- **TF-IDF Score**: TF × IDF (balances frequency with distinctiveness)

**Configuration**:
- **Max features**: 5,000 most frequent terms
- **N-gram range**: Unigrams and bigrams (1,2)
- **Min document frequency**: 2 (term must appear in at least 2 documents)
- **Max document frequency**: 0.95 (exclude terms appearing in >95% of documents)

**Output**:
- Sparse matrix of shape (n_samples, n_features)
- App Store: (830, 1688) for training
- Play Store: (795, 1368) for training

**Advantages**:
- Computationally efficient
- Interpretable features (actual words)
- Works well with linear classifiers
- Captures term importance

**Limitations**:
- Ignores word order and context
- Sparse representation
- No semantic understanding

#### Approach 2: IndoBERT Embeddings

**Model**: IndoBERT-base-p1
- Pre-trained Indonesian BERT model
- Developed by IndoNLP team
- 12 transformer layers, 768 hidden units
- Trained on Indonesian Wikipedia, news, and social media text

**Embedding Process**:
1. Tokenize text using IndoBERT tokenizer
2. Feed through BERT model
3. Extract [CLS] token representation (768-dimensional vector)
4. Use as sentence-level embedding

**Configuration**:
- **Max sequence length**: 512 tokens
- **Batch size**: 16 (for memory efficiency)
- **Embedding dimension**: 768
- **Caching**: Save embeddings to disk to avoid recomputation

**Output**:
- Dense matrix of shape (n_samples, 768)
- Each review represented as 768-dimensional vector
- Captures contextual semantic information

**Advantages**:
- Contextual understanding
- Semantic similarity capture
- Pre-trained on Indonesian text
- Dense representation

**Limitations**:
- Computationally expensive
- Less interpretable
- Requires more memory
- Fixed embedding size

### 3.4.3 Data Splitting Strategy

**Split Ratio**: 80% training, 20% testing

**Method**: Stratified train-test split
- Will preserve class distribution in both sets
- Random state = 42 (for reproducibility)
- Ensures representative samples in test set

**Filtering Empty Strings**:
Critical preprocessing step to be applied BEFORE splitting:
```python
# Filter empty strings
df_filtered = df.dropna(subset=['ulasan_bersih'])
df_filtered = df_filtered[df_filtered['ulasan_bersih'].str.strip() != '']

# Then split
X_train, X_test, y_train, y_test = train_test_split(
    df_filtered['ulasan_bersih'],
    df_filtered['sentimen_multiclass'],
    test_size=0.2,
    random_state=42,
    stratify=df_filtered['sentimen_multiclass']
)
```

**Expected Dataset Sizes**:
Based on initial analysis, after filtering empty strings resulting from stopword removal:
- Approximately 95-97% of reviews will contain analyzable text
- Training set: ~80% of usable samples
- Test set: ~20% of usable samples

**Importance of Empty String Filtering**:
- Empty strings cannot be vectorized by TF-IDF (no terms to extract)
- Empty strings cannot be embedded by BERT (no content to encode)
- Ensures both models use identical, valid test sets
- Prevents misleading evaluation metrics from invalid inputs

### 3.4.4 Data Validation Plan

**Quality Checks to Perform**:
1. Verify no missing values in `ulasan_bersih` after filtering
2. Confirm all text converted to lowercase
3. Ensure no special characters remaining
4. Validate all reviews in Indonesian (post-translation/filtering)
5. Verify stratified split maintains class distribution
6. Ensure no data leakage between train and test sets

**Planned Output Artifacts**:
- `progres_preprocessing_app.csv` - App Store preprocessed data
- `progres_preprocessing_play.csv` - Play Store preprocessed data
- `train_indobert.npy` - Cached IndoBERT training embeddings (for efficiency)
- `test_indobert.npy` - Cached IndoBERT test embeddings (for efficiency)

---

## 3.5 CRISP-DM Phase 4: Modeling

### 3.5.1 Modeling Approach

The modeling phase implements a **controlled comparison** between two feature engineering strategies using Support Vector Machine (SVM) as the sole classifier. This design allows performance differences to be attributed to feature representation rather than algorithmic variation.

**Two Pipelines**:
1. **TF-IDF + SVM**: Traditional bag-of-words approach
2. **IndoBERT + SVM**: Contextual embedding approach

### 3.5.2 Support Vector Machine (SVM) Classifier

**Algorithm Selection Rationale**:

**Why SVM?**
- Effective for high-dimensional data (TF-IDF creates thousands of features)
- Works well with both sparse (TF-IDF) and dense (BERT) representations
- Robust to overfitting with proper regularization
- Kernel flexibility enables non-linear decision boundaries
- Established strong performance in text classification

**SVM Fundamentals**:
- Finds optimal hyperplane that maximizes margin between classes
- Uses kernel trick to handle non-linearly separable data
- Regularization parameter C controls bias-variance tradeoff

**Implementation**:
```python
from sklearn.svm import SVC

svm_model = SVC(
    kernel='linear',  # or 'rbf'
    C=1.0,            # regularization
    random_state=42,
    class_weight='balanced'  # handle class imbalance
)
```

### 3.5.3 Hyperparameter Tuning

**Optimization Method**: Grid Search with Cross-Validation
- **Cross-validation folds**: 10-fold stratified CV
- **Scoring metric**: F1-score (macro average)
- **Search space**: Systematic grid of parameter combinations

**TF-IDF + SVM Hyperparameters**:

Grid search space:
```python
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__kernel': ['linear', 'rbf', 'poly']
}
```

**Parameters tested**:
- **C (Regularization)**: Controls model complexity
  - Low C (0.1): Simpler model, larger margin, more regularization
  - High C (100): Complex model, smaller margin, less regularization
  
- **Kernel**: Decision boundary shape
  - **Linear**: Straight line/hyperplane (for linearly separable data)
  - **RBF (Radial Basis Function)**: Non-linear, Gaussian-shaped boundaries
  - **Polynomial**: Polynomial-shaped boundaries

**Best Parameters Found**:
- **App Store TF-IDF**: C=100, kernel='linear'
- **Play Store TF-IDF**: C=100, kernel='linear'

**IndoBERT + SVM Hyperparameters**:

Grid search space:
```python
param_grid = {
    'C': [0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf', 'poly']
}
```

**Best Parameters Found**:
- **App Store IndoBERT**: C=0.01, kernel='linear'
- **Play Store IndoBERT**: C=0.01, kernel='linear'

**Key Insight**: Linear kernels consistently outperform non-linear kernels, suggesting that sentiment classes are approximately linearly separable in both TF-IDF and IndoBERT feature spaces.

### 3.5.4 Model Training Process

**Pipeline Architecture**:

**TF-IDF Pipeline**:
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )),
    ('svm', SVC(
        C=100,
        kernel='linear',
        class_weight='balanced',
        random_state=42
    ))
])

# Train
tfidf_pipeline.fit(X_train, y_train)
```

**IndoBERT Pipeline**:
```python
# Step 1: Generate embeddings (cached)
train_embeddings = get_bert_embeddings(X_train, 'train_indobert.npy')
test_embeddings = get_bert_embeddings(X_test, 'test_indobert.npy')

# Step 2: Train SVM on embeddings
bert_svm = SVC(
    C=0.01,
    kernel='linear',
    class_weight='balanced',
    random_state=42
)
bert_svm.fit(train_embeddings, y_train)
```

**Training Details**:
- **Class weighting**: `balanced` (inversely proportional to class frequencies)
- **Random state**: 42 (ensures reproducibility)
- **Convergence**: Default scikit-learn settings
- **Training time**: 
  - TF-IDF: ~30 seconds per model
  - IndoBERT: ~10-15 minutes per model (including embedding generation)

### 3.5.5 Model Persistence

**Saved Models** (using Python pickle):
- `svm_pipeline_tfidf_app.pkl` - App Store TF-IDF model
- `svm_pipeline_bert_app.pkl` - App Store IndoBERT model
- `svm_pipeline_tfidf_play.pkl` - Play Store TF-IDF model
- `svm_pipeline_bert_play.pkl` - Play Store IndoBERT model

**Model Files Include**:
- Trained SVM classifier
- Feature extraction pipeline (TF-IDF vectorizer)
- Preprocessing parameters
- Class labels and mappings

---

## 3.6 CRISP-DM Phase 5: Evaluation

### 3.6.1 Evaluation Strategy

**Evaluation Objectives**:
1. Quantify model performance using multiple metrics
2. Compare TF-IDF vs. IndoBERT feature representations
3. Analyze cross-platform differences (App Store vs. Play Store)
4. Identify class-specific strengths and weaknesses
5. Assess prediction bias and distribution alignment

### 3.6.2 Evaluation Metrics

**Primary Metrics**:

**1. Accuracy**:
- Formula: (TP + TN) / (TP + TN + FP + FN)
- Measures overall correctness
- Limitation: Can be misleading with class imbalance

**2. Macro F1-Score** (Primary metric):
- Formula: Average of per-class F1-scores
- Treats all classes equally (unweighted)
- Best metric for imbalanced datasets
- Range: 0 to 1 (higher is better)

**3. Weighted F1-Score**:
- Formula: Weighted average of per-class F1-scores (by support)
- Accounts for class imbalance
- More aligned with accuracy

**Per-Class Metrics**:

**4. Precision**:
- Formula: TP / (TP + FP)
- "When model predicts positive, how often is it correct?"
- Measures false positive rate

**5. Recall (Sensitivity)**:
- Formula: TP / (TP + FN)
- "Of all actual positives, how many did model find?"
- Measures false negative rate

**6. F1-Score**:
- Formula: 2 × (Precision × Recall) / (Precision + Recall)
- Harmonic mean of precision and recall
- Balances both metrics

**7. Confusion Matrix**:
- Shows actual vs. predicted class distribution
- Reveals confusion patterns between classes
- Useful for identifying systematic errors

### 3.6.3 Cross-Validation

**Method**: 10-fold Stratified Cross-Validation
- Used during hyperparameter tuning
- Ensures robust parameter selection
- Stratification maintains class distribution in each fold

**Process**:
```python
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,  # 10-fold CV
    scoring='f1_macro',  # Optimization metric
    n_jobs=-1,  # Parallel processing
    verbose=1
)
```

### 3.6.4 Validation Approach

**Hold-out Test Set Evaluation**:
- **Never used during training or hyperparameter tuning**
- Provides unbiased performance estimate
- Used for all reported results in Chapter 4

**Evaluation Process**:
1. Train model on training set (664/636 samples)
2. Tune hyperparameters using 10-fold CV on training set
3. Select best hyperparameters
4. Evaluate final model on hold-out test set (166/159 samples)
5. Report test set metrics (no further tuning allowed)

**Preventing Overfitting**:
- Stratified splitting ensures representative test set
- Cross-validation for hyperparameter selection
- Test set remains completely unseen
- No iterative adjustment based on test performance

### 3.6.5 Comparative Analysis Framework

**Comparisons Conducted**:

**1. Feature Engineering Comparison**:
- TF-IDF vs. IndoBERT (same platform, same model)
- Metric: Macro F1-score (primary), Accuracy (secondary)

**2. Cross-Platform Comparison**:
- App Store vs. Play Store (same feature method, same model)
- Identifies platform-specific challenges

**3. Per-Class Analysis**:
- Negatif vs. Netral vs. Positif performance
- Identifies which sentiment classes are harder to predict

**4. Prediction Bias Analysis**:
- Compare predicted distribution to ground truth distribution
- Measure over-prediction or under-prediction of each class

---

## 3.7 CRISP-DM Phase 6: Deployment

### 3.7.1 Deployment Objectives

**Primary Goal**: Make sentiment analysis models accessible to non-technical stakeholders through interactive dashboard

**Deployment Requirements**:
1. User-friendly interface (no coding required)
2. Real-time predictions on new reviews
3. Visualization of historical sentiment trends
4. Platform selection (App Store vs. Play Store)
5. Model selection (TF-IDF vs. IndoBERT)

### 3.7.2 Deployment Architecture

**Technology Stack**:
- **Framework**: Streamlit (Python web framework)
- **Backend**: Python 3.x
- **Models**: Pickle-serialized scikit-learn models
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Hosting**: Local server (can be deployed to cloud)

**Dashboard Components**:

**1. Model Selection Panel**:
- Platform selector: App Store / Play Store
- Feature method selector: TF-IDF / IndoBERT
- Dynamically loads appropriate model

**2. Prediction Interface**:
- Text input box for new reviews
- Real-time sentiment prediction
- Confidence scores display
- Preprocessing visualization

**3. Historical Analysis**:
- Sentiment distribution charts (pie charts, bar charts)
- Time series trends (if timestamp available)
- Rating vs. sentiment correlation analysis

**4. Model Performance Metrics**:
- Confusion matrices
- Classification reports
- Per-class metrics display

**5. Word Cloud Visualization**:
- Separate clouds for Positive, Neutral, Negative reviews
- Identifies dominant keywords per sentiment

### 3.7.3 Deployment Pipeline

**Step 1: Model Loading**:
```python
import pickle

# Load selected model
with open(f'models/svm_pipeline_{method}_{platform}.pkl', 'rb') as f:
    model = pickle.load(f)
```

**Step 2: Real-time Prediction**:
```python
def predict_sentiment(review_text):
    # Preprocess
    cleaned_text = preprocess_text(review_text)
    
    # Predict
    prediction = model.predict([cleaned_text])
    probabilities = model.predict_proba([cleaned_text])
    
    return prediction[0], probabilities[0]
```

**Step 3: Visualization Update**:
- Dashboard updates in real-time as user inputs change
- Interactive charts respond to filters and selections

### 3.7.4 User Workflow

**Typical User Journey**:
1. **Launch Dashboard**: `streamlit run dashboard_app.py`
2. **Select Configuration**: Choose platform and feature method
3. **Enter Review**: Type or paste Indonesian review text
4. **View Prediction**: See sentiment classification and confidence
5. **Explore Analytics**: View historical patterns and model performance

**Use Cases**:
- **Customer Support**: Prioritize negative sentiment reviews for response
- **Product Management**: Identify feature requests and pain points
- **Marketing**: Track sentiment trends over time
- **Competitive Analysis**: Compare sentiment across platforms

### 3.7.5 Monitoring and Maintenance

**Performance Monitoring**:
- Track prediction distribution over time
- Identify drift in sentiment patterns
- Monitor model confidence scores

**Model Updates**:
- Retrain models quarterly with new review data
- Update lexicon with emerging Indonesian slang
- Refresh IndoBERT embeddings if model updates

**Error Handling**:
- Input validation (reject non-Indonesian text)
- Graceful handling of edge cases (very short/long reviews)
- Logging of prediction errors for analysis

---

## 3.8 Research Tools and Technologies

### 3.8.1 Development Environment

**Programming Language**: Python 3.10+

**Development Tools**:
- **IDE**: Visual Studio Code / Google Colab
- **Version Control**: Git / GitHub
- **Environment Management**: Conda / pip

### 3.8.2 Key Libraries

**Data Processing**:
- `pandas` 2.0+ - Data manipulation and analysis
- `numpy` 1.24+ - Numerical computing

**Text Processing**:
- `nltk` 3.8+ - Tokenization, stopwords
- `Sastrawi` 1.0+ - Indonesian stemmer
- `googletrans` 4.0+ - Translation

**Machine Learning**:
- `scikit-learn` 1.3+ - SVM, TF-IDF, evaluation metrics
- `transformers` 4.30+ - IndoBERT model loading
- `torch` 2.0+ - PyTorch backend for BERT

**Visualization**:
- `matplotlib` 3.7+ - Static plots
- `seaborn` 0.12+ - Statistical visualizations
- `plotly` 5.14+ - Interactive charts
- `wordcloud` 1.9+ - Word cloud generation

**Deployment**:
- `streamlit` 1.24+ - Dashboard framework
- `pickle` - Model serialization

### 3.8.3 Hardware Specifications

**Development Machine**:
- Processor: Intel Core i5/i7 or AMD Ryzen 5/7
- RAM: 16GB minimum (32GB recommended for BERT)
- Storage: 50GB free space (for models and datasets)
- GPU: Optional but recommended for IndoBERT (NVIDIA with CUDA support)

**Cloud Resources** (if applicable):
- Google Colab Pro (for GPU-accelerated BERT embedding generation)
- RAM: 25GB+ for large batch processing

---

## 3.9 Ethical Considerations

### 3.9.1 Data Privacy

**User Anonymity**:
- Review usernames anonymized in final dataset
- No personally identifiable information (PII) collected
- Reviews are publicly available data (app store reviews)

**Data Usage**:
- Academic research purposes only
- No commercial exploitation
- Compliance with app store terms of service

### 3.9.2 Bias and Fairness

**Potential Biases**:
- Class imbalance (negative reviews overrepresented)
- Platform bias (Android vs. iOS user demographics)
- Lexicon bias (InSet lexicon may not cover all slang)

**Mitigation Strategies**:
- Balanced class weighting in SVM
- Separate models per platform
- Macro F1-score prioritizes minority classes
- Transparent reporting of limitations

### 3.9.3 Research Integrity

**Reproducibility**:
- Fixed random seeds (random_state=42)
- Documented preprocessing steps
- Saved models and data artifacts
- Clear methodology description

**Transparency**:
- Report both successful and unsuccessful experiments
- Acknowledge limitations
- Provide code and documentation

---

## 3.10 Chapter Summary

This chapter has described the comprehensive research methodology following the CRISP-DM framework:

1. **Business Understanding**: Defined sentiment analysis objectives for Disney+ Hotstar reviews
2. **Data Understanding**: Collected and explored 838 reviews per platform (2020-2025)
3. **Data Preparation**: Implemented systematic Indonesian text preprocessing pipeline
4. **Modeling**: Developed TF-IDF and IndoBERT-based SVM classifiers with hyperparameter tuning
5. **Evaluation**: Established multi-metric evaluation framework prioritizing macro F1-score
6. **Deployment**: Designed Streamlit dashboard for stakeholder access

**Key Methodological Contributions**:
- Controlled comparison framework (single classifier, two feature methods)
- Rigorous handling of empty string filtering (critical for Indonesian preprocessing)
- Stratified splitting with class balancing for imbalanced datasets
- Comprehensive cross-platform analysis (App Store vs. Play Store)

The next chapter (Chapter IV) presents the results obtained by applying this methodology, including model performance metrics, cross-platform comparisons, and detailed evaluation analysis.

---

**End of Chapter III**
