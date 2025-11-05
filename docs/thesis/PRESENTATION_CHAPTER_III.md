# 📊 PRESENTATION OUTLINE: CHAPTER III - METHODOLOGY
## **Compact 5-Slide Structure: Thesis Flow & CRISP-DM Framework**

**Context**: Research Methodology following CRISP-DM  
**Target**: 5 compact slides covering overall flow + framework  
**Time Allocation**: ~6-7 minutes  
**Approach**: Systematic explanation from problem to deployment

---

## **SLIDE 1: Thesis Research Flow Overview (60 seconds)**

**Title**: Research Flow: From Problem to Deployment

**Visual**: Three-panel flowchart showing MULAI → FENOMENA → PERMASALAHAN → SOLUSI PENELITIAN → DATA → PREPROCESSING → MODELING → HASIL → MANFAAT → SELESAI

**Content**:

```
🎯 SYSTEMATIC RESEARCH PROGRESSION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 PHASE 1: FENOMENA (Problem Context)
   🎬 OTT streaming growth in Indonesia & Southeast Asia
   ⭐ Rating disparity: App Store 4.8/5 vs Play Store 2.0/5
   💰 2023 Disney+ Hotstar price increase → subscriber decline

📍 PHASE 2: PERMASALAHAN (Research Problem)
   ❓ Incomplete ratings (missing aspects/inconsistent)
   🎯 2023 price increase impact on sentiment unclear

📍 PHASE 3: SOLUSI PENELITIAN (Solution)
   🤖 SVM sentiment analysis comparing:
      • TF-IDF (Traditional bag-of-words)
      • IndoBERT embeddings (Modern contextual)

📍 PHASE 4: HASIL (Results)
   📊 Raw data → Cleaned data → Trained models
   ✅ Sentiment distribution analysis
   ✅ TF-IDF vs IndoBERT performance comparison

📍 PHASE 5: MANFAAT (Impact)
   🎓 Academic: Indonesian NLP contribution
   💼 Business: Actionable insights for product decisions
   ⚙️ Practical: Interactive dashboard deployment
```

**Speaking Points** (60 sec):
> "Our research follows a systematic 5-phase flow. We identified the phenomenon—Disney+ Hotstar's rating disparity and 2023 price increase. This led to our research problem—incomplete ratings failing to capture nuanced sentiment. Our solution? Compare traditional TF-IDF versus modern IndoBERT for Indonesian sentiment classification using SVM. The results provide actionable insights for business decisions, and we deliver this through an interactive dashboard."

---

## **SLIDE 2: CRISP-DM Framework - The Foundation (90 seconds)**

**Title**: CRISP-DM: Industry-Standard Methodology for Data Mining

**Visual**: Circular CRISP-DM diagram with 6 interconnected phases

**Content**:

```
📐 6-PHASE ITERATIVE FRAMEWORK:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────┐
│ 1️⃣ BUSINESS UNDERSTANDING                          │
│    • Research Objectives: Automated sentiment        │
│      classification for Indonesian reviews          │
│    • Success Criteria: Model performance &          │
│      business value metrics                         │
│    • Primary Question: TF-IDF vs IndoBERT?          │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 2️⃣ DATA UNDERSTANDING                              │
│    • Sources: App Store + Play Store                │
│    • Total: 838 × 2 platforms = 1,676 reviews       │
│    • Periods: 2020-2022 (419) + 2023-2025 (419)     │
│    • Collection: April 2025 scraping                │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 3️⃣ DATA PREPARATION                                │
│    • 6-Stage Pipeline:                              │
│      1. Translation (Google Translate)              │
│      2. Cleaning (lowercase + noise removal)        │
│      3. Tokenization (NLTK)                         │
│      4. Stopword Removal (758 Indonesian stopwords) │
│      5. Stemming (Sastrawi)                         │
│      6. Final Text (`ulasan_bersih`)                │
│    • Labeling: InSet lexicon (10,218 terms)         │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 4️⃣ MODELING                                        │
│    • Feature Extraction:                            │
│      - TF-IDF: Max 5,000 features                   │
│        • N-grams tested: (1,1), (1,2), (1,3)        │
│      - IndoBERT: 768-dimensional embeddings         │
│    • Classifier: SVM (Linear kernel)                │
│    • Tuning: GridSearchCV (10-fold CV)              │
│      - TF-IDF: n-gram (1,1)-(1,3), C ∈ {0.001,      │
│        0.01, 0.1, 1, 10, 100}, kernel ∈ {linear,    │
│        rbf, poly}                                   │
│      - IndoBERT: C ∈ {0.001, 0.01, 0.1, 1, 10,      │
│        100}, kernel ∈ {linear, rbf, poly}           │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 5️⃣ EVALUATION                                      │
│    • Primary Metric: Macro F1-Score                 │
│      (handles 82% negative imbalance)               │
│    • Secondary Metric: Accuracy                     │
│    • Validation: Stratified 80:20 split             │
│    • Analysis: Cross-platform comparison            │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 6️⃣ DEPLOYMENT                                      │
│    • Platform: Streamlit dashboard                  │
│    • Features: Real-time prediction, visualizations │
│    • Users: Customer support, product managers      │
└─────────────────────────────────────────────────────┘

✅ WHY CRISP-DM?
• Industry-standard framework (proven methodology)
• Iterative approach (can revisit phases as needed)
• Ensures reproducibility and transparency
• Bridges business objectives to technical implementation
```

**Speaking Points** (90 sec):
> "CRISP-DM ensures systematic progression through six iterative phases. We start by understanding business needs—sentiment analysis for Disney+ Hotstar with macro F1 ≥ 0.50 as our success criterion. Data understanding involves collecting 1,676 reviews across two platforms and time periods split by the 2023 price increase. Data preparation transforms raw Indonesian text through a 6-stage preprocessing pipeline (translation, cleaning, tokenization, stopword removal, stemming, final text). Modeling compares TF-IDF versus IndoBERT features with SVM classifiers tuned via grid search—we test three n-gram settings for TF-IDF and multiple C and kernel parameters for both methods. Evaluation uses macro F1-score as the primary metric to handle severe class imbalance—82% negative on Play Store. Finally, deployment makes models accessible through an interactive Streamlit dashboard for real-time predictions."

---

## **SLIDE 3: Data Pipeline Flow - From Raw to Predictions (90 seconds)**

**Title**: End-to-End Data Pipeline

**Visual**: Flowchart showing DATA → PREPROCESSING → LABELING → FEATURE EXTRACTION → CLASSIFICATION → RESULTS

**Content**:

```
📊 COMPLETE DATA TRANSFORMATION PIPELINE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔹 INPUT: RAW DATA
   • App Store: 838 reviews (2020-2025)
   • Play Store: 838 reviews (2020-2025)
   • Temporal Split:
     - Period 1 (2020-2022): Pre-price increase - 419 each
     - Period 2 (2023-2025): Post-price increase - 419 each
   • Attributes: userName, score (1-5★), content, timestamp

🔹 STAGE 1: PREPROCESSING (6 Steps)
   Step 1: Translation → Indonesian (googletrans)
   Step 2: Cleaning → lowercase, strip url/punctuation/numbers, collapse spaces
   Step 3: Tokenization → Word tokens (NLTK)
   Step 4: Stopword Removal → Filter 758 Indonesian stopwords
      ⚠️ Creates empty strings (App: 8, Play: 43)
   Step 5: Stemming → Root form (Sastrawi)
      Example: "menyenangkan" → "senang"
   Step 6: Final Text → stored as `ulasan_bersih`

🔹 STAGE 2: SENTIMENT LABELING (Lexicon-Based)
   • Method: InSet dictionary (10,218 terms)
     - 3,609 positive words
     - 6,609 negative words
   • Algorithm:
     IF pos_count > neg_count → "Positif"
     IF neg_count > pos_count → "Negatif"
     IF pos_count == neg_count → "Netral"
   
   • Output Distribution:
     ┌──────────────┬─────────┬─────────┬──────────┐
     │ Platform     │ Positif │ Netral  │ Negatif  │
     ├──────────────┼─────────┼─────────┼──────────┤
     │ App Store    │ 16%     │ 18%     │ 66% ⚠️   │
     │ Play Store   │ 7%      │ 11%     │ 82% 🔴   │
     └──────────────┴─────────┴─────────┴──────────┘
     → Severe class imbalance justifies Macro F1 metric

🔹 STAGE 3: FEATURE EXTRACTION (2 Methods)
   
   ┌─────────────────────────────────────────────────┐
   │ Method 1: TF-IDF (Traditional)                  │
   ├─────────────────────────────────────────────────┤
   │ • Max features: 5,000                           │
   │ • N-gram tested: (1,1), (1,2), (1,3)            │
   │ • Output: Sparse matrix                         │
   │   - App Store: (830, 1688)                      │
   │   - Play Store: (795, 1368)                     │
   │ • Advantages:                                   │
   │   ✅ Efficient (~30 sec training)               │
   │   ✅ Interpretable (actual words)               │
   │ • Limitations:                                  │
   │   ❌ No word order/context                      │
   └─────────────────────────────────────────────────┘
   
   ┌─────────────────────────────────────────────────┐
   │ Method 2: IndoBERT (Modern)                     │
   ├─────────────────────────────────────────────────┤
   │ • Model: IndoBERT-base-p1                       │
   │ • Embedding dim: 768                            │
   │ • Output: Dense matrix (n_samples, 768)         │
   │ • Advantages:                                   │
   │   ✅ Contextual understanding                   │
   │   ✅ Semantic similarity                        │
   │ • Limitations:                                  │
   │   ❌ Expensive (~10-15 min training)            │
   │   ❌ Less interpretable                         │
   └─────────────────────────────────────────────────┘

🔹 STAGE 4: CLASSIFICATION (SVM)
   • Model: Support Vector Machine
   • Hyperparameter Tuning: GridSearchCV (10-fold CV)
   • Search Space:
     - TF-IDF: n-gram {(1,1),(1,2),(1,3)} × C ∈ {0.001, 0.01, 0.1, 1, 10, 100}, kernel ∈ {linear, rbf, poly}
     - IndoBERT: C ∈ {0.001, 0.01, 0.1, 1, 10, 100}, kernel ∈ {linear, rbf, poly}
   • Best Parameters Found (reported in Chapter IV):
     ┌──────────────────┬──────┬──────────┐
     │ Model            │ C    │ Kernel   │
     ├──────────────────┼──────┼──────────┤
     │ TF-IDF App       │ 100  │ Linear ✅│
     │ TF-IDF Play      │ 100  │ Linear ✅│
     │ IndoBERT App     │ 0.01 │ Linear ✅│
     │ IndoBERT Play    │ 0.01 │ Linear ✅│
     └──────────────────┴──────┴──────────┘
     → Linear kernels = sentiment is linearly separable
   
   • Training: 80% stratified split
   • Testing: 20% hold-out (App: 166, Play: ~159)
   • Class Weighting: Balanced (inversely proportional)

🔹 STAGE 5: RESULTS & EVALUATION
   • Primary Metric: Macro F1-Score
   • Secondary Metric: Accuracy
   • Analysis: Confusion matrices, per-class F1
   • Comparison: TF-IDF vs IndoBERT, App vs Play
```

**Speaking Points** (90 sec):
> "Raw reviews enter a 6-stage preprocessing pipeline. Translation ensures all text is Indonesian, cleaning normalizes case and strips noise, tokenization splits into words, stopword removal filters 758 function words—critical note: this creates empty strings that MUST be filtered before modeling. Sastrawi stemming reduces morphological variants to root forms, producing the `ulasan_bersih` column. InSet lexicon provides ground truth labels showing severe imbalance—82% negative on Play Store. We then extract features using two methods: TF-IDF creates sparse word-frequency vectors, testing three n-gram configurations, while IndoBERT produces dense 768-dimensional contextual embeddings. Both feed into SVM classifiers optimized via grid search across multiple hyperparameter combinations. After stratified 80:20 split with class balancing, we evaluate on held-out test sets using macro F1 as primary metric."

---

## **SLIDE 4: Why This Methodology? Critical Design Choices (75 seconds)**

**Title**: Methodological Justifications: Key Design Decisions

**Content**:

```
🎯 5 CRITICAL DESIGN DECISIONS & RATIONALES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣ WHY COMPARE TF-IDF VS INDOBERT?
   ❓ Research Question: Can traditional methods match modern transformers?
   ✅ Controlled Comparison: Same classifier (SVM), same data
      → Isolates feature engineering impact
   💡 Practical Value: TF-IDF cheaper/faster vs IndoBERT sophisticated
   📊 Cost-Benefit: TF-IDF 10× faster (0.07s vs 0.82s inference)

2️⃣ WHY SVM AS SOLE CLASSIFIER?
   ✅ Handles both sparse (TF-IDF) and dense (IndoBERT) features
   ✅ Linear kernels emerged optimal → sentiment linearly separable
   ✅ Robust to overfitting with proper regularization (C parameter)
   ✅ Eliminates algorithmic variance → fair feature comparison
   ❌ Alternative (Naive Bayes, Random Forest) = adds confounding variables

3️⃣ WHY LEXICON-BASED LABELING (InSet)?
   ✅ No pre-labeled Indonesian sentiment dataset available
   ✅ InSet: 10,218 Indonesian terms (largest available lexicon)
      - 3,609 positive terms
      - 6,609 negative terms
   ✅ Provides consistent, reproducible ground truth
   ⚠️ Limitation: May miss slang/colloquialisms (acknowledged)

4️⃣ WHY MACRO F1 AS PRIMARY METRIC? (Most Critical)
   🔴 Class Imbalance Reality:
      • App Store: 66% negative
      • Play Store: 82% negative
   
   ❌ ACCURACY TRAP:
      Naive baseline (always predict "Negatif"):
      • Play Store: 82% accuracy WITHOUT learning!
      • App Store: 66% accuracy WITHOUT learning!
      → Accuracy is dangerously misleading
   
   ✅ MACRO F1 SOLUTION:
      • Treats all 3 classes equally (unweighted average)
      • Forces minority class detection (Netral, Positif)
      • Aligns with business needs:
        - Negatif: Identify technical issues ✅
        - Netral: Early churn signals (retention) 💼
        - Positif: Marketing insights (amplify features) 📈
   
   📊 Example Impact:
      Play Store TF-IDF:
      • Accuracy: 73.21% (looks good!)
      • Macro F1: 0.38 (reveals poor balance)
      • Positif F1: 0.11 (near-failure, only 8% recall)
      → Accuracy hides minority class blindness

5️⃣ WHY STRATIFIED SPLIT + CLASS BALANCING?
   ✅ Stratified Split: Test set mirrors real-world distribution
   ✅ SVM Class Weighting: Balanced (inversely proportional to freq)
   ✅ Combined Effect: Minority classes not ignored during training
   📊 Result:
      • Training set maintains 66%/82% negative distribution
      • Model forced to learn ALL three classes
      • Macro F1 ensures evaluation fairness

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 SUMMARY: Every choice addresses class imbalance systematically
   → Prevents accuracy-driven optimization that ignores business value
```

**Speaking Points** (75 sec):
> "Every methodological choice has a rationale. We compare TF-IDF versus IndoBERT to answer whether expensive transformers justify their cost—TF-IDF is 10× faster. SVM as the sole classifier eliminates algorithmic noise—performance differences come purely from feature engineering. Lexicon-based labeling with InSet provides ground truth in the absence of pre-labeled datasets. 

> The most critical decision: Macro F1 as primary metric. With 82% negative reviews on Play Store, a naive baseline achieves 82% accuracy without learning anything—it simply predicts 'Negatif' every time. Accuracy would misleadingly favor models that ignore minority classes. Macro F1 forces balanced detection—critical because Netral reviews indicate churn risk, and Positif reviews reveal features worth amplifying in marketing. 

> Our three-pronged approach—stratified splitting, class weighting, and macro F1—ensures fair evaluation despite extreme imbalance. This prevents accuracy-driven optimization that would deploy a model blind to business-critical minority classes."

---

## **SLIDE 5: From Modeling to Deployment - Delivering Value (60 seconds)**

**Title**: MANFAAT: Three-Layer Impact from Methodology to Practice

**Visual**: Three concentric circles or pyramid showing Academic → Business → Practical deployment

**Content**:

```
🎯 METHODOLOGY DELIVERS VALUE AT 3 LEVELS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎓 LAYER 1: ACADEMIC CONTRIBUTION
   📚 Literature Contribution:
      • Indonesian NLP methodology advancement
      • Controlled TF-IDF vs IndoBERT comparison
      • Handling severe class imbalance (82% negative)
   🔬 Methodological Innovation:
      • InSet lexicon application for ground truth
      • Stratified evaluation framework for imbalanced data
      • Cross-platform comparative analysis

💼 LAYER 2: BUSINESS INSIGHTS
   🔍 Temporal Analysis:
      • Pre-price increase (2020-2022): Baseline sentiment
      • Post-price increase (2023-2025): Impact assessment
      • Natural experiment design
   
   📊 Actionable Intelligence:
      ┌──────────────┬───────────────────────────────────┐
      │ Sentiment    │ Business Action                   │
      ├──────────────┼───────────────────────────────────┤
      │ Negatif      │ Prioritize technical fixes        │
      │ (66-82%)     │ (login, payment, OTP issues)      │
      ├──────────────┼───────────────────────────────────┤
      │ Netral       │ Identify churn risk users         │
      │ (11-18%)     │ Proactive retention campaigns     │
      ├──────────────┼───────────────────────────────────┤
      │ Positif      │ Amplify successful features       │
      │ (7-16%)      │ Marketing material extraction     │
      └──────────────┴───────────────────────────────────┘
   
   💡 Why Macro F1 Matters for Business:
      • Accuracy-optimized model: Detects 100% Negatif, 0% Netral/Positif
        → Misses churn signals & marketing opportunities
      • Macro F1-optimized model: Balanced detection across all classes
        → Comprehensive business intelligence

⚙️ LAYER 3: PRACTICAL DEPLOYMENT
   🖥️ Streamlit Dashboard Features:
      
      ┌─────────────────────────────────────────────────┐
      │ 1. Model Selection Panel                        │
      │    • Platform: App Store / Play Store           │
      │    • Method: TF-IDF / IndoBERT                  │
      │    → Dynamically loads appropriate .pkl model   │
      └─────────────────────────────────────────────────┘
      
      ┌─────────────────────────────────────────────────┐
      │ 2. Real-Time Prediction Engine                  │
      │    • Input: Paste Indonesian review text        │
      │    • Output: Sentiment + Confidence scores      │
      │    • Speed: 0.07s (TF-IDF) vs 0.82s (IndoBERT)  │
      │    → No coding required                         │
      └─────────────────────────────────────────────────┘
      
      ┌─────────────────────────────────────────────────┐
      │ 3. Historical Analytics                         │
      │    • Sentiment distribution (pie/bar charts)    │
      │    • Time series trends (2020-2025)             │
      │    • Rating-sentiment correlation               │
      └─────────────────────────────────────────────────┘
      
      ┌─────────────────────────────────────────────────┐
      │ 4. Model Performance Metrics                    │
      │    • Confusion matrices (actual vs predicted)   │
      │    • Classification reports (precision/recall)  │
      │    • Per-class F1 breakdown                     │
      └─────────────────────────────────────────────────┘
      
      ┌─────────────────────────────────────────────────┐
      │ 5. Linguistic Insights                          │
      │    • Word clouds per sentiment category         │
      │    • Dominant keywords extraction               │
      │    • Cross-platform comparison visualizations   │
      └─────────────────────────────────────────────────┘
   
   👥 Target Users:
      • Customer Support: Prioritize negative review responses
      • Product Managers: Extract feature requests from Netral/Positif
      • Marketing Teams: Track sentiment trends, identify amplification opportunities
      • Executives: Monitor overall sentiment health, price impact

   🚀 Deployment Flow:
      User Input (Review) 
         → Preprocessing Pipeline (5 stages)
         → Feature Extraction (TF-IDF or IndoBERT)
         → SVM Prediction (Load cached .pkl model)
         → Output Display (Sentiment + Confidence)
         → Visualization Update (Real-time charts)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 METHODOLOGY → DEPLOYMENT BRIDGE:
   Academic rigor ensures reproducibility
   Business focus drives actionable metrics (Macro F1)
   User-friendly interface democratizes access to insights
```

**Speaking Points** (60 sec):
> "Our methodology delivers value at three levels. Academically, we contribute to Indonesian NLP literature by comparing traditional versus modern approaches with rigorous handling of severe class imbalance. For business stakeholders, we provide actionable intelligence—temporal analysis around the 2023 price increase identifies sentiment patterns. The macro F1 metric ensures we capture ALL business-critical insights: negative reviews for technical fixes, neutral reviews as churn signals, and positive reviews for marketing amplification. 

> For practical adoption, we deploy an interactive Streamlit dashboard requiring zero coding knowledge. Paste a review, select platform and model, receive instant sentiment classification with confidence scores in 0.07 seconds. The dashboard includes five modules: real-time prediction, historical analytics, performance metrics, word clouds, and cross-platform comparisons. Target users span customer support, product managers, marketing teams, and executives. 

> This bridges the gap between academic research and practical business intelligence—rigorous methodology ensures trustworthy results, while user-friendly deployment democratizes access to insights."

---

## 📋 **PRESENTATION STRUCTURE SUMMARY**

| Slide # | Title | Time | Key Focus |
|---------|-------|------|-----------|
| **1** | Thesis Research Flow Overview | 60s | FENOMENA → PERMASALAHAN → SOLUSI → HASIL → MANFAAT |
| **2** | CRISP-DM Framework | 90s | 6 phases with detailed examples |
| **3** | Data Pipeline Flow | 90s | DATA → PREPROCESSING → LABELING → FEATURE EXTRACTION → CLASSIFICATION |
| **4** | Methodological Justifications | 75s | 5 critical design decisions (Why TF-IDF vs IndoBERT? Why Macro F1?) |
| **5** | Deployment & Impact | 60s | Academic → Business → Practical (Dashboard) |
| **TOTAL** | | **6:15 min** | Complete methodology coverage |

---

## 🎤 **PRESENTATION SCRIPT - FULL FLOW**

### **Opening (15 seconds)**
> "Good morning/afternoon. Chapter 3 presents our research methodology following CRISP-DM, the industry-standard framework for data mining. Let me walk you through our systematic approach from problem identification to deployed solution."

### **Slide 1 → 2 Transition (5 seconds)**
> "Our research begins with identifying the phenomenon—Disney+ Hotstar's rating disparity and 2023 price increase. To structure this investigation systematically, we adopted the CRISP-DM framework..."

### **Slide 2 → 3 Transition (5 seconds)**
> "Within CRISP-DM's structure, let me detail the complete data pipeline—how raw Indonesian reviews transform into actionable predictions..."

### **Slide 3 → 4 Transition (5 seconds)**
> "Every design choice in this pipeline has a rationale. Why these specific methods? Let me justify five critical decisions..."

### **Slide 4 → 5 Transition (5 seconds)**
> "These methodological choices aren't just academic—they deliver practical value at three levels..."

### **Closing (15 seconds)**
> "To summarize: CRISP-DM ensures systematic progression from business problem to deployed solution. We compare TF-IDF versus IndoBERT using SVM to isolate feature engineering impact. Macro F1 as primary metric handles severe class imbalance—critical for capturing business-critical minority classes like churn signals and marketing insights. The result? A rigorous methodology validated through an interactive dashboard that democratizes access to sentiment intelligence. Thank you."

---

## 🔑 **MEMORIZABLE DEFENSE ANSWERS**

**If committee asks about methodology, emphasize these 3 points:**

### **Q1: "Why CRISP-DM?"**
**Answer** (30 seconds):
> "CRISP-DM is the industry standard for data science projects, providing a systematic yet iterative framework from business understanding to deployment. Unlike linear methodologies, CRISP-DM allows revisiting earlier phases as insights emerge. It ensures reproducibility through documented procedures and bridges academic research to practical implementation. In our case, it structured the progression from Disney+ Hotstar's business problem—understanding sentiment around price increases—through data collection, preprocessing, modeling, evaluation, and finally deployment as an accessible dashboard."

---

### **Q2: "What makes your comparison controlled?"**
**Answer** (30 seconds):
> "Single classifier—SVM. Same data—1,676 Indonesian reviews. Same preprocessing—6-stage pipeline (translation, cleaning, tokenization, stopword removal, stemming, final text). Only the feature extraction differs: TF-IDF represents the bag-of-words tradition with 5,000 sparse features, IndoBERT represents contextual transformers with 768 dense embeddings. This isolates feature engineering impact, answering whether expensive modern methods justify their cost. Our finding: TF-IDF wins with +0.075 average macro F1 advantage, 10× faster inference, and superior interpretability for business stakeholders."

---

### **Q3: "How do you handle extreme class imbalance?"**
**Answer** (45 seconds):
> "Three-pronged approach addressing imbalance at every stage. First, stratified train-test split preserves the real-world distribution—82% negative on Play Store—ensuring test set mirrors production. Second, SVM class weighting set to 'balanced' inversely scales by frequency, forcing the model to learn minority classes during training. Third, and most critical: macro F1-score as primary evaluation metric.

> With 82% negative reviews, accuracy is dangerously misleading. A naive baseline that always predicts 'Negatif' achieves 82% accuracy without learning anything. Macro F1 treats all three classes equally—unweighted average—forcing balanced detection. This aligns with business needs: Negatif reviews identify technical issues, Netral reviews signal churn risk, Positif reviews reveal marketing opportunities. Accuracy-driven optimization would deploy a model blind to business-critical minority classes. Our approach prevents this trap."

---

## 📊 **VISUAL SUGGESTIONS FOR EACH SLIDE**

### **Slide 1 Visual:**
- Use the provided 3-panel flowchart image
- Highlight each phase with different colors:
  - FENOMENA: Red (problem context)
  - PERMASALAHAN: Orange (research gap)
  - SOLUSI: Blue (methodology)
  - HASIL: Green (results)
  - MANFAAT: Purple (impact)

### **Slide 2 Visual:**
- Standard CRISP-DM circular diagram
- Annotate each phase with your specific example:
  - Business Understanding: "Macro F1 ≥ 0.50"
  - Data Understanding: "1,676 reviews, 2020-2025"
   - Data Preparation: "6-stage pipeline"
  - Modeling: "TF-IDF vs IndoBERT + SVM"
  - Evaluation: "Macro F1 primary metric"
  - Deployment: "Streamlit dashboard"

### **Slide 3 Visual:**
- Linear flowchart with 5 stages
- Show example transformation at each stage:
  - Input: "Aplikasi ini sangat bagus dan menyenangkan"
  - After tokenization: ["Aplikasi", "ini", "sangat", "bagus", "dan", "menyenangkan"]
  - After stopword removal: ["aplikasi", "sangat", "bagus", "menyenangkan"]
  - After stemming: ["aplikasi", "sangat", "bagus", "senang"]
  - Final: "aplikasi sangat bagus senang"

### **Slide 4 Visual:**
- Split screen comparison table:
  - Left column: TF-IDF characteristics
  - Right column: IndoBERT characteristics
  - Middle: Accuracy vs Macro F1 example showing the trap

### **Slide 5 Visual:**
- Three concentric circles or layered pyramid:
  - Outer layer: Academic (literature contribution)
  - Middle layer: Business (actionable insights)
  - Inner layer: Practical (dashboard interface)
- Include small dashboard screenshot mockup

---

## ✅ **PRE-PRESENTATION CHECKLIST**

**Content Covered:**
- ✅ 3.1 Thesis Overall Flow (Slide 1: FENOMENA → MANFAAT)
- ✅ 3.2 CRISP-DM Framework (Slide 2: 6 phases detailed)
- ✅ Data Pipeline Details (Slide 3: 6-stage preprocessing + feature extraction)
- ✅ Methodological Justifications (Slide 4: 5 critical decisions)
- ✅ Deployment & Impact (Slide 5: 3-layer value delivery)

**Time Management:**
- Total: 6 minutes 15 seconds
- Buffer for questions: 3-4 minutes
- Total with Q&A: ~10 minutes (within thesis defense allocation)

**Key Messages:**
1. CRISP-DM = systematic, reproducible, industry-standard
2. Controlled comparison = isolates feature engineering impact
3. Macro F1 = handles imbalance, aligns with business needs
4. Deployment = bridges academic rigor to practical access

**Potential Committee Questions:**
1. "Why not fine-tune IndoBERT instead of just using embeddings?"
2. "How do you validate InSet lexicon accuracy?"
3. "What if slang/colloquialisms aren't in InSet?"
4. "Why not use other classifiers like Naive Bayes or Random Forest?"
5. "How generalizable is this methodology to other Indonesian apps?"

**Backup Slides (Optional):**
- Research timeline (April 2025 data collection)
- Hardware specifications (16GB RAM, GPU optional)
- Ethical considerations (data privacy, bias mitigation)
- Library versions (scikit-learn 1.3+, transformers 4.30+)

---

**End of Presentation Chapter III**
