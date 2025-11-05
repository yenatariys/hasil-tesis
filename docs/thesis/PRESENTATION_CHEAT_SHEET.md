# 📋 THESIS DEFENSE CHEAT SHEET: CHAPTERS III-V
## **1-Page Quick Reference for Presentation**

---

## 🎯 **CHAPTER III: METHODOLOGY (6 minutes)**

### **🔹 SLIDE 1: Research Flow (60s)**
- **FENOMENA**: OTT growth, Rating gap (App 4.8 vs Play 2.0), 2023 price increase
- **PERMASALAHAN**: Incomplete ratings, price impact unclear
- **SOLUSI**: SVM sentiment analysis (TF-IDF vs IndoBERT)
- **HASIL**: Trained models, sentiment distributions
- **MANFAAT**: Academic + Business + Dashboard

### **🔹 SLIDE 2: CRISP-DM Framework (90s)**
1. **Business Understanding**: Macro F1 ≥ 0.50 success criteria
2. **Data Understanding**: 1,676 reviews (838 × 2 platforms), 2020-2025
3. **Data Preparation**: 5-stage pipeline (Translation → Tokenization → Stopword Removal → Stemming)
4. **Modeling**: TF-IDF (5,000 features) vs IndoBERT (768 dims) + SVM (Linear kernel)
5. **Evaluation**: Macro F1 primary, Accuracy secondary
6. **Deployment**: Streamlit dashboard

### **🔹 SLIDE 3: Data Pipeline (90s)**
- **Preprocessing**: InSet lexicon labeling → 66% App, 82% Play negative (severe imbalance)
- **Features**: TF-IDF (sparse, 1,368-1,688) vs IndoBERT (dense, 768)
- **SVM Tuning**: GridSearchCV 10-fold CV → Linear kernels win both
  - TF-IDF: C=100 | IndoBERT: C=0.01
- **Split**: Stratified 80:20, Class weighting = balanced

### **🔹 SLIDE 4: Why This Methodology? (75s)**
**5 Critical Decisions:**
1. **TF-IDF vs IndoBERT**: Controlled comparison (same SVM, isolates features)
2. **SVM Only**: Handles sparse + dense, eliminates algorithmic variance
3. **InSet Lexicon**: 10,218 terms (no pre-labeled Indonesian dataset exists)
4. **⭐ Macro F1 Primary**: Accuracy trap—82% negative baseline = 82% accuracy without learning!
   - Macro F1 forces ALL 3 classes equally → captures business value (Netral=churn, Positif=marketing)
5. **Stratified + Balanced**: Test mirrors reality, class weighting prevents minority ignorance

### **🔹 SLIDE 5: Deployment Impact (60s)**
- **Academic**: Indonesian NLP methodology, controlled comparison
- **Business**: Temporal analysis (pre/post 2023), Actionable insights per sentiment
- **Practical**: Streamlit dashboard (real-time prediction, 0.07s TF-IDF, no coding)

---

## 📊 **CHAPTER IV: RESULTS (7 minutes)**

### **🔹 SLIDE 1: Model Performance (60s)**
**Official vs Scraped Ratings:**
- App Store: 4.8 → 2.21 (Δ -2.59, 53.6% one-star) ⚠️ CRASHED
- Play Store: 2.0 → 2.27 (Δ +0.27, 53.8% one-star) ✅ Slight improvement

**TF-IDF vs IndoBERT (Macro F1):**
- App Store: **0.57** vs 0.47 | TF-IDF +0.10 advantage ✅ **BEST MODEL**
- Play Store: 0.38 vs 0.33 | TF-IDF +0.05 advantage

**Speed:** TF-IDF 10× faster (0.07s vs 0.82s inference)

### **🔹 SLIDE 2: Keyword Analysis (90s)**
**Evidence from 1,676 reviews:**

| Sentiment | Top Keywords | Business Action |
|-----------|-------------|-----------------|
| **Negatif** (66-82%) | 'masuk' (login), 'bayar' (payment), 'kode' (OTP) | 🔴 Fix auth/billing/OTP |
| **Netral** (11-18%) | 'film', 'langgan' (subscription), 'fitur' | 🟡 Churn signals, feature requests |
| **Positif** (7-16%) | 'langgan' (satisfied), 'mantap' (excellent), 'chromecast' | 🟢 Amplify features in marketing |

**Cross-Sentiment Insight:**
- 'masuk'/'bayar'/'kode' = NEGATIVE-ONLY → Actionable problems
- 'mantap'/'oke' = POSITIVE-ONLY → Satisfaction markers
- TF-IDF interpretable, IndoBERT black-box

### **🔹 SLIDE 3: Cross-Platform Asymmetry (75s)**
| Characteristic | App Store | Play Store | Δ Difference |
|----------------|-----------|------------|--------------|
| Avg Rating | 4.8★ | 2.0★ | -2.8★ gap |
| Negative % | 66% | 82% | +16 points |
| Review Length | 19.3 words | 13.2 words | -31% shorter |
| Class Balance | Moderate | Extreme | Requires platform-specific models |

**Temporal (Pre/Post 2023 Price):**
- App Store: +1.7% negative (p=0.45 NS)
- Play Store: +3.6% negative (p=0.06 marginal) → Android users MORE price-sensitive

### **🔹 SLIDE 4: Why TF-IDF Wins (60s)**
**App Store TF-IDF (0.57 Macro F1) = BEST MODEL:**
- Per-class F1: Negatif 0.79, Netral 0.30, Positif 0.62
- 50% better Netral detection than Play Store (0.30 vs 0.19)
- 6.5× better Positif detection than Play Store (0.62 vs 0.11)

**Accuracy Paradox:**
- Play Store TF-IDF: 73.21% accuracy (highest) BUT 0.38 macro F1 (worst)
- Proves accuracy misleading for imbalanced data

**Why TF-IDF Beats IndoBERT:**
- Simpler generalizes better for limited data (1,676 reviews)
- 10× faster, interpretable, lower memory

---

## 🎓 **CHAPTER V: CONCLUSIONS (5 minutes)**

### **🔹 SLIDE 1: Key Findings (90s)**
**3 Major Findings:**
1. **TF-IDF Outperforms IndoBERT**: +0.075 avg macro F1, 10× faster
   - Limited Indonesian data (1,676) favors simpler methods
2. **Severe Class Imbalance**: 82% Play Store negative
   - Macro F1 essential → captures business value of ALL classes
3. **Cross-Platform Asymmetry**: iOS 4.8★ vs Android 2.0★
   - Platform-specific models required

### **🔹 SLIDE 2: Research Contributions (60s)**
**3 Contributions:**
1. **Methodological**: Controlled TF-IDF vs IndoBERT comparison for Indonesian
2. **Empirical**: TF-IDF superior for limited data (challenges transformer dominance)
3. **Practical**: Deployed dashboard (bridges academia → business)

### **🔹 SLIDE 3: Limitations & Future Work (60s)**
**Limitations:**
- InSet lexicon may miss slang/colloquialisms
- Limited to 1,676 reviews (2020-2025)
- Binary price periods (pre/post 2023)

**Future Work:**
- Fine-tune IndoBERT (vs embeddings-only)
- Expand to multi-OTT comparison (Netflix, Viu)
- Real-time monitoring with continuous retraining

---

## 🔑 **3 MEMORIZABLE DEFENSE ANSWERS**

### **Q1: "Why Macro F1 over Accuracy?"**
**Answer (30s):**
> "With 82% negative reviews on Play Store, a naive baseline achieves 82% accuracy by always predicting 'Negatif' without learning. Macro F1 treats all 3 classes equally, forcing minority detection. Business requires ALL sentiments: Negatif (technical fixes), Netral (churn signals), Positif (marketing insights). Accuracy-driven optimization would deploy a model blind to business-critical minority classes."

---

### **Q2: "Why TF-IDF beats IndoBERT?"**
**Answer (30s):**
> "Three reasons: First, limited data—1,676 Indonesian reviews favor simpler methods that generalize better. IndoBERT's 110M parameters risk overfitting. Second, 10× speed advantage—0.07s vs 0.82s enables real-time production. Third, interpretability—TF-IDF shows 'masuk'/'bayar' drive negative sentiment; IndoBERT is a black box. For business stakeholders, explainable wins."

---

### **Q3: "What's the main research contribution?"**
**Answer (30s):**
> "We challenge the transformer-dominance assumption for low-resource Indonesian NLP. Through controlled comparison—same SVM, same data, only features differ—we prove traditional TF-IDF outperforms modern IndoBERT by +0.075 macro F1 with 10× efficiency. Deployed dashboard delivers actionable insights: 'masuk'/'bayar' negative keywords guide product fixes, 'mantap' positive keywords inform marketing. Academic rigor meets practical business impact."

---

## 📊 **CRITICAL NUMBERS TO MEMORIZE**

| Metric | Value | Context |
|--------|-------|---------|
| **Dataset** | 1,676 reviews | 838 App + 838 Play |
| **Class Imbalance** | 66% App, 82% Play | Negative dominance |
| **Best Model** | App Store TF-IDF | 0.57 Macro F1 |
| **Worst Model** | Play Store IndoBERT | 0.33 Macro F1 |
| **Speed Advantage** | 10× | TF-IDF 0.07s vs IndoBERT 0.82s |
| **Accuracy Paradox** | Play TF-IDF | 73.21% accuracy, 0.38 F1 (worst) |
| **Success Criteria** | Macro F1 ≥ 0.50 | ✅ App TF-IDF (0.57), ❌ Play models |
| **Top Negative Keywords** | masuk, bayar, kode | Login, payment, OTP issues |
| **InSet Lexicon** | 10,218 terms | 3,609 pos + 6,609 neg |
| **Preprocessing Stages** | 5 stages | Translation → Stemming |
| **Feature Dimensions** | 5,000 vs 768 | TF-IDF sparse vs IndoBERT dense |
| **SVM Kernels** | Linear (all 4) | Sentiment linearly separable |

---

## ⏱️ **TIME ALLOCATION (18 minutes total)**

| Chapter | Slides | Time | Key Message |
|---------|--------|------|-------------|
| **III: Methodology** | 5 slides | 6:15 min | CRISP-DM, Controlled comparison, Macro F1 justification |
| **IV: Results** | 4 slides | 7:00 min | TF-IDF wins, Keyword insights, Accuracy paradox |
| **V: Conclusion** | 3 slides | 5:00 min | 3 findings, 3 contributions, Future work |
| **Buffer** | - | 2:45 min | Questions, transitions |
| **TOTAL** | 12 slides | **21:00 min** | Includes Q&A buffer |

---

## 🎤 **OPENING & CLOSING SCRIPTS**

### **Opening (20 seconds)**
> "Good morning/afternoon. I will present Chapters 3 through 5 covering methodology, results, and conclusions of my thesis on Indonesian sentiment analysis for Disney+ Hotstar reviews. My research compares traditional TF-IDF versus modern IndoBERT for sentiment classification using SVM, with deployment through an interactive dashboard. Let's begin with methodology."

### **Closing (30 seconds)**
> "To summarize: Using CRISP-DM methodology, we conducted a controlled comparison proving TF-IDF outperforms IndoBERT for Indonesian sentiment analysis—plus 0.075 macro F1 advantage, 10× faster, more interpretable. Macro F1 as primary metric was critical for handling 82% negative imbalance and capturing business-critical minority classes. The deployed dashboard bridges academic research to practical business intelligence, providing real-time sentiment prediction with actionable keyword insights. Thank you. I'm ready for questions."

---

## ⚠️ **ANTICIPATED COMMITTEE QUESTIONS**

| Question | Quick Answer | Where to Elaborate |
|----------|--------------|-------------------|
| "Why not fine-tune IndoBERT?" | "Resource constraints + risk of overfitting with limited 1,676 reviews" | Future Work (Ch. V) |
| "InSet lexicon accuracy?" | "10,218 terms, largest Indonesian lexicon available. Acknowledge slang limitation" | Methodology 3.4.4 |
| "Why only SVM?" | "Controlled comparison—isolates feature engineering impact" | Methodology Slide 4 |
| "Generalizability?" | "Methodology generalizable, findings specific to Disney+ Hotstar. Future: Multi-OTT" | Limitations (Ch. V) |
| "Statistical significance?" | "App: p=0.45 NS, Play: p=0.06 marginal for temporal shift" | Results Slide 3 |
| "Dashboard users?" | "Customer support (prioritize negatives), Product (features), Marketing (trends)" | Deployment Slide 5 |

---

## ✅ **FINAL PRE-PRESENTATION CHECKLIST**

**Physical Preparation:**
- [ ] Charge laptop (minimum 80%)
- [ ] Backup presentation on USB drive
- [ ] Print this cheat sheet (1 page reference)
- [ ] Bring water bottle

**Content Verification:**
- [ ] Practice opening script (20 seconds)
- [ ] Practice closing script (30 seconds)
- [ ] Memorize 3 defense answers (Q1: Macro F1, Q2: TF-IDF wins, Q3: Contribution)
- [ ] Review critical numbers table (12 key metrics)
- [ ] Know where to find backup details (CHAPTER_IV_RESULTS_DISCUSSION.md Section 4.8)

**Mental Preparation:**
- [ ] Arrive 15 minutes early
- [ ] Breathe deeply before starting
- [ ] Remember: You know your research better than anyone
- [ ] If stuck: "That's an excellent question. Let me refer to my results..." (buy time)

---

**🎯 KEY MANTRA: "Macro F1 captures business value. TF-IDF wins with efficiency. Dashboard delivers impact."**

---

**End of Cheat Sheet**
