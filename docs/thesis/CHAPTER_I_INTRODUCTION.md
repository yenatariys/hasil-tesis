# CHAPTER I: INTRODUCTION

## 1.1 Background

### 1.1.1 Growth of OTT Platforms in Indonesia

The digital entertainment landscape in Indonesia has undergone a remarkable transformation in recent years, driven primarily by the rapid adoption of Over-The-Top (OTT) streaming platforms. OTT platforms, which deliver media content directly to viewers via the internet, bypassing traditional distribution channels such as cable and broadcast television, have experienced exponential growth in the Indonesian market.

According to industry reports, **OTT platform consumption in Indonesia has grown at an impressive rate of approximately 40% per year**, propelling Indonesia to become the **leading OTT consumption market in Southeast Asia**. This explosive growth is fueled by several converging factors:

1. **Increased Internet Penetration**: Widespread availability of affordable mobile internet and improved telecommunications infrastructure
2. **Smartphone Adoption**: Rising smartphone ownership across diverse socioeconomic segments
3. **Changing Consumer Preferences**: Shift from traditional television viewing to on-demand, personalized content
4. **Diverse Content Offerings**: Growing library of local and international content catering to Indonesian audiences
5. **Competitive Pricing**: Affordable subscription models making premium content accessible to broader demographics

The momentum of this growth shows no signs of abating. **Statista projections indicate that the number of OTT users in Indonesia is expected to reach 197.9 million by 2030**, representing a substantial portion of Indonesia's total population. This projection underscores the strategic importance of the Indonesian market for both domestic and international streaming platforms.

### 1.1.2 Disney+ Hotstar in the Indonesian Market

Among the numerous OTT platforms operating in Indonesia—including Netflix, Amazon Prime Video, Viu, iQiyi, and local players like Vidio and GoPlay—**Disney+ Hotstar has distinguished itself by becoming the best-selling paid streaming application in Indonesia**. The platform, which combines Disney's vast entertainment library with Hotstar's sports and regional content, officially launched for public access in Indonesia in September 2020.

Disney+ Hotstar's success in the Indonesian market is reflected in its substantial user engagement and review volume. **As of March 1, 2025**, the platform has amassed significant attention across both major mobile application distribution platforms:

**Apple App Store Performance**:
- **75,400 total reviews**
- **4.8 average rating** (out of 5.0)
- Strong positive sentiment indicators

**Google Play Store Performance**:
- **117,000 total reviews**
- **2.0 average rating** (out of 5.0)
- Substantially lower rating despite higher review volume

### 1.1.3 The Rating Paradox: Platform Discrepancy

The stark contrast in average ratings between the App Store (4.8) and Play Store (2.0) presents an intriguing paradox that challenges conventional assumptions about application quality assessment. **This 2.8-point rating differential—with the Play Store rating being 58.3% lower—demonstrates that an application's popularity and market leadership do not guarantee consistent user satisfaction across all platforms.**

This phenomenon raises several critical questions:

1. **Platform Demographics**: Do iOS and Android users have fundamentally different expectations or usage patterns?
2. **Review Behavior**: Are there systematic differences in how users on different platforms express satisfaction or dissatisfaction?
3. **Technical Performance**: Does the application perform differently across platforms due to operating system constraints or optimization issues?
4. **User Composition**: Are the user bases demographically or psychographically distinct between platforms?

### 1.1.4 Limitations of Rating-Based Assessment

While numerical ratings (1-5 stars) serve as a convenient metric for assessing application popularity and perceived quality, they suffer from significant limitations:

**1. Lack of Context**:
- A single rating does not explain *why* a user is satisfied or dissatisfied
- Different users may assign the same rating for entirely different reasons
- No visibility into which specific features or aspects influenced the rating

**2. Aspect Ambiguity**:
- Ratings may reflect assessment of a single feature rather than overall experience
- Users might rate based on recent experience rather than holistic evaluation
- External factors (e.g., mood, comparison with competitors) may influence ratings

**3. Misalignment Between Rating and Review Text**:
- Reviews may contain positive text but assign low ratings (e.g., "Great app but needs feature X" → 3 stars)
- Conversely, negative text may accompany high ratings (e.g., sarcasm, conditional praise)
- Rating scales may be interpreted differently across cultures and individuals

**4. Temporal Bias**:
- Ratings do not capture changes in sentiment over time
- Recent negative experiences may overshadow long-term satisfaction
- Platform updates or changes in service quality are not reflected in aggregate ratings

These limitations underscore the necessity of moving beyond simple rating aggregation to a more nuanced understanding of user sentiment through natural language processing and sentiment analysis of review text.

### 1.1.5 The 2023 Price Increase and Its Impact

A critical event in Disney+ Hotstar's Indonesian market trajectory occurred in **2023 when the platform implemented a subscription price increase**. This pricing adjustment represents a natural experiment for sentiment analysis, as **the price change led to a documented decrease in subscriber numbers**, indicating tangible user response to the policy change.

This event creates a valuable opportunity to examine:
- How pricing changes influence user sentiment expressed in reviews
- Whether sentiment shifted significantly between the pre-increase period (2020-2022) and post-increase period (2023-2025)
- How negative feedback manifests in textual reviews versus numerical ratings
- Whether user concerns about value proposition are reflected in review language

### 1.1.6 Research Motivation: The Need for Sentiment Analysis

Given the limitations of rating-based assessment and the complex dynamics surrounding Disney+ Hotstar's market performance, **there is a clear need for comprehensive sentiment analysis that examines actual review text to uncover deeper patterns in user perception**.

Sentiment analysis—the computational task of identifying and extracting subjective opinions or emotions from text—offers several advantages over simple rating analysis:

**1. Granular Understanding**:
- Identifies specific features, aspects, or issues that drive satisfaction or dissatisfaction
- Reveals nuanced opinions beyond binary positive/negative classification

**2. Temporal Trend Analysis**:
- Tracks sentiment evolution over time
- Enables before-after comparisons (e.g., pre-price increase vs. post-price increase periods)

**3. Cross-Platform Comparison**:
- Systematically compares user sentiment across App Store and Play Store
- Identifies platform-specific concerns or preferences

**4. Lexicon-Based Baseline**:
- Utilizes Indonesian sentiment lexicons (InSet) to establish initial sentiment labels
- Provides interpretable baseline for machine learning approaches

**5. Machine Learning Enhancement**:
- Applies supervised learning (Support Vector Machine) to classify sentiment automatically
- Compares traditional (TF-IDF) versus modern (IndoBERT) feature engineering approaches

By conducting sentiment analysis on Disney+ Hotstar Indonesian reviews, this research aims to provide actionable insights that transcend the limitations of rating-based metrics and offer a data-driven understanding of user perception across platforms and time periods.

---

## 1.2 Problem Statement

Despite Disney+ Hotstar's market-leading position as the best-selling paid streaming application in Indonesia, significant questions remain regarding user sentiment and satisfaction, particularly in light of:

1. **Extreme rating disparity** between Apple App Store (4.8/5.0) and Google Play Store (2.0/5.0)
2. **Subscription price increase in 2023** resulting in subscriber decline
3. **Lack of systematic analysis** of user review text beyond aggregate ratings
4. **Temporal dynamics** of sentiment evolution from 2020-2022 (pre-price increase) to 2023-2025 (post-price increase)
5. **Platform-specific user behavior** differences between iOS and Android user bases

Traditional rating-based assessment fails to capture the nuanced opinions, specific concerns, and evolving perceptions expressed in textual reviews. Without comprehensive sentiment analysis, stakeholders lack the deep insights necessary for data-driven decision-making regarding product development, customer service, pricing strategy, and platform-specific optimizations.

**Central Problem**: How can automated sentiment analysis techniques be applied to Indonesian-language Disney+ Hotstar reviews to systematically classify user opinions, compare cross-platform sentiment patterns, and evaluate temporal sentiment trends across the pre-price increase (2020-2022) and post-price increase (2023-2025) periods?

---

## 1.3 Research Questions

This study addresses the following research questions:

**RQ1: Data Processing and Sentiment Labeling**
> How is raw review data processed to obtain sentiment labeling for Indonesian-language Disney+ Hotstar reviews?

This question examines the complete data pipeline from raw review collection through preprocessing (translation, cleaning, tokenization, stopword removal, stemming) to sentiment labeling using the Indonesian InSet lexicon.

**RQ2: Sentiment Distribution Across Platforms and Time Periods**
> How is the sentiment distribution formed on both platforms (App Store vs. Play Store) across both time periods (2020-2022 vs. 2023-2025)?

This question investigates whether sentiment patterns differ systematically between platforms and whether the 2023 price increase correlates with observable shifts in sentiment distribution.

**RQ3: Model Performance Comparison**
> How does the performance of Support Vector Machine (SVM) compare when using two different feature extraction methods (TF-IDF vectorization vs. IndoBERT embeddings) for sentiment classification?

This question evaluates which feature engineering approach—traditional bag-of-words (TF-IDF) or modern transformer-based contextual embeddings (IndoBERT)—yields superior sentiment classification performance for Indonesian text.

---

## 1.4 Research Objectives

Based on the research questions, this study aims to achieve the following objectives:

**Objective 1: Document Data Processing Pipeline**
> To understand and document how raw Disney+ Hotstar review data is collected, preprocessed, and labeled with sentiment categories through lexicon-based methodology.

This includes detailing the Indonesian text preprocessing pipeline (handling stopwords, stemming with Sastrawi, etc.) and the application of the InSet sentiment lexicon for initial sentiment assignment.

**Objective 2: Analyze Sentiment Distribution Patterns**
> To identify and characterize the sentiment distribution patterns formed on both platforms (App Store and Play Store) across both time periods (2020-2022 and 2023-2025).

This involves quantifying sentiment prevalence (positive, neutral, negative), examining cross-platform differences, and assessing temporal trends to understand the impact of the 2023 price increase on user sentiment.

**Objective 3: Evaluate SVM Model Performance**
> To evaluate and compare the classification performance of Support Vector Machine (SVM) models using two distinct feature extraction approaches: TF-IDF vectorization and IndoBERT contextual embeddings.

This includes comprehensive performance assessment using metrics appropriate for imbalanced datasets (accuracy, macro F1-score, precision, recall) and identifying which feature engineering strategy is more effective for Indonesian sentiment classification.

---

## 1.5 Research Scope and Limitations

### 1.5.1 Scope

**Geographic Scope**:
- **Focus**: Indonesia market exclusively
- **Language**: Indonesian (Bahasa Indonesia) reviews only
- **Rationale**: Disney+ Hotstar's Indonesian market represents a unique case study with significant OTT consumption growth

**Temporal Scope**:
- **Period 1 (Pre-Price Increase)**: 2020-2022
  - September 2020: Disney+ Hotstar public launch in Indonesia
  - Until December 2022: Before subscription price increase
- **Period 2 (Post-Price Increase)**: 2023-2025
  - 2023: Year of price increase implementation and subscriber decline
  - January 2023 - April 2025: Current period reflecting price change impact

**Note**: Review statistics (75.4K and 117K reviews, average ratings) as of March 1, 2025. Actual dataset scraping performed April 7, 2025.

**Platform Scope**:
- **Apple App Store**: iOS user reviews
- **Google Play Store**: Android user reviews
- **Rationale**: These two platforms represent the dominant mobile application distribution channels

**Dataset Scope**:
- **Sample Size**: 838 reviews per platform (1,676 total reviews)
  - Period 1 (2020-2022): 419 reviews per platform
  - Period 2 (2023-2025): 419 reviews per platform
- **Balancing**: Equal sample sizes per period to ensure fair temporal comparison

**Sentiment Categories**:
- **Positive**: Favorable opinions, satisfaction, praise
- **Neutral**: Balanced or mixed sentiments, factual statements
- **Negative**: Unfavorable opinions, complaints, dissatisfaction

**Machine Learning Scope**:
- **Classifier**: Support Vector Machine (SVM) exclusively
- **Feature Methods**: 
  1. TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
  2. IndoBERT (Indonesian BERT) contextual embeddings
- **Baseline**: InSet lexicon-based sentiment labeling

### 1.5.2 Limitations

**Dataset Limitations**:
- Limited to publicly available reviews (does not include private feedback)
- Sample size (838 per platform) may not capture all sentiment nuances
- Focuses on Indonesian language only (excludes multilingual reviews)

**Labeling Limitations**:
- Lexicon-based labels may contain errors compared to human annotation
- Neutral sentiment particularly challenging to distinguish
- Does not account for sarcasm, irony, or complex linguistic phenomena

**Temporal Limitations**:
- Does not capture intra-year seasonal variations
- Binary period comparison (before/after) may oversimplify continuous trends
- Cannot attribute causation definitively to price increase alone

**Model Limitations**:
- Single classifier (SVM) tested; other algorithms not explored in depth
- IndoBERT used for feature extraction only (no fine-tuning due to resource constraints)
- Class imbalance in training data may affect minority class performance

---

## 1.6 Research Benefits

This research provides value to multiple stakeholder groups:

### 1.6.1 Academic Contributions

**Theoretical Contributions**:
1. **Literature Enrichment**: Contributes to the growing body of knowledge in text mining and natural language processing (NLP), specifically focusing on sentiment analysis of streaming application reviews in Indonesian language context.

2. **Methodological Comparison**: Provides empirical evidence comparing traditional (TF-IDF) and modern (IndoBERT) feature engineering approaches for Indonesian sentiment classification, informing future research on optimal techniques for low-resource languages.

3. **Cross-Platform Analysis Framework**: Establishes a systematic framework for comparing sentiment across multiple application distribution platforms, applicable to other applications and markets.

4. **Temporal Analysis Methodology**: Demonstrates how sentiment analysis can track opinion evolution over time and assess the impact of specific events (e.g., pricing changes).

**Knowledge Dissemination**:
5. **Reproducible Research**: Provides comprehensive documentation of methodology, enabling replication and extension by future researchers.

6. **Indonesian NLP Advancement**: Contributes to the development of sentiment analysis techniques for Indonesian language, addressing the relative scarcity of NLP resources for non-English languages.

7. **Encouragement for Future Research**: Establishes baseline performance metrics and identifies research gaps (e.g., aspect-based sentiment analysis, fine-tuned IndoBERT, ensemble methods) that future studies can address.

### 1.6.2 Practical Contributions for Developers and Management

**Disney+ Hotstar Product Team**:
1. **User Perception Insights**: Provides detailed understanding of user sentiment on both App Store and Play Store, revealing platform-specific concerns and preferences.

2. **Data-Driven Decision Making**: Enables evidence-based prioritization of feature development, bug fixes, and customer service improvements based on sentiment analysis findings.

3. **Pricing Strategy Evaluation**: Offers quantitative assessment of sentiment shift following the 2023 price increase, informing future pricing and value proposition strategies.

4. **Platform Optimization**: Identifies platform-specific issues that may explain rating disparities (e.g., technical performance differences, user demographic variations).

5. **Competitive Intelligence**: Benchmark sentiment against competitors and track changes in competitive positioning over time.

6. **Early Warning System**: Deployed sentiment analysis dashboard serves as real-time monitoring tool for identifying emerging issues before they escalate.

**Broader Industry Applications**:
7. **Transferable Methodology**: Other OTT platforms and application developers can adapt this sentiment analysis framework for their own review monitoring.

8. **Resource Efficiency**: Demonstrates that traditional methods (TF-IDF + SVM) can achieve competitive performance compared to resource-intensive deep learning approaches, enabling smaller organizations to implement sentiment analysis.

### 1.6.3 Community and Consumer Benefits

**For Prospective Users**:
1. **Objective Assessment Tool**: Provides more nuanced, data-driven application assessment beyond simple rating aggregation, helping users make informed subscription decisions.

2. **Sentiment Transparency**: Reveals actual user concerns and praise points through systematic analysis, offering clearer picture of application strengths and weaknesses.

3. **Platform Selection Guidance**: Cross-platform comparison helps users understand whether iOS or Android version may better suit their needs.

**For Review Ecosystem**:
4. **Constructive Feedback Culture**: Encourages more thoughtful, detailed review writing by demonstrating how textual reviews are analyzed and valued for improvement insights.

5. **Combating Fake Reviews**: Systematic sentiment analysis can help identify anomalous patterns potentially indicative of fake or manipulated reviews.

6. **Empowering Consumer Voice**: Demonstrates how user feedback is systematically processed and can influence product development, encouraging genuine engagement.

**Societal Impact**:
7. **Digital Literacy**: Promotes understanding of how AI and machine learning techniques are applied to user-generated content in everyday applications.

8. **Transparency in Technology**: Contributes to broader transparency about how companies analyze and respond to user feedback in the digital economy.

---

## 1.7 Research Methodology Overview

This research follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** framework, which provides a structured, systematic approach to data science projects. The methodology consists of six iterative phases:

1. **Business Understanding**: Define sentiment analysis objectives for Disney+ Hotstar
2. **Data Understanding**: Collect and explore review data from App Store and Play Store
3. **Data Preparation**: Preprocess Indonesian text and apply lexicon-based sentiment labeling
4. **Modeling**: Develop SVM classifiers with TF-IDF and IndoBERT feature engineering
5. **Evaluation**: Assess model performance using accuracy, macro F1-score, and confusion matrices
6. **Deployment**: Implement Streamlit dashboard for real-time sentiment monitoring

**Key Methodological Choices**:
- **Lexicon-based baseline**: InSet Indonesian sentiment lexicon for initial labeling
- **Controlled comparison**: Single classifier (SVM) to isolate feature engineering effects
- **Comprehensive preprocessing**: 5-stage Indonesian text cleaning pipeline
- **Balanced evaluation**: Macro F1-score to handle class imbalance
- **Cross-validation**: 10-fold stratified cross-validation for hyperparameter tuning

Detailed methodology is presented in Chapter III.

---

## 1.8 Thesis Organization

This thesis is organized into five chapters:

**Chapter I: Introduction**
- Background on OTT growth in Indonesia and Disney+ Hotstar
- Problem statement and research motivation
- Research questions, objectives, scope, and benefits
- Overview of methodology

**Chapter II: Literature Review**
- Theoretical foundations of sentiment analysis
- Review of related work in Indonesian NLP
- Survey of feature engineering approaches (TF-IDF vs. BERT)
- Support Vector Machine theory and applications

**Chapter III: Research Methodology**
- Detailed CRISP-DM framework implementation
- Data collection and preprocessing pipeline
- Feature engineering techniques (TF-IDF and IndoBERT)
- SVM model development and hyperparameter tuning
- Evaluation metrics and deployment strategy

**Chapter IV: Results and Discussion**
- Modeling phase results (hyperparameter optimization, feature engineering)
- Evaluation phase results (model performance, confusion matrices, per-class metrics)
- Cross-platform comparison (App Store vs. Play Store)
- Temporal analysis (2020-2022 vs. 2023-2025)
- Discussion of findings, limitations, and implications

**Chapter V: Conclusion**
- Summary of key findings
- Theoretical and practical contributions
- Limitations and challenges
- Recommendations for stakeholders
- Future research directions

---

## 1.9 Chapter Summary

This chapter has introduced the research context, motivation, and framework for sentiment analysis of Disney+ Hotstar Indonesian user reviews. Key points include:

1. **Market Context**: Indonesia leads Southeast Asia in OTT consumption (40% annual growth), with Disney+ Hotstar as market-leading paid streaming app.

2. **Research Problem**: Stark rating disparity between App Store (4.8) and Play Store (2.0), coupled with 2023 price increase and subscriber decline, necessitates deeper sentiment analysis beyond ratings.

3. **Research Questions**: How is data processed for sentiment labeling? What are sentiment distributions across platforms and time periods? How do TF-IDF and IndoBERT compare for SVM classification?

4. **Research Objectives**: Document preprocessing pipeline, analyze sentiment patterns, and evaluate feature engineering approaches.

5. **Research Scope**: 838 reviews per platform (1,676 total), covering 2020-2022 (pre-price increase) and 2023-2025 (post-price increase), using SVM with lexicon-based baseline.

6. **Research Benefits**: Academic contributions to Indonesian NLP, practical insights for Disney+ Hotstar management, and community benefits through objective application assessment.

The next chapter (Chapter II) reviews relevant literature on sentiment analysis, Indonesian NLP, feature engineering techniques, and Support Vector Machine applications to text classification.

---

**End of Chapter I**
