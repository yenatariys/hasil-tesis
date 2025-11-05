# Cross-Platform Comparison Analysis
## Disney+ Hotstar: App Store vs Play Store

**Analysis Date:** November 5, 2025  
**Total Reviews:** 1,676 (838 per platform)  
**Collection Date:** April 7th, 2025

**Detailed Platform Reports:**
- [App Store Detailed Analysis](EVALUATION_RESULTS_APPSTORE.md)
- [Play Store Detailed Analysis](EVALUATION_RESULTS_PLAYSTORE.md)

**Supporting Analysis:**
- [Word Frequency Analysis](../docs/analysis/WORD_FREQUENCY_ANALYSIS.md)
- [Wordcloud Visualizations](../docs/analysis/wordclouds/README.md)
- [Raw Word Frequencies](../outputs/word_frequencies/word_frequencies.json)

## 1. Initial Sentiment Distribution Comparison

| Sentiment | App Store | Play Store | Gap |
|-----------|-----------|------------|-----|
| **Negatif** | 66.35% | 82.22% | +15.87% |
| **Netral** | 17.54% | 10.74% | -6.80% |
| **Positif** | 16.11% | 7.04% | -9.07% |

**Key Distribution Insights:**
- Play Store shows significantly higher negative sentiment (+15.87%)
- App Store has more balanced distribution
- Both platforms indicate majority user dissatisfaction

## 2. Model Performance Comparison

### TF-IDF + SVM Model

| Metric | App Store | Play Store | Better Platform |
|--------|-----------|------------|-----------------|
| Accuracy | 66.87% | 73.21% | Play Store (+6.34%) |
| Macro F1 | 0.57 | 0.38 | App Store (+0.19) |
| Weighted F1 | 0.67 | 0.72 | Play Store (+0.05) |

### IndoBERT + SVM Model

| Metric | App Store | Play Store | Better Platform |
|--------|-----------|------------|-----------------|
| Accuracy | 66.27% | 72.62% | Play Store (+6.35%) |
| Macro F1 | 0.47 | 0.33 | App Store (+0.14) |
| Weighted F1 | 0.64 | 0.71 | Play Store (+0.07) |

**Model Performance Insights:**
1. Play Store shows higher accuracy but lower F1 scores
2. App Store achieves better balance across classes
3. TF-IDF consistently outperforms IndoBERT
4. Class imbalance affects Play Store metrics more severely

## 3. Key User Concerns by Platform

### App Store Distinctive Issues
1. **Authentication Problems**
   - OTP/code-related issues more prominent
   - Login difficulties frequently mentioned
   - Account access problems

2. **Device Integration**
   - Apple TV integration
   - Chromecast compatibility
   - Platform-specific features

### Play Store Distinctive Issues
1. **Streaming Quality**
   - Video buffering
   - Audio syncing
   - Image quality concerns

2. **Mobile-Specific**
   - APK-related issues
   - Mobile compatibility
   - Telkomsel integration

### Common Issues Across Platforms
1. **Subscription Management**
   - Payment processing
   - Subscription activation
   - Billing concerns

2. **Content Access**
   - Film availability
   - Streaming permissions
   - Content restrictions

## 4. Rating vs Sentiment Correlation

| Metric | App Store | Play Store |
|--------|-----------|------------|
| MAE | 1.2387 | 1.4672 |
| RMSE | 1.6231 | 1.8453 |
| Pearson Correlation | 0.4896 | 0.3824 |
| Spearman Correlation | 0.4854 | 0.3791 |

**Rating Analysis Insights:**
- App Store shows stronger rating-sentiment alignment
- Play Store ratings diverge more from sentiment
- Both platforms show moderate correlation
- App Store users more consistent in rating-review alignment

## 5. Platform-Specific Characteristics

### App Store
- More balanced sentiment distribution
- Better multi-class classification
- Stronger rating-sentiment correlation
- Focus on authentication and device integration

### Play Store
- Higher negative sentiment concentration
- Better binary classification (negative vs non-negative)
- More streaming quality concerns
- Mobile-centric issues predominant

## Conclusion

1. **User Satisfaction:**
   - Both platforms show high dissatisfaction
   - App Store users slightly less negative
   - Play Store needs urgent attention

2. **Technical Focus:**
   - App Store: Authentication and integration
   - Play Store: Streaming and mobile functionality

3. **Model Performance:**
   - TF-IDF + SVM: Best overall approach
   - Platform-specific tuning recommended
   - Class imbalance needs addressing

4. **Recommendations:**
   - Prioritize Play Store improvements
   - Focus on platform-specific issues
   - Address common subscription concerns