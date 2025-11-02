# Language Distribution Analysis Results

## Quick Summary

**Generated**: November 2, 2025  
**Script**: `language_distribution_analysis.py`  
**Data Source**: `data/lex_labeled_review_app.csv`, `data/lex_labeled_review_play.csv`

---

## Language Distribution Statistics

### App Store (838 reviews, U.S. store)
| Language   | Count | Percentage |
|------------|-------|------------|
| Indonesian | 326   | 38.9%      |
| English    | 307   | 36.6%      |
| Unclear    | 202   | 24.1%      |
| Mixed      | 3     | 0.4%       |

### Play Store (838 reviews, Indonesian store)
| Language   | Count | Percentage |
|------------|-------|------------|
| Indonesian | 561   | 66.9%      |
| English    | 12    | 1.4%       |
| Unclear    | 263   | 31.4%      |
| Mixed      | 2     | 0.2%       |

---

## Key Insights

### 1. App Store Language Mix
- **Nearly balanced**: 38.9% Indonesian vs 36.6% English
- Reflects U.S. App Store accessibility from Indonesia
- 36.6% require translation

### 2. Play Store Indonesian Dominance
- **Strong Indonesian majority**: 66.9%
- Minimal English: 1.4%
- Only 1.4% require translation

### 3. High "Unclear" Rate
- **App Store**: 24.1% unclear reviews
- **Play Store**: 31.4% unclear reviews
- Causes:
  - Very short reviews (1-5 words): "Kayak titit", "Hebat"
  - Colloquial slang not in formal indicators: "gw kira gratis"
  - Technical terms: "I dont even get any verification code"
  - Mixed informal language without clear patterns

### 4. Code-Switching Rare
- **App Store**: 0.4% (3 reviews)
- **Play Store**: 0.2% (2 reviews)
- Example: "Where is Spiderman far from home? Tolong lah jangan nanggung-nanggung..."

---

## Detection Methodology

### Language Indicators Used

**Indonesian (30 words):**
- Function words: yang, tidak, saya, untuk, dengan, dari, ini, itu, dan, atau, adalah, akan, ada, bisa, sudah, sangat, ke, di, pada, dalam, sama, seperti, juga, karena
- Common colloquial: nya, ku, mu, kalau, kalo, aja

**English (24 words):**
- Function words: the, is, are, was, were, have, has, had, will, would, could, should, can, may, this, that, these, those, what, where, when, why, how, not, but, and

### Morphological Patterns

**Indonesian suffixes**: -nya, -kan, -an, -ku, -mu  
**English suffixes**: -ing, -ed, -tion, -ness, -ful

### Classification Rules

1. **Indonesian**: If Indonesian indicators > 0 and English indicators = 0
2. **English**: If English indicators > 0 and Indonesian indicators = 0
3. **Mixed**: If both present and neither dominates (< 1.5x difference)
4. **Unclear**: No clear indicators or morphological patterns

---

## Sample Reviews by Category

### App Store - Indonesian
```
"Adain terus ya kerjasamanya sama telkomsel, apalagi kalo ada promo paket pake telkomsel poin"
"Perfect! Sesuai janji kasih bintang 5 klo bisa di Apple TV and no ads. Thank you."
```

### App Store - English
```
"i did not get OTP code or a call. i can enjoy disney hotstar only from website."
"Please make a support for chromecast so our family can watch in TV"
```

### App Store - Unclear
```
"Kayak titit"
"gw kira gratis"
"Udah coba berlangganan, status ga berubah saldo dana ttp kepotong"
```

### App Store - Mixed
```
"Where is Spiderman far from home? Tolong lah jangan nanggung-nanggung, masa iya ada Homecoming tapi ..."
```

### Play Store - Indonesian
```
"Apk tidak guna"
"Mantap sih cuman masih kurang film nya"
```

### Play Store - English
```
"Please fix the "WiFi bug" , i couldn't open the app on any wifi"
"while watching the resolution often changes to low, please fix it"
```

### Play Store - Unclear
```
"Hebat"
"Ga bisa" mutar video"
"Lumayanlah alternatif hiburan Gratis otomatis, krn pake HALO."
```

### Play Store - Mixed
```
"Developer klo bisa tolong masukan film the doll ya 👍🏻👍🏻"
```

---

## Implications for Data Preprocessing

### 1. Translation Necessity
- **Mandatory step**: 36.6% of App Store and 1.4% of Play Store contain English
- Without translation: Incomplete lexicon matching, biased sentiment analysis
- Tool: Google Translate API (`googletrans==3.1.0a0`)

### 2. Handling "Unclear" Reviews
- 24.1-31.4% of reviews are ambiguous
- These are typically:
  - Already Indonesian but using informal/slang language
  - Very short (1-5 words) without clear indicators
  - Will be translated anyway (no harm in re-translating Indonesian → Indonesian)
- Strategy: Translate all reviews uniformly

### 3. Code-Switching Edge Cases
- Only 0.2-0.4% mix languages
- Translation handles these: each sentence translated independently
- No special preprocessing required

---

## Generated Outputs

1. **`outputs/app_store_language_distribution.csv`**
   - Full dataset with detected language labels
   - Columns: text, rating, detected_language

2. **`outputs/play_store_language_distribution.csv`**
   - Full dataset with detected language labels
   - Columns: content, score, detected_language

3. **`outputs/language_distribution_summary.csv`**
   - Statistical summary table
   - Columns: Language, App Store Count, App Store %, Play Store Count, Play Store %

---

## Comparison with Manual Inspection

Previous assumptions (before analysis):
- App Store: 94.2% English, 5.8% Indonesian ❌
- Play Store: 78.3% Indonesian, 19.1% English ❌

**Actual results (after analysis)**:
- App Store: 38.9% Indonesian, 36.6% English, 24.1% Unclear ✅
- Play Store: 66.9% Indonesian, 1.4% English, 31.4% Unclear ✅

**Learning**: Always validate assumptions with actual data analysis!

---

## Usage for Thesis Documentation

### For Data Preparation Section (CRISP-DM Phase 3):

**Copy-paste ready text:**

> Automated language detection analysis was performed on the original review text using heuristic-based pattern matching with 30 common Indonesian indicators (yang, tidak, saya, etc.) and 24 English indicators (the, is, are, etc.), plus morphological patterns.
>
> **App Store** (838 reviews, scraped from U.S. store):
> - Indonesian: 326 reviews (38.9%)
> - English: 307 reviews (36.6%)
> - Unclear: 202 reviews (24.1%) - very short reviews, technical terms, colloquial slang
> - Mixed: 3 reviews (0.4%)
>
> **Play Store** (838 reviews, scraped from Indonesian store):
> - Indonesian: 561 reviews (66.9%)
> - English: 12 reviews (1.4%)
> - Unclear: 263 reviews (31.4%) - very short reviews, technical terms, colloquial slang
> - Mixed: 2 reviews (0.2%)
>
> **Key findings**: App Store shows nearly balanced English (36.6%) and Indonesian (38.9%) distribution, reflecting U.S. store accessibility from Indonesia. Play Store demonstrates strong Indonesian dominance (66.9%) with minimal English presence (1.4%). The high unclear rate (24.1-31.4%) consists primarily of very short reviews (1-5 words) or colloquial slang not captured by formal indicators.
>
> **Implication**: Translation step is mandatory to ensure uniform Indonesian text for lexicon-based sentiment analysis, as 36.6% of App Store reviews contain English content.

---

## Reproducibility

To regenerate this analysis:
```bash
python language_distribution_analysis.py
```

Requirements:
- pandas
- re (built-in)
- Data files in `data/` folder
- Output directory `outputs/` (auto-created)

---

**Document Version**: 1.0  
**Last Updated**: November 2, 2025  
**Author**: Automated analysis via language_distribution_analysis.py
