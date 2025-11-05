# WordCloud Analysis & Documentation

## Overview

Wordclouds are generated directly from word frequency analysis of preprocessed reviews (`ulasan_bersih` column), ensuring perfect consistency between frequency tables and visual representations.

**Related Analysis:**
- [Word Frequency Data](../../../outputs/word_frequencies/word_frequencies.json)
- [App Store Evaluation](../../../outputs/reports/EVALUATION_RESULTS_APPSTORE.md)
- [Play Store Evaluation](../../../outputs/reports/EVALUATION_RESULTS_PLAYSTORE.md)
- [Platform Comparison](../../../outputs/reports/PLATFORM_COMPARISON_ANALYSIS.md)

## Configuration

### Visual Parameters
- **Dimensions:** 1200×600 pixels
- **Max Words:** 50 per wordcloud
- **Background:** White
- **Color Schemes:**
  - Negatif: Red colormap
  - Netral: Blue colormap
  - Positif: Green colormap

### Text Processing
- **Source Column:** `ulasan_bersih` (preprocessed text)
- **Stopwords:** 132 Indonesian common words removed
- **Business Terms Preserved:** 40 domain-specific keywords

## App Store Wordclouds (838 total reviews)

### 1. Negatif Sentiment (503 reviews, 60.0%)
- **Most Frequent:** film (128), langgan (90), masuk (75)
- **Authentication:** kode (70), otp (59), login (35)
- **Payment:** bayar (57)
- **Technical:** tv (72), buka (35)

### 2. Netral Sentiment (205 reviews, 24.5%)
- **Most Frequent:** film (34), tv (25), langgan (25)
- **Technical:** otp (18), download (12), login (12)
- **Platform:** apple (12), chromecast (10)
- **Features:** fitur (10), unduh (11)

### 3. Positif Sentiment (124 reviews, 14.8%)
- **Most Frequent:** langgan (34), film (21), dukung (15)
- **Features:** tv (12), tambah (11), sedia (10)
- **Value:** harga (11), paket (9)
- **Platform:** apple (9), chromecast (8)

## Play Store Wordclouds (838 total reviews)

### 1. Negatif Sentiment (467 reviews, 55.7%)
- **Most Frequent:** langgan (117), film (95), nonton (78)
- **Payment:** bayar (77)
- **Technical:** download (48), gambar (39), suara (34)
- **Access:** login (38), buka (35), masuk (34)

### 2. Netral Sentiment (227 reviews, 27.1%)
- **Most Frequent:** film (38), login (21)
- **Core Features:** nonton (20), langgan (20)
- **Platform:** apk (14), telkomsel (8)
- **Mixed Terms:** suka (11), gk (8)

### 3. Positif Sentiment (105 reviews, 12.5%)
- **Most Frequent:** langgan (48), film (14)
- **Satisfaction:** mantap (13), oke (11)
- **Features:** nonton (12), video (7)
- **Value:** paket (7), gratis (5)

## Key Differences Between Platforms

### Content & Features
- App Store: More focus on device integration (apple, chromecast)
- Play Store: More streaming-related terms (nonton, tonton, video)

### Technical Issues
- App Store: Authentication-heavy (otp, kode, login)
- Play Store: Streaming quality focus (gambar, suara)

### Payment & Subscription
- Both platforms: 'langgan' consistently high-frequency
- Play Store: More payment provider mentions (telkomsel)

## Generation Process
1. Word frequency counting from preprocessed text
2. Stopword removal (except business terms)
3. Frequency-proportional sizing
4. Sentiment-specific color schemes
5. High-resolution PNG output (150 DPI)

## Wordcloud Generation Criteria

### 1. Data Processing
- **Source Files:** 
  - App Store: `data/processed/lex_labeled_review_app.csv` (838 reviews)
  - Play Store: `data/processed/lex_labeled_review_play.csv` (838 reviews)
- **Text Column:** `ulasan_bersih` (preprocessed review text)
- **NaN Handling:** Dropped before processing

### 2. Text Filtering

**Indonesian Stopwords (REMOVED):**
```python
INDONESIAN_STOPWORDS = {
    'ada', 'adalah', 'agar', 'akan', 'aku', 'anda', 'apa', 'atau', 'bagai',
    'bagaimana', 'bagi', 'bahkan', 'bahwa', 'banyak', 'begitu', 'biasa', 'bila',
    # Common Indonesian connectors, pronouns, fillers
}
```

**Business Terms (PRESERVED):**
```python
BUSINESS_TERMS_ALLOWLIST = {
    'film', 'langgan', 'langganan', 'nonton', 'tonton', 'bayar', 'download', 'unduh',
    'tv', 'kode', 'otp', 'login', 'masuk', 'buka', 'konten', 'subtitle', 'gambar',
    'layar', 'suara', 'buffer', 'loading', 'lemot', 'gratis', 'harga', 'dukungan',
    'dukung', 'paket', 'telkomsel', 'mantap', 'oke', 'kualitas', 'server', 'lag',
    'stream', 'hotstar', 'chromecast', 'error', 'salah', 'akun', 'rekening'
}
```

### 3. Visual Parameters
```python
WordCloud(
    width=1200,           # Image width
    height=600,           # Image height
    background_color='white',
    colormap={
        'Negatif': 'Reds',    # Red color scheme for negative
        'Netral': 'Blues',    # Blue for neutral
        'Positif': 'Greens'   # Green for positive
    },
    max_words=50,         # Show top 50 most frequent words
    relative_scaling=0.5, # Word size proportional to frequency
    min_font_size=10     # Minimum readable text size
)
```

### 4. Sentiment Groups

**App Store (838 total):**
- Negatif: 503 reviews (60.0%)
- Netral: 205 reviews (24.5%)
- Positif: 124 reviews (14.8%)

**Play Store (838 total):**
- Negatif: 467 reviews (55.7%)
- Netral: 227 reviews (27.1%)
- Positif: 105 reviews (12.5%)

### 5. Processing Steps
1. Load CSV files
2. Filter by sentiment (Negatif/Netral/Positif)
3. Remove NaN values from `ulasan_bersih` column
4. Join all review text for each sentiment
5. Remove Indonesian stopwords (EXCEPT business terms)
6. Generate wordcloud with size proportional to word frequency
7. Apply sentiment-specific color scheme
8. Save as high-resolution PNG (1200×600, 150 DPI)

## Generated Files

### App Store (838 reviews)

1. **`wordcloud_appstore_negatif.png`** (503 reviews, 60.0%)
   - Top words: film, langgan, masuk, tv, kode, otp, bayar, login
   - Dominant issues: Authentication (masuk, kode, otp) + Payment (bayar, langgan)

2. **`wordcloud_appstore_netral.png`** (205 reviews, 24.5%)
   - Top words: film, tv, langgan, otp, download, login, chromecast
   - Mixed feedback about features and content

3. **`wordcloud_appstore_positif.png`** (124 reviews, 14.8%)
   - Top words: langgan, film, dukung, tv, harga, tambah, sedia, paket
   - Satisfaction with subscription value and content variety

### Play Store (838 reviews)

4. **`wordcloud_playstore_negatif.png`** (467 reviews, 55.7%)
   - Top words: langgan, film, nonton, bayar, download, tonton, hp, gambar
   - Dominant issues: Streaming quality (gambar, nonton) + Technical problems

5. **`wordcloud_playstore_netral.png`** (227 reviews, 27.1%)
   - Top words: film, login, nonton, langgan, apk, bayar, video
   - Mixed experiences with app functionality

6. **`wordcloud_playstore_positif.png`** (105 reviews, 12.5%)
   - Top words: langgan, film, mantap, nonton, oke, paket, update
   - Positive feedback on service and updates

## Comparison with Notebook Wordclouds

| Aspect | Original Notebook Wordclouds | New Frequency-Based Wordclouds |
|--------|------------------------------|--------------------------------|
| **Data Source** | `ulasan_bersih` column | ✅ `ulasan_bersih` column |
| **Stopwords** | Extensive custom Indonesian list | ✅ Indonesian-only stopwords minus business keywords |
| **film, langgan, nonton** | ❌ Filtered out | ✅ Prominently displayed |
| **Consistency with frequency tables** | ⚠️ Different words shown | ✅ Perfect match |
| **Business intelligence** | Secondary technical details | ✅ Primary business concerns |
| **Use case** | Academic presentation (filtered view) | Thesis defense (complete view) |

## Usage Recommendation

**For Thesis Defense:**
- Use these **new wordclouds** (perfect consistency with frequency analysis)
- Show direct correlation: "As shown in frequency table, 'langgan' (117 mentions) and 'film' (95 mentions) are top concerns"
- Visual + numerical evidence = stronger argument

**For Academic Publication:**
- Consider using both sets:
  - Frequency-based (this): Complete business intelligence
  - Notebook wordclouds: Filtered view highlighting secondary issues

## Regeneration

To regenerate these wordclouds:

```powershell
python scripts/generate_wordcloud_from_frequencies.py
```

Output will be saved to: `docs/analysis/wordclouds/`

## Technical Details

- **Resolution:** 1200×600 pixels, 150 DPI
- **Format:** PNG with white background
- **Color Scheme:**
  - Negatif: Red colormap
  - Netral: Blue colormap
  - Positif: Green colormap
- **Font Scaling:** Relative scaling (0.5) for proportional sizing
- **Minimum Font Size:** 10pt (ensures readability)

---

**Note:** These wordclouds complement `WORD_FREQUENCY_ANALYSIS.md` by providing visual representation of the exact same data, ensuring consistency between quantitative (frequency tables) and qualitative (visual) analysis.
