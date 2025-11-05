# WordCloud Visualizations

**Generated:** November 5, 2025  
**Source:** `scripts/generate_wordcloud_from_frequencies.py`  
**Data Column:** `ulasan_bersih` (preprocessed text)

## Overview

These wordclouds are generated directly from the same data used in `WORD_FREQUENCY_ANALYSIS.md`, ensuring **perfect consistency** between frequency tables and visual representations.

### Key Characteristics

- ✅ **Data Source:** Same `ulasan_bersih` column as frequency analysis
- ✅ **Stopwords:** Indonesian-only stopwords (business terms like film/langgan/nonton remain)
- ✅ **Words Included:** film, langgan, nonton, tonton, bayar, login, etc. (ALL business-relevant terms)
- ✅ **Word Size:** Proportional to actual frequency counts (relative_scaling=0.5)
- ✅ **Max Words:** 50 most frequent words per wordcloud

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
