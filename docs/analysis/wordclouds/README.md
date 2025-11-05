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
- ✅ **Word Size:** Proportional to actual frequency counts
- ✅ **Max Words:** 50 per wordcloud

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
