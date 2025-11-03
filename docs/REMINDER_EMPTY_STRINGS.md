# 📝 IMPORTANT REMINDER: Empty String Filtering

**Date Created:** November 3, 2025  
**Purpose:** Track empty strings after preprocessing for both platforms

---

## 🔢 Quick Reference

| Platform | Original Dataset | Empty Strings | Usable Samples | Train Set | Test Set |
|----------|-----------------|---------------|----------------|-----------|----------|
| **App Store** | 838 | **8** (0.95%) | 830 | 664 | **166** |
| **Play Store** | 838 | **41** (4.89%) | 797 | ~638 | **~160** |

---

## ⚠️ Key Points to Remember

1. **All 838 reviews are NON-NULL** ✓
2. **But some are EMPTY STRINGS** after preprocessing ⚠️
3. **Empty strings must be filtered** before model training (cannot be vectorized)
4. **This is CORRECT and NECESSARY** - not an error!

---

## 📍 Where This is Documented

### ✅ Already Updated:
- [x] `docs/technical/DATA_FILTERING_NOTE.md` - Full detailed explanation
- [x] `docs/thesis/THESIS_EVALUATION_PHASE.md` - Added note in section 5.1.2

### ⚠️ Need to Update:
- [ ] `docs/technical/evaluation_phase.md` - Update test set sizes
- [ ] `outputs/reports/EVALUATION_RESULTS_COMBINED.md` - Update sample counts
- [ ] Any other docs mentioning "168 samples" or "20% of 838 = 168"

---

## 🔍 Why Reviews Become Empty

Reviews that become empty strings typically contain only:
- 👍 **Emojis** (e.g., "👍👍👍")
- ❗ **Punctuation** (e.g., "!!!", "...")
- 🔢 **Numbers** (e.g., "12345")
- 📝 **Stopwords only** (e.g., "di dan ke")
- 🎯 **Very short text** that doesn't survive preprocessing

**Example Flow:**
```
Original: "👍👍👍 !!!"
→ After cleaning: "" (empty)
→ After stemming: "" (still empty)
→ MUST BE FILTERED OUT
```

---

## 📊 Impact on Your Thesis

### What to Write:

**In Methodology Chapter (Chapter 3):**
> "After text preprocessing (cleaning, tokenization, stopword removal, and stemming), reviews that became empty strings were filtered out before model training. This affected 8 samples (0.95%) for App Store and 41 samples (4.89%) for Play Store, resulting in final modeling datasets of 830 and 797 samples respectively."

**In Evaluation Chapter (Chapter 5):**
> "The test sets contain 166 samples for App Store and approximately 160 samples for Play Store (20% stratified split of the filtered datasets)."

---

## 🎯 Quick Verification

To check these numbers in your notebooks:

**Cell Output to Look For:**
```
--- Checks after regenerating 'ulasan_bersih' ---
Number of non-null values in 'ulasan_bersih': 838
Number of NaN values in 'ulasan_bersih': 0
Number of empty strings ('') in 'ulasan_bersih': X  ← THIS NUMBER
```

**App Store:** X = 8  
**Play Store:** X = 41

---

## ✅ Filtering Code (Keep This!)

```python
# Drop rows where 'ulasan_bersih' is null or empty before splitting
df_filtered_for_tuning = df.dropna(subset=['ulasan_bersih'])
df_filtered_for_tuning = df_filtered_for_tuning[df_filtered_for_tuning['ulasan_bersih'].str.strip() != '']
```

**DO NOT REMOVE THIS CODE!** It's necessary for proper model training.

---

## 🚨 Common Misconception

❌ **WRONG:** "My dataset has 838 samples, so test set should be 168"  
✅ **CORRECT:** "My dataset has 838 samples, but 8 became empty after preprocessing, so I have 830 usable samples, and test set is 166"

---

## 📁 Related Files

1. **Detailed Documentation:**
   - `docs/technical/DATA_FILTERING_NOTE.md` (Full explanation)

2. **Updated Thesis Chapter:**
   - `docs/thesis/THESIS_EVALUATION_PHASE.md` (Added note in 5.1.2)

3. **Notebooks:**
   - `notebooks/appstore/Tesis-Appstore-FIX.ipynb` (8 empty strings)
   - `notebooks/playstore/Tesis-Playstore-FIX.ipynb` (41 empty strings)

---

**Remember:** This is STANDARD and CORRECT data preprocessing! 🎓

**File Location:** `docs/REMINDER_EMPTY_STRINGS.md`
