"""
Dataset Statistics Calculator
This script calculates all the descriptive statistics for the App Store and Play Store datasets
Used in the Data Understanding section of the thesis (CRISP-DM Phase 2)
"""

import pandas as pd
from datetime import datetime

# Load datasets
print("Loading datasets...")
app_df = pd.read_csv('lex_labeled_review_app.csv')
play_df = pd.read_csv('lex_labeled_review_play.csv')

print(f"App Store reviews: {len(app_df)}")
print(f"Play Store reviews: {len(play_df)}")
print("\n" + "="*80 + "\n")

# ============================================================================
# 1. DATE FORMAT ANALYSIS
# ============================================================================
print("1. DATE FORMAT ANALYSIS")
print("-" * 80)

# Check date format by examining first few entries
print(f"App Store 'date' column sample:")
print(app_df['date'].head(3).to_string())
print(f"\nPlay Store 'at' column sample:")
print(play_df['at'].head(3).to_string())
print("\n")

# ============================================================================
# 2. TEMPORAL COVERAGE
# ============================================================================
print("2. TEMPORAL COVERAGE")
print("-" * 80)

# App Store temporal analysis
app_df['date_parsed'] = pd.to_datetime(app_df['date'])
app_earliest = app_df['date_parsed'].min()
app_latest = app_df['date_parsed'].max()
app_years = sorted(app_df['date_parsed'].dt.year.unique())

print(f"App Store:")
print(f"  Earliest review: {app_earliest.strftime('%B %d, %Y')} ({app_earliest.strftime('%b %d, %Y')})")
print(f"  Latest review:   {app_latest.strftime('%B %d, %Y')} ({app_latest.strftime('%b %d, %Y')})")
print(f"  Years with data: {app_years}")

# Calculate coverage duration
app_months = (app_latest.year - app_earliest.year) * 12 + (app_latest.month - app_earliest.month)
app_years_duration = app_months // 12
app_remaining_months = app_months % 12
print(f"  Coverage duration: {app_months} months ({app_years_duration} years, {app_remaining_months} months)")

# Play Store temporal analysis
play_df['date_parsed'] = pd.to_datetime(play_df['at'])
play_earliest = play_df['date_parsed'].min()
play_latest = play_df['date_parsed'].max()
play_years = sorted(play_df['date_parsed'].dt.year.unique())

print(f"\nPlay Store:")
print(f"  Earliest review: {play_earliest.strftime('%B %d, %Y')} ({play_earliest.strftime('%b %d, %Y')})")
print(f"  Latest review:   {play_latest.strftime('%B %d, %Y')} ({play_latest.strftime('%b %d, %Y')})")
print(f"  Years with data: {play_years}")

# Calculate coverage duration
play_months = (play_latest.year - play_earliest.year) * 12 + (play_latest.month - play_earliest.month)
play_years_duration = play_months // 12
play_remaining_months = play_months % 12
print(f"  Coverage duration: {play_months} months ({play_years_duration} years, {play_remaining_months} months)")
print("\n")

# ============================================================================
# 3. RATING ANALYSIS
# ============================================================================
print("3. RATING ANALYSIS")
print("-" * 80)

# App Store ratings
app_avg_rating = app_df['rating'].mean()
app_1star_pct = (app_df['rating'] == 1).sum() / len(app_df) * 100
app_5star_pct = (app_df['rating'] == 5).sum() / len(app_df) * 100

print(f"App Store:")
print(f"  Average rating: {app_avg_rating:.2f} stars")
print(f"  1-star proportion: {app_1star_pct:.1f}%")
print(f"  5-star proportion: {app_5star_pct:.1f}%")

# Play Store ratings
play_avg_rating = play_df['score'].mean()
play_1star_pct = (play_df['score'] == 1).sum() / len(play_df) * 100
play_5star_pct = (play_df['score'] == 5).sum() / len(play_df) * 100

print(f"\nPlay Store:")
print(f"  Average rating: {play_avg_rating:.2f} stars")
print(f"  1-star proportion: {play_1star_pct:.1f}%")
print(f"  5-star proportion: {play_5star_pct:.1f}%")
print("\n")

# ============================================================================
# 4. REVIEW LENGTH ANALYSIS
# ============================================================================
print("4. REVIEW LENGTH ANALYSIS")
print("-" * 80)

# App Store review length (word count)
app_df['word_count'] = app_df['text'].astype(str).str.split().str.len()
app_mean_length = app_df['word_count'].mean()
app_max_length = app_df['word_count'].max()

print(f"App Store:")
print(f"  Mean review length: {app_mean_length:.2f} words")
print(f"  Max review length: {app_max_length} words")

# Play Store review length (word count)
play_df['word_count'] = play_df['content'].astype(str).str.split().str.len()
play_mean_length = play_df['word_count'].mean()
play_max_length = play_df['word_count'].max()

print(f"\nPlay Store:")
print(f"  Mean review length: {play_mean_length:.2f} words")
print(f"  Max review length: {play_max_length} words")
print("\n")

# ============================================================================
# 5. REVIEW LENGTH CATEGORIZATION
# ============================================================================
print("5. REVIEW LENGTH CATEGORIZATION")
print("-" * 80)

# Define categorization function
def categorize_length(word_count):
    """Categorize review length based on word count"""
    if word_count <= 5:
        return 'Very short (1-5 words)'
    elif word_count <= 15:
        return 'Short (6-15 words)'
    elif word_count <= 50:
        return 'Medium (16-50 words)'
    elif word_count <= 100:
        return 'Long (51-100 words)'
    else:
        return 'Very long (>100 words)'

# Apply categorization
app_df['length_category'] = app_df['word_count'].apply(categorize_length)
play_df['length_category'] = play_df['word_count'].apply(categorize_length)

# Calculate proportions
app_length_dist = app_df['length_category'].value_counts(normalize=True).mul(100).round(1)
play_length_dist = play_df['length_category'].value_counts(normalize=True).mul(100).round(1)

print("App Store review length distribution:")
for category in ['Very short (1-5 words)', 'Short (6-15 words)', 'Medium (16-50 words)', 
                 'Long (51-100 words)', 'Very long (>100 words)']:
    pct = app_length_dist.get(category, 0.0)
    print(f"  {category}: {pct}%")

print("\nPlay Store review length distribution:")
for category in ['Very short (1-5 words)', 'Short (6-15 words)', 'Medium (16-50 words)', 
                 'Long (51-100 words)', 'Very long (>100 words)']:
    pct = play_length_dist.get(category, 0.0)
    print(f"  {category}: {pct}%")

# Calculate very short proportion specifically
app_very_short_pct = app_length_dist.get('Very short (1-5 words)', 0.0)
play_very_short_pct = play_length_dist.get('Very short (1-5 words)', 0.0)

print(f"\nVery short reviews (<5 words):")
print(f"  App Store: {app_very_short_pct}%")
print(f"  Play Store: {play_very_short_pct}%")
print("\n")

# ============================================================================
# 6. SUMMARY TABLE
# ============================================================================
print("6. SUMMARY TABLE")
print("-" * 80)

summary_data = {
    'Characteristic': [
        'Date format',
        'Earliest review',
        'Latest review',
        'Years with data',
        'Coverage duration',
        'Avg. rating',
        '1-star proportion',
        '5-star proportion',
        'Mean review length',
        'Max review length',
        'Very short reviews (<5 words)'
    ],
    'App Store': [
        'YYYY-MM-DD (date only)',
        f"{app_earliest.strftime('%b %d, %Y')}",
        f"{app_latest.strftime('%b %d, %Y')}",
        ', '.join(map(str, app_years)),
        f"{app_months} months ({app_years_duration} years, {app_remaining_months} months)",
        f"{app_avg_rating:.2f} stars",
        f"{app_1star_pct:.1f}%",
        f"{app_5star_pct:.1f}%",
        f"{app_mean_length:.2f} words",
        f"{app_max_length} words",
        f"{app_very_short_pct}%"
    ],
    'Play Store': [
        'YYYY-MM-DD HH:MM:SS (timestamp)',
        f"{play_earliest.strftime('%b %d, %Y')}",
        f"{play_latest.strftime('%b %d, %Y')}",
        ', '.join(map(str, play_years)),
        f"{play_months} months ({play_years_duration} years, {play_remaining_months} months)",
        f"{play_avg_rating:.2f} stars",
        f"{play_1star_pct:.1f}%",
        f"{play_5star_pct:.1f}%",
        f"{play_mean_length:.2f} words",
        f"{play_max_length} words",
        f"{play_very_short_pct}%"
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))
print("\n")

# ============================================================================
# 7. SAMPLE REVIEWS BY LENGTH CATEGORY
# ============================================================================
print("7. SAMPLE REVIEWS BY LENGTH CATEGORY (App Store)")
print("-" * 80)

for category in ['Very short (1-5 words)', 'Short (6-15 words)', 'Medium (16-50 words)']:
    print(f"\n{category}:")
    samples = app_df[app_df['length_category'] == category]['text'].head(2)
    for i, sample in enumerate(samples, 1):
        word_count = len(str(sample).split())
        print(f"  {i}. \"{sample}\" ({word_count} words)")

print("\n" + "="*80)
print("Calculation complete! All statistics have been extracted from:")
print("  - lex_labeled_review_app.csv (838 reviews)")
print("  - lex_labeled_review_play.csv (838 reviews)")
print("="*80)
