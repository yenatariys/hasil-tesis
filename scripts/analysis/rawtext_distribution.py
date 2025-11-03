import pandas as pd

# Load raw datasets
app_df = pd.read_csv('data/lex_labeled_review_app.csv')
play_df = pd.read_csv('data/lex_labeled_review_play.csv')

# Count words in each review
# App Store uses 'text' column
app_df['word_count'] = app_df['text'].astype(str).str.split().str.len()
# Play Store uses 'content' column
play_df['word_count'] = play_df['content'].astype(str).str.split().str.len()

def categorize_length(word_count):
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

# Categorize word counts
app_df['length_category'] = app_df['word_count'].apply(categorize_length)
play_df['length_category'] = play_df['word_count'].apply(categorize_length)

# Calculate percentage distribution of length categories
app_dist = app_df['length_category'].value_counts(normalize=True).mul(100).round(1)
play_dist = play_df['length_category'].value_counts(normalize=True).mul(100).round(1)