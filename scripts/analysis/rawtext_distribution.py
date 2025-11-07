
import pandas as pd
import os

# Load raw datasets
app_df = pd.read_csv('data/processed/lex_labeled_review_app.csv')
play_df = pd.read_csv('data/processed/lex_labeled_review_play.csv')

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

# Prepare output directory
output_dir = 'outputs/results/rawtext_distribution/'
os.makedirs(output_dir, exist_ok=True)

# Save App Store distribution
app_output_path = os.path.join(output_dir, 'app_store_rawtext_distribution.txt')
with open(app_output_path, 'w', encoding='utf-8') as f:
    f.write('App Store Review Length Distribution (%):\n')
    f.write(app_dist.to_string())
    f.write('\n')

# Save Play Store distribution
play_output_path = os.path.join(output_dir, 'play_store_rawtext_distribution.txt')
with open(play_output_path, 'w', encoding='utf-8') as f:
    f.write('Play Store Review Length Distribution (%):\n')
    f.write(play_dist.to_string())
    f.write('\n')