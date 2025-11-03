import pandas as pd

# Load raw datasets
app_df = pd.read_csv('data/lex_labeled_review_app.csv')
play_df = pd.read_csv('data/lex_labeled_review_play.csv')

# Parse date columns to datetime objects
app_df['date_parsed'] = pd.to_datetime(app_df['date'], errors='coerce')
play_df['date_parsed'] = pd.to_datetime(play_df['at'], errors='coerce')

# Extract year from parsed dates
app_df['year'] = app_df['date_parsed'].dt.year
play_df['year'] = play_df['date_parsed'].dt.year

# Count reviews per year
print("App Store yearly distribution:")
print(app_df['year'].value_counts().sort_index())

print("\nPlay Store yearly distribution:")
print(play_df['year'].value_counts().sort_index())