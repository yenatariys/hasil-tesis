import pandas as pd
from sklearn.model_selection import train_test_split

# File paths
appstore_path = 'data/processed/lex_labeled_review_app.csv'
playstore_path = 'data/processed/lex_labeled_review_play.csv'

# Function to print distribution
def get_distribution(df, set_type):
    counts = df['sentimen_multiclass'].value_counts()
    percentages = df['sentimen_multiclass'].value_counts(normalize=True) * 100
    result = []
    for label in counts.index:
        result.append({
            'set': set_type,
            'sentiment': label,
            'count': counts[label],
            'percentage': round(percentages[label], 2)
        })
    return result

def print_and_collect(df, set_type):
    dist = get_distribution(df, set_type)
    print(f'--- {set_type} ---')
    for row in dist:
        print(f"{row['sentiment']}: {row['count']} ({row['percentage']:.2f}%)")
    print(f'Total: {len(df)}\n')
    return dist

# Process each platform
def process_platform(csv_path, platform_name):
    df = pd.read_csv(csv_path)
    print(f'Platform: {platform_name}')
    all_rows = []
    all_rows += print_and_collect(df, 'Full')
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sentimen_multiclass'])
    all_rows += print_and_collect(train_df, 'Train')
    all_rows += print_and_collect(test_df, 'Test')
    # Save to CSV
    import os
    os.makedirs('outputs', exist_ok=True)
    out_path = f"outputs/{platform_name.lower().replace(' ', '')}_sentiment_distribution.csv"
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}\n")

if __name__ == '__main__':
    print('App Store:')
    process_platform(appstore_path, 'App Store')
    print('Play Store:')
    process_platform(playstore_path, 'Play Store')
