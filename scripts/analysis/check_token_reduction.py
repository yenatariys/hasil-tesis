import pandas as pd

# Load App Store data
df_app = pd.read_csv(r'data\processed\lex_labeled_review_app.csv')

# Calculate token reduction
initial_tokens = df_app['initial_token_count'].sum()
after_stopword = df_app['token_count'].sum()
reduction_pct = ((initial_tokens - after_stopword) / initial_tokens) * 100

print("=" * 60)
print("APP STORE TOKEN ANALYSIS:")
print("=" * 60)
print(f"Total initial tokens: {initial_tokens:,}")
print(f"Total after stopword removal: {after_stopword:,}")
print(f"Token reduction: {reduction_pct:.1f}%")
print(f"Average initial tokens per review: {df_app['initial_token_count'].mean():.1f}")
print(f"Average after stopword: {df_app['token_count'].mean():.1f}")
print(f"Column name for sentiment label: '{df_app.columns[-1]}'")

print("\n" + "=" * 60)
print("PLAY STORE TOKEN ANALYSIS:")
print("=" * 60)

# Load Play Store data
df_play = pd.read_csv(r'data\processed\lex_labeled_review_play.csv')

initial_tokens_play = df_play['initial_token_count'].sum()
after_stopword_play = df_play['token_count'].sum()
reduction_pct_play = ((initial_tokens_play - after_stopword_play) / initial_tokens_play) * 100

print(f"Total initial tokens: {initial_tokens_play:,}")
print(f"Total after stopword removal: {after_stopword_play:,}")
print(f"Token reduction: {reduction_pct_play:.1f}%")
print(f"Average initial tokens per review: {df_play['initial_token_count'].mean():.1f}")
print(f"Average after stopword: {df_play['token_count'].mean():.1f}")
print(f"Column name for sentiment label: '{df_play.columns[-1]}'")

print("\n" + "=" * 60)
print("VERIFY LABEL COLUMN NAMES:")
print("=" * 60)
print(f"App Store CSV columns: {', '.join(df_app.columns)}")
print(f"Play Store CSV columns: {', '.join(df_play.columns)}")
