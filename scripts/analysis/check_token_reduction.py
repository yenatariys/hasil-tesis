
import pandas as pd
import os

# Prepare output directory and file
output_dir = 'outputs/results/token_reduction/'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'token_reduction_verification.txt')

def write_and_print(text, file):
	print(text)
	file.write(text + '\n')

with open(output_path, 'w', encoding='utf-8') as out:
	# Load App Store data
	df_app = pd.read_csv(r'data\processed\lex_labeled_review_app.csv')

	# Calculate token reduction
	initial_tokens = df_app['initial_token_count'].sum()
	after_stopword = df_app['token_count'].sum()
	reduction_pct = ((initial_tokens - after_stopword) / initial_tokens) * 100

	write_and_print("=" * 60, out)
	write_and_print("ANALISIS TOKEN APP STORE:", out)
	write_and_print("=" * 60, out)
	write_and_print(f"Total token awal: {initial_tokens:,}", out)
	write_and_print(f"Total setelah stopword removal: {after_stopword:,}", out)
	write_and_print(f"Reduksi Token: {reduction_pct:.1f}%", out)
	write_and_print(f"Rata-rata token awal per ulasan: {df_app['initial_token_count'].mean():.1f}", out)
	write_and_print(f"Rata-rata setelah stopword removal: {df_app['token_count'].mean():.1f}", out)
	write_and_print(f"Nama kolom untuk label sentimen: '{df_app.columns[-1]}'", out)

	write_and_print("\n" + "=" * 60, out)
	write_and_print("ANALISIS TOKEN PLAY STORE:", out)
	write_and_print("=" * 60, out)

	# Load Play Store data
	df_play = pd.read_csv(r'data\processed\lex_labeled_review_play.csv')

	initial_tokens_play = df_play['initial_token_count'].sum()
	after_stopword_play = df_play['token_count'].sum()
	reduction_pct_play = ((initial_tokens_play - after_stopword_play) / initial_tokens_play) * 100

	write_and_print(f"Total token awal: {initial_tokens_play:,}", out)
	write_and_print(f"Total setelah stopword removal: {after_stopword_play:,}", out)
	write_and_print(f"Reduksi Token: {reduction_pct_play:.1f}%", out)
	write_and_print(f"Rata-rata token awal per ulasan: {df_play['initial_token_count'].mean():.1f}", out)
	write_and_print(f"Rata-rata setelah stopword removal: {df_play['token_count'].mean():.1f}", out)
	write_and_print(f"Nama kolom untuk label sentimen: '{df_play.columns[-1]}'", out)

	write_and_print("\n" + "=" * 60, out)
	write_and_print("VERIFIKASI NAMA KOLOM LABEL:", out)
	write_and_print("=" * 60, out)
	write_and_print(f"Kolom CSV App Store: {', '.join(df_app.columns)}", out)
	write_and_print(f"Kolom CSV Play Store: {', '.join(df_play.columns)}", out)
