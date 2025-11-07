"""
Dataset Statistics Calculator
This script calculates all the descriptive statistics for the App Store and Play Store datasets
Used in the Data Understanding section of the thesis (CRISP-DM Phase 2)
"""


import pandas as pd
from datetime import datetime
import os

# Prepare output directory and file
output_dir = 'outputs/results/datasets_statistics/'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'dataset_statistics.txt')

def write_and_print(text, file):
    print(text)
    file.write(text + '\n')

# Load datasets
with open(output_path, 'w', encoding='utf-8') as out:
    write_and_print("Loading datasets...", out)
    app_df = pd.read_csv('data/processed/lex_labeled_review_app.csv')
    play_df = pd.read_csv('data/processed/lex_labeled_review_play.csv')

    write_and_print(f"Ulasan App Store: {len(app_df)}", out)
    write_and_print(f"Ulasan Play Store: {len(play_df)}", out)
    write_and_print("\n" + "="*80 + "\n", out)

    # ============================================================================
    # 1. ANALISIS FORMAT TANGGAL
    # ============================================================================
    write_and_print("1. ANALISIS FORMAT TANGGAL", out)
    write_and_print("-" * 80, out)

    # Check date format by examining first few entries
    write_and_print(f"Contoh kolom 'date' di App Store:", out)
    write_and_print(app_df['date'].head(3).to_string(), out)
    write_and_print(f"\nContoh kolom 'at' di Play Store:", out)
    write_and_print(play_df['at'].head(3).to_string(), out)
    write_and_print("\n", out)

    # ============================================================================
    # 2. CAKUPAN TEMPORAL
    # ============================================================================
    write_and_print("2. CAKUPAN TEMPORAL", out)
    write_and_print("-" * 80, out)

    # App Store temporal analysis
    app_df['date_parsed'] = pd.to_datetime(app_df['date'])
    app_earliest = app_df['date_parsed'].min()
    app_latest = app_df['date_parsed'].max()
    app_years = sorted(app_df['date_parsed'].dt.year.unique())

    write_and_print(f"App Store:", out)
    write_and_print(f"  Ulasan Pertama: {app_earliest.strftime('%B %d, %Y')} ({app_earliest.strftime('%b %d, %Y')})", out)
    write_and_print(f"  Ulasan Terakhir:   {app_latest.strftime('%B %d, %Y')} ({app_latest.strftime('%b %d, %Y')})", out)
    write_and_print(f" Tahun dengan Ulasan: {app_years}", out)

    # Menghitung Rentang Cakupan
    app_months = (app_latest.year - app_earliest.year) * 12 + (app_latest.month - app_earliest.month)
    app_years_duration = app_months // 12
    app_remaining_months = app_months % 12
    write_and_print(f"  Rentang Cakupan: {app_months} bulan ({app_years_duration} tahun, {app_remaining_months} bulan)", out)

    # Play Store temporal analysis
    play_df['date_parsed'] = pd.to_datetime(play_df['at'])
    play_earliest = play_df['date_parsed'].min()
    play_latest = play_df['date_parsed'].max()
    play_years = sorted(play_df['date_parsed'].dt.year.unique())

    write_and_print(f"\nPlay Store:", out)
    write_and_print(f"  Ulasan Pertama: {play_earliest.strftime('%B %d, %Y')} ({play_earliest.strftime('%b %d, %Y')})", out)
    write_and_print(f"  Ulasan Terakhir:   {play_latest.strftime('%B %d, %Y')} ({play_latest.strftime('%b %d, %Y')})", out)
    write_and_print(f"  Tahun dengan Ulasan: {play_years}", out)

    # Menghitung Rentang Cakupan
    play_months = (play_latest.year - play_earliest.year) * 12 + (play_latest.month - play_earliest.month)
    play_years_duration = play_months // 12
    play_remaining_months = play_months % 12
    write_and_print(f"  Rentang Cakupan: {play_months} bulan ({play_years_duration} tahun, {play_remaining_months} bulan)", out)
    write_and_print("\n", out)

    # ============================================================================
    # 3. ANALISIS RATING
    # ============================================================================
    write_and_print("3. ANALISIS RATING", out)
    write_and_print("-" * 80, out)

    # App Store ratings
    app_avg_rating = app_df['rating'].mean()
    app_1star_pct = (app_df['rating'] == 1).sum() / len(app_df) * 100
    app_5star_pct = (app_df['rating'] == 5).sum() / len(app_df) * 100

    write_and_print(f"App Store:", out)
    write_and_print(f"  Rata-rata rating: {app_avg_rating:.2f} bintang", out)
    write_and_print(f"  Proporsi bintang 1: {app_1star_pct:.1f}%", out)
    write_and_print(f"  Proporsi bintang 5: {app_5star_pct:.1f}%", out)

    # Play Store ratings
    play_avg_rating = play_df['score'].mean()
    play_1star_pct = (play_df['score'] == 1).sum() / len(play_df) * 100
    play_5star_pct = (play_df['score'] == 5).sum() / len(play_df) * 100

    write_and_print(f"\nPlay Store:", out)
    write_and_print(f"  Rata-rata rating: {play_avg_rating:.2f} bintang", out)
    write_and_print(f"  Proporsi bintang 1: {play_1star_pct:.1f}%", out)
    write_and_print(f"  Proporsi bintang 5: {play_5star_pct:.1f}%", out)
    write_and_print("\n", out)

# ============================================================================
# 1. ANALISIS FORMAT TANGGAL
# ============================================================================
print("1. ANALISIS FORMAT TANGGAL")
print("-" * 80)

# Periksa format tanggal dengan melihat beberapa baris
print(f"Contoh kolom 'date' di App Store:")
print(app_df['date'].head(3).to_string())
print(f"\nContoh kolom 'at' di Play Store:")
print(play_df['at'].head(3).to_string())
print("\n")

# ============================================================================
# 2. CAKUPAN TEMPORAL
# ============================================================================
print("2. CAKUPAN TEMPORAL")
print("-" * 80)

# Analisis temporal App Store
app_df['date_parsed'] = pd.to_datetime(app_df['date'])
app_earliest = app_df['date_parsed'].min()
app_latest = app_df['date_parsed'].max()
app_years = sorted(app_df['date_parsed'].dt.year.unique())

print(f"App Store:")
print(f"  Ulasan Pertama: {app_earliest.strftime('%B %d, %Y')} ({app_earliest.strftime('%b %d, %Y')})")
print(f"  Ulasan Terakhir:   {app_latest.strftime('%B %d, %Y')} ({app_latest.strftime('%b %d, %Y')})")
print(f"  Tahun dengan Ulasan: {app_years}")

# Menghitung Rentang Cakupan
app_months = (app_latest.year - app_earliest.year) * 12 + (app_latest.month - app_earliest.month)
app_years_duration = app_months // 12
app_remaining_months = app_months % 12
print(f"  Durasi Cakupan: {app_months} bulan ({app_years_duration} tahun, {app_remaining_months} bulan)")

# Analisis temporal Play Store
play_df['date_parsed'] = pd.to_datetime(play_df['at'])
play_earliest = play_df['date_parsed'].min()
play_latest = play_df['date_parsed'].max()
play_years = sorted(play_df['date_parsed'].dt.year.unique())

print(f"\nPlay Store:")
print(f"  Ulasan Pertama: {play_earliest.strftime('%B %d, %Y')} ({play_earliest.strftime('%b %d, %Y')})")
print(f"  Ulasan Terakhir:   {play_latest.strftime('%B %d, %Y')} ({play_latest.strftime('%b %d, %Y')})")
print(f"  Tahun dengan Ulasan: {play_years}")

# Menghitung Rentang Cakupan
play_months = (play_latest.year - play_earliest.year) * 12 + (play_latest.month - play_earliest.month)
play_years_duration = play_months // 12
play_remaining_months = play_months % 12
print(f"  Durasi Cakupan: {play_months} bulan ({play_years_duration} tahun, {play_remaining_months} bulan)")
print("\n")

# ============================================================================
# 3. ANALISIS RATING
# ============================================================================
print("3. ANALISIS RATING")
print("-" * 80)

# Rating App Store
app_avg_rating = app_df['rating'].mean()
app_1star_pct = (app_df['rating'] == 1).sum() / len(app_df) * 100
app_5star_pct = (app_df['rating'] == 5).sum() / len(app_df) * 100

print(f"App Store:")
print(f"  Rata-rata rating: {app_avg_rating:.2f} bintang")
print(f"  Proporsi bintang 1: {app_1star_pct:.1f}%")
print(f"  Proporsi bintang 5: {app_5star_pct:.1f}%")

# Rating Play Store
play_avg_rating = play_df['score'].mean()
play_1star_pct = (play_df['score'] == 1).sum() / len(play_df) * 100
play_5star_pct = (play_df['score'] == 5).sum() / len(play_df) * 100

print(f"\nPlay Store:")
print(f"  Rata-rata rating: {play_avg_rating:.2f} bintang")
print(f"  Proporsi bintang 1: {play_1star_pct:.1f}%")
print(f"  Proporsi bintang 5: {play_5star_pct:.1f}%")
print("\n")

# ============================================================================
# 4. ANALISIS PANJANG ULASAN
# ============================================================================
print("4. ANALISIS PANJANG ULASAN")
print("-" * 80)

# Panjang Ulasan App Store (jumlah kata)
app_df['word_count'] = app_df['text'].astype(str).str.split().str.len()
app_mean_length = app_df['word_count'].mean()
app_max_length = app_df['word_count'].max()

print(f"App Store:")
print(f"  Rata-rata panjang ulasan: {app_mean_length:.2f} kata")
print(f"  Maksimal panjang ulasan: {app_max_length} kata")

# Panjang Ulasan Play Store (jumlah kata)
play_df['word_count'] = play_df['content'].astype(str).str.split().str.len()
play_mean_length = play_df['word_count'].mean()
play_max_length = play_df['word_count'].max()

print(f"\nPlay Store:")
print(f"  Rata-rata panjang ulasan: {play_mean_length:.2f} kata")
print(f"  Maksimal panjang ulasan: {play_max_length} kata")
print("\n")

# ============================================================================
# 5. KATEGORISASI PANJANG ULASAN
# ============================================================================
print("5. KATEGORISASI PANJANG ULASAN")
print("-" * 80)

# Define categorization function
def categorize_length(word_count):
    """Categorize review length based on word count"""
    if word_count <= 5:
        return 'Sangat pendek (1-5 kata)'
    elif word_count <= 15:
        return 'Pendek (6-15 kata)'
    elif word_count <= 50:
        return 'Sedang (16-50 kata)'
    elif word_count <= 100:
        return 'Panjang (51-100 kata)'
    else:
        return 'Sangat panjang (>100 kata)'

# Apply categorization
app_df['length_category'] = app_df['word_count'].apply(categorize_length)
play_df['length_category'] = play_df['word_count'].apply(categorize_length)

# Menghitung Proporsi
app_length_dist = app_df['length_category'].value_counts(normalize=True).mul(100).round(1)
play_length_dist = play_df['length_category'].value_counts(normalize=True).mul(100).round(1)

print("Distribusi Panjang Ulasan App Store:")
for category in ['Sangat pendek (1-5 kata)', 'Pendek (6-15 kata)', 'Sedang (16-50 kata)', 
                 'Panjang (51-100 kata)', 'Sangat panjang (>100 kata)']:
    pct = app_length_dist.get(category, 0.0)
    print(f"  {category}: {pct}%")

print("\nDistribusi Panjang Ulasan Play Store:")
for category in ['Sangat pendek (1-5 kata)', 'Pendek (6-15 kata)', 'Sedang (16-50 kata)', 
                 'Panjang (51-100 kata)', 'Sangat panjang (>100 kata)']:
    pct = play_length_dist.get(category, 0.0)
    print(f"  {category}: {pct}%")

# Menghitung persentase ulasan sangat pendek
app_very_short_pct = app_length_dist.get('Sangat pendek (1-5 kata)', 0.0)
play_very_short_pct = play_length_dist.get('Sangat pendek (1-5 kata)', 0.0)

print(f"\nUlasan sangat pendek (1-5 kata):")
print(f"  App Store: {app_very_short_pct}%")
print(f"  Play Store: {play_very_short_pct}%")
print("\n")

# ============================================================================
# 6. TABEL RINGKASAN
# ============================================================================
print("6. TABEL RINGKASAN")
print("-" * 80)

summary_data = {
    'Karakteristik': [
        'Format tanggal',
        'Ulasan pertama',
        'Ulasan terakhir',
        'Tahun dengan ulasan',
        'Durasi cakupan',
        'Rata-rata rating',
        'Proporsi bintang 1',
        'Proporsi bintang 5',
        'Rata-rata panjang ulasan',
        'Panjang maksimal ulasan',
        'Ulasan sangat pendek (1-5 kata)'
    ],
    'App Store': [
        'YYYY-MM-DD (date only)',
        f"{app_earliest.strftime('%b %d, %Y')}",
        f"{app_latest.strftime('%b %d, %Y')}",
        ', '.join(map(str, app_years)),
        f"{app_months} bulan ({app_years_duration} tahun, {app_remaining_months} bulan)",
        f"{app_avg_rating:.2f} bintang",
        f"{app_1star_pct:.1f}%",
        f"{app_5star_pct:.1f}%",
        f"{app_mean_length:.2f} kata",
        f"{app_max_length} kata",
        f"{app_very_short_pct}%"
    ],
    'Play Store': [
        'YYYY-MM-DD HH:MM:SS (timestamp)',
        f"{play_earliest.strftime('%b %d, %Y')}",
        f"{play_latest.strftime('%b %d, %Y')}",
        ', '.join(map(str, play_years)),
        f"{play_months} bulan ({play_years_duration} tahun, {play_remaining_months} bulan)",
        f"{play_avg_rating:.2f} bintang",
        f"{play_1star_pct:.1f}%",
        f"{play_5star_pct:.1f}%",
        f"{play_mean_length:.2f} kata",
        f"{play_max_length} kata",
        f"{play_very_short_pct}%"
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))
print("\n")

# ============================================================================
# 7. CONTOH ULASAN BERDASARKAN PANJANGNYA
# ============================================================================
print("7. CONTOH ULASAN BERDASARKAN PANJANGNYA (App Store)")
print("-" * 80)

for category in ['Sangat pendek (1-5 kata)', 'Pendek (6-15 kata)', 'Sedang (16-50 kata)']:
    print(f"\n{category}:")
    samples = app_df[app_df['length_category'] == category]['text'].head(2)
    for i, sample in enumerate(samples, 1):
        word_count = len(str(sample).split())
        print(f"  {i}. \"{sample}\" ({word_count} words)")

print("\n" + "="*80)
print("Kalkulasi berhasil! Semua statistik telah diekstrak dari:")
print("  - lex_labeled_review_app.csv (838 ulasan)")
print("  - lex_labeled_review_play.csv (838 ulasan)")
print("="*80)
