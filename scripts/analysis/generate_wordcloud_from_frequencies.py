"""
Generate WordClouds from Word Frequency Analysis Results

This script generates wordcloud visualizations using the exact same methodology
as word_frequency_analysis.py, ensuring perfect consistency between frequency
tables and visual representations.

Key Features:
- Uses ulasan_bersih column (preprocessed text)
- Removes Indonesian stopwords while keeping business-critical terms
    (film, langgan, nonton, bayar, dll.)
- Word sizes proportional to actual frequencies
- Generates 6 wordclouds: 2 platforms × 3 sentiments

Output: Saves PNG files to docs/analysis/wordclouds/
"""

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# File paths
APP_STORE_CSV = 'data/processed/lex_labeled_review_app.csv'
PLAY_STORE_CSV = 'data/processed/lex_labeled_review_play.csv'
OUTPUT_DIR = 'docs/analysis/wordclouds'

# Indonesian stopwords (common connectors, pronouns, fillers)
INDONESIAN_STOPWORDS = {
    'ada', 'adalah', 'agar', 'akan', 'aku', 'anda', 'apa', 'atau', 'bagai', 'bagaimanapun',
    'bagaimana', 'bagi', 'bahkan', 'bahwa', 'banyak', 'baru', 'begitu', 'biasa', 'bila',
    'bisa', 'bukan', 'cara', 'cukup', 'dalam', 'dan', 'dengan', 'di', 'dia', 'diri',
    'itu', 'jadi', 'jalan', 'jangan', 'jika', 'juga', 'kala', 'kalau', 'kali', 'kami',
    'kamu', 'karena', 'ke', 'kebetulan', 'kecil', 'kecuali', 'keinginan', 'keluar',
    'kemudian', 'kenapa', 'kepada', 'ketika', 'kita', 'lagi', 'lah', 'lain', 'lalu',
    'lebih', 'macam', 'maka', 'makanya', 'mampu', 'mana', 'masih', 'masing', 'masuk',
    'maupun', 'melainkan', 'melalui', 'mereka', 'mesti', 'mungkin', 'namun', 'nanti',
    'oleh', 'pada', 'para', 'pernah', 'perlu', 'pula', 'saat', 'saja', 'saling', 'sama',
    'sampai', 'sangat', 'saya', 'sebab', 'sebuah', 'sebagai', 'sebagian', 'sebelum',
    'sebuah', 'sedang', 'sedikit', 'sehingga', 'sejak', 'sekali', 'selain', 'selalu',
    'seluruh', 'semakin', 'semua', 'sementara', 'sendiri', 'seorang', 'sering', 'serta',
    'setelah', 'setiap', 'siapa', 'sini', 'situ', 'suatu', 'supaya', 'sudah', 'tadi',
    'tahu', 'tapi', 'tanpa', 'tentang', 'terhadap', 'termasuk', 'tersebut', 'tetapi',
    'tidak', 'tinggi', 'tugas', 'untuk', 'walau', 'walaupun', 'yang', 'ya', 'yah', 'yg'
}

# Business-critical terms we want to keep in the visualization
BUSINESS_TERMS_ALLOWLIST = {
    'film', 'langgan', 'langganan', 'nonton', 'tonton', 'bayar', 'download', 'unduh',
    'tv', 'kode', 'otp', 'login', 'masuk', 'buka', 'konten', 'subtitle', 'gambar',
    'layar', 'suara', 'buffer', 'loading', 'lemot', 'gratis', 'harga', 'dukungan',
    'dukung', 'paket', 'telkomsel', 'mantap', 'oke', 'kualitas', 'server', 'lag',
    'stream', 'hotstar', 'chromecast', 'error', 'salah', 'akun', 'rekening'
}

# Combine stopwords (Indonesian only) minus allowlist
CUSTOM_STOPWORDS = set(INDONESIAN_STOPWORDS) - BUSINESS_TERMS_ALLOWLIST

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(filepath):
    """Load CSV file and return dataframe"""
    try:
        df = pd.read_csv(filepath)
        print(f"[OK] Loaded {filepath}: {len(df)} reviews")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}")
        return None
    except Exception as e:
        print(f"[ERROR] Loading {filepath}: {str(e)}")
        return None

def generate_wordcloud_from_text(text, title, sentiment, platform, max_words=50):
    """
    Generate wordcloud from text with minimal stopwords filtering.
    Only removes generic English stopwords, keeps Indonesian business terms.
    """
    if not text or text.strip() == '':
        print(f"[WARNING] No text available for {title}")
        return None
    
    # Color scheme based on sentiment
    colormap = {
        'Negatif': 'Reds',
        'Netral': 'Blues', 
        'Positif': 'Greens'
    }
    
    try:
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            colormap=colormap.get(sentiment, 'viridis'),
            stopwords=CUSTOM_STOPWORDS,
            max_words=max_words,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text)
        
        # Create figure
        plt.figure(figsize=(15, 7.5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout(pad=0)
        
        # Save to file
        filename = f"wordcloud_{platform.lower()}_{sentiment.lower()}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[OK] Saved: {filepath}")
        return wordcloud
        
    except Exception as e:
        print(f"[ERROR] Generating wordcloud for {title}: {str(e)}")
        return None

def process_sentiment_wordcloud(df, platform, sentiment, text_column='ulasan_bersih'):
    """Process and generate wordcloud for a specific sentiment"""
    # Filter by sentiment and drop missing/empty values
    reviews_series = df[df['sentimen_multiclass'] == sentiment][text_column].dropna()
    
    if reviews_series.empty:
        print(f"[WARNING] No {sentiment} reviews found for {platform}")
        return
    
    # Combine all text for this sentiment
    text_content = ' '.join(reviews_series.astype(str).tolist())
    
    # Count reviews (matches word_frequency_analysis.py)
    review_count = len(reviews_series)
    
    # Generate wordcloud
    title = f"WordCloud - {platform} {sentiment} ({review_count} reviews)"
    print(f"\n[PROCESSING] {title}")
    
    generate_wordcloud_from_text(
        text=text_content,
        title=title,
        sentiment=sentiment,
        platform=platform.replace(' ', '')
    )

def main():
    """Main execution function"""
    print("="*70)
    print("WORDCLOUD GENERATION FROM WORD FREQUENCY ANALYSIS")
    print("="*70)
    print("\nConfiguration:")
    print(f"- Text Column: ulasan_bersih")
    print(f"- Stopwords: {len(CUSTOM_STOPWORDS)} Indonesian words (after allowlist)")
    print(f"- Business Terms Preserved: {len(BUSINESS_TERMS_ALLOWLIST)} keywords")
    print(f"- Output: {OUTPUT_DIR}/")
    print(f"- Max Words: 50 per wordcloud")
    print("="*70)
    
    # Load datasets
    print("\n[STEP 1] Loading datasets...")
    df_app = load_data(APP_STORE_CSV)
    df_play = load_data(PLAY_STORE_CSV)
    
    if df_app is None or df_play is None:
        print("\n[ERROR] Failed to load datasets. Exiting.")
        return
    
    # Verify ulasan_bersih column exists
    for df, name in [(df_app, 'App Store'), (df_play, 'Play Store')]:
        if 'ulasan_bersih' not in df.columns:
            print(f"[ERROR] Column 'ulasan_bersih' not found in {name} dataset")
            return
        if 'sentimen_multiclass' not in df.columns:
            print(f"[ERROR] Column 'sentimen_multiclass' not found in {name} dataset")
            return
    
    sentiments = ['Negatif', 'Netral', 'Positif']
    
    # Generate App Store wordclouds
    print("\n" + "="*70)
    print("[STEP 2] Generating App Store Wordclouds")
    print("="*70)
    for sentiment in sentiments:
        process_sentiment_wordcloud(df_app, 'App Store', sentiment)
    
    # Generate Play Store wordclouds
    print("\n" + "="*70)
    print("[STEP 3] Generating Play Store Wordclouds")
    print("="*70)
    for sentiment in sentiments:
        process_sentiment_wordcloud(df_play, 'Play Store', sentiment)
    
    print("\n" + "="*70)
    print("[COMPLETE] All wordclouds generated successfully!")
    print("="*70)
    print(f"\nOutput location: {os.path.abspath(OUTPUT_DIR)}/")
    print("\nGenerated files:")
    for platform in ['appstore', 'playstore']:
        for sentiment in ['negatif', 'netral', 'positif']:
            print(f"  - wordcloud_{platform}_{sentiment}.png")
    
    print("\n[NOTE] These wordclouds use the same ulasan_bersih column as")
    print("       word_frequency_analysis.py, ensuring perfect consistency")
    print("       between frequency tables and visualizations.")
    print("="*70)

if __name__ == "__main__":
    main()
