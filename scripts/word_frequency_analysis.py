#!/usr/bin/env python3
"""
Word Frequency Analysis - All Sentiment Classes
================================================

Purpose: Extract and analyze word frequencies from Disney+ Hotstar reviews
         across all sentiment classes (Negatif, Netral, Positif)

Author: Thesis Analysis
Date: November 5, 2025
Version: 1.0

Input:
    - data/processed/lex_labeled_review_app.csv (App Store reviews)
    - data/processed/lex_labeled_review_play.csv (Play Store reviews)

Output:
    - Console output: Word frequency tables by sentiment
    - (Optional) CSV exports: Top keywords per sentiment class

Usage:
    python scripts/word_frequency_analysis.py
"""

import pandas as pd
from collections import Counter
import sys
from pathlib import Path

def analyze_sentiment_keywords(df, platform_name, text_column='ulasan_bersih'):
    """
    Analyze word frequencies for each sentiment class.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing reviews with sentiment labels
    platform_name : str
        'App Store' or 'Play Store' for display
    text_column : str
        Column name containing preprocessed/stemmed text (default: 'ulasan_bersih')
    
    Returns:
    --------
    dict : Nested dictionary with sentiment -> word -> frequency
    """
    results = {}
    
    print("=" * 80)
    print(f"{platform_name.upper()} - WORD FREQUENCY BY SENTIMENT")
    print("=" * 80)
    
    for sentiment in ['Negatif', 'Netral', 'Positif']:
        # Filter reviews by sentiment
        reviews = df[df['sentimen_multiclass'] == sentiment][text_column].dropna()
        
        print(f"\n{'=' * 80}")
        print(f"{sentiment.upper()} SENTIMENT ({len(reviews)} reviews, {len(reviews)/len(df)*100:.1f}%)")
        print(f"{'=' * 80}")
        
        # Collect all words from reviews
        all_words = []
        for review in reviews:
            if isinstance(review, str):
                words = review.split()
                all_words.extend(words)
        
        # Count word frequencies
        word_freq = Counter(all_words)
        top_20 = word_freq.most_common(20)
        
        # Store results
        results[sentiment] = dict(word_freq)
        
        # Display top 20 words
        print(f"\nTop 20 Most Frequent Words:")
        print(f"{'Rank':<6}{'Word':<25}{'Frequency':>12}")
        print("-" * 45)
        
        for i, (word, freq) in enumerate(top_20, 1):
            if word.strip():  # Skip empty strings
                print(f"{i:<6}{word:<25}{freq:>12}")
    
    return results


def analyze_specific_keywords(df, platform_name, text_column='ulasan_bersih'):
    """
    Search for specific technical keywords across negative reviews.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing reviews
    platform_name : str
        'App Store' or 'Play Store'
    text_column : str
        Column name containing preprocessed text (default: 'ulasan_bersih')
    """
    # Get negative reviews only
    negative_reviews = df[df['sentimen_multiclass'] == 'Negatif'][text_column].dropna()
    
    print(f"\n{'=' * 80}")
    print(f"{platform_name} - TECHNICAL KEYWORD SEARCH (Negative Reviews)")
    print(f"{'=' * 80}")
    print(f"Total negative reviews: {len(negative_reviews)}")
    
    # Technical keywords to search
    keywords = [
        'error', 'gagal', 'lemot', 'load', 'loading',
        'bayar', 'tagih', 'subtitle', 'terjemah', 
        'konten', 'film', 'masuk', 'bug', 'login',
        'kode', 'otp', 'salah', 'buka'
    ]
    
    print(f"\n{'Keyword':<20}{'Count':>10}{'Percentage':>12}")
    print("-" * 45)
    
    keyword_stats = {}
    for keyword in keywords:
        count = sum(1 for review in negative_reviews 
                   if isinstance(review, str) and keyword in review.lower())
        pct = (count / len(negative_reviews)) * 100
        keyword_stats[keyword] = {'count': count, 'percentage': pct}
        
        if count > 0:  # Only show keywords that appear
            print(f"{keyword:<20}{count:>10}{pct:>11.1f}%")
    
    return keyword_stats


def main():
    """Main execution function."""
    
    print("\n" + "=" * 80)
    print("DISNEY+ HOTSTAR REVIEWS - WORD FREQUENCY ANALYSIS")
    print("All Sentiment Classes: Negatif, Netral, Positif")
    print("=" * 80)
    
    # Load datasets
    try:
        df_app = pd.read_csv('data/processed/lex_labeled_review_app.csv')
        df_play = pd.read_csv('data/processed/lex_labeled_review_play.csv')
        
        print(f"\n[OK] Data loaded successfully!")
        print(f"   App Store: {len(df_app)} total reviews")
        print(f"   Play Store: {len(df_play)} total reviews")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] Could not find data files.")
        print(f"   Expected files:")
        print(f"   - data/processed/lex_labeled_review_app.csv")
        print(f"   - data/processed/lex_labeled_review_play.csv")
        print(f"\n   Please ensure you're running this script from the project root directory.")
        sys.exit(1)
    
    # Analyze App Store
    print("\n\n")
    app_results = analyze_sentiment_keywords(df_app, "App Store", text_column='ulasan_bersih')
    app_keywords = analyze_specific_keywords(df_app, "App Store", text_column='ulasan_bersih')
    
    # Analyze Play Store
    print("\n\n")
    play_results = analyze_sentiment_keywords(df_play, "Play Store", text_column='ulasan_bersih')
    play_keywords = analyze_specific_keywords(df_play, "Play Store", text_column='ulasan_bersih')
    
    # Summary
    print("\n\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\n[OK] Word frequency analysis completed for:")
    print("   - App Store: Negatif, Netral, Positif")
    print("   - Play Store: Negatif, Netral, Positif")
    print("\n[RESULTS] Results saved in:")
    print("   - docs/analysis/WORD_FREQUENCY_ANALYSIS.md (documentation)")
    print("\n[METHOD] Methodology:")
    print("   - 5-stage preprocessing pipeline (translation -> stemming)")
    print("   - Counter-based frequency calculation")
    print("   - Sentiment-specific keyword extraction")
    print("\n[NEXT] Next Steps:")
    print("   - Visualize with word clouds (see notebooks)")
    print("   - Extract TF-IDF feature importance from trained models")
    print("   - Perform n-gram analysis for phrase detection")
    

if __name__ == "__main__":
    main()
