

from __future__ import annotations
"""Utility to trace reviews that become empty strings during preprocessing."""

import os

import re
from typing import Dict, List

import pandas as pd


def cleaning(text: str) -> str:
    """Replicate the cleaning stage from the notebook pipeline."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Replicate the whitespace tokenization step."""
    return text.split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Apply the stopword list used in the exploratory notebooks."""
    stopwords_id = {
        "yang", "untuk", "pada", "ke", "para", "namun", "menurut", "antara",
        "dia", "dua", "ia", "seperti", "jika", "sehingga", "kembali", "dan",
        "tidak", "ini", "karena", "oleh", "sebagai", "mereka", "dengan", "dari",
        "atau", "akan", "saya", "ada", "telah", "dalam", "sudah", "tersebut",
        "itu", "bisa", "saat", "dapat", "hanya", "bahwa", "lebih", "masih",
        "juga", "setelah", "ketika", "adalah", "hingga", "serta", "di", "agar",
        "pernah", "terhadap", "yaitu", "kami", "secara", "bagi", "hal", "kita",
        "dimana", "harus",
        # Custom stopwords from the preprocessing notebook
        "ga", "aplikasi", "disney", "bagus", "tolong", "gak", "nya", "menonton",
        "anak", "udah", "yg", "banget", "sama", "banyak", "sangat", "terus",
        "apalagi", "agak", "terutama", "klo",
    }
    return [token for token in tokens if token not in stopwords_id]


def analyze_preprocessing_pipeline(df: pd.DataFrame, text_col: str, platform: str) -> Dict[str, object]:
    """Identify rows that become empty after stopword removal."""
    results: Dict[str, object] = {
        "platform": platform,
        "total_reviews": len(df),
        "empty_after_preprocessing": 0,
        "details": [],
    }

    for idx, row in df.iterrows():
        original_text = row[text_col]

        cleaned = cleaning(original_text)
        tokens = tokenize(cleaned)
        filtered_tokens = remove_stopwords(tokens)

        if not filtered_tokens:
            results["empty_after_preprocessing"] += 1
            detail = {
                "index": idx,
                "original_text": (original_text[:100] + "...") if len(original_text) > 100 else original_text,
                "original_length": len(original_text),
                "cleaned_text": cleaned,
                "tokens_before_stopwords": tokens,
                "token_count_before_stopwords": len(tokens),
                "tokens_after_stopwords": filtered_tokens,
                "token_count_after_stopwords": 0,
                "became_empty_at": "stopword_removal",
                "reason": "All tokens were removed as stopwords",
            }
            results["details"].append(detail)

    return results


def print_analysis_results(results: Dict[str, object], file=None) -> None:
    """Print a plain-text summary for the analysis, and optionally write to file."""
    def write_and_print(text):
        print(text)
        if file:
            file.write(text + '\n')

    write_and_print(f"\nSUMMARY FOR {results['platform'].upper()}")
    write_and_print("=" * 80)
    write_and_print(f"Total reviews: {results['total_reviews']}")
    write_and_print(f"Empty after preprocessing: {results['empty_after_preprocessing']}")
    percentage = 0.0
    if results['total_reviews']:
        percentage = (results['empty_after_preprocessing'] / results['total_reviews']) * 100
    write_and_print(f"Percentage: {percentage:.2f}%")
    write_and_print("=" * 80)

    details: List[Dict[str, object]] = results.get("details", [])  # type: ignore[assignment]
    if not details:
        write_and_print("No records became empty after stopword removal.")
        return

    for position, detail in enumerate(details, start=1):
        write_and_print("-" * 80)
        write_and_print(f"Empty review #{position} (index {detail['index']})")
        write_and_print(f"Original length: {detail['original_length']} characters")
        write_and_print(f"Original text: {detail['original_text']}")
        write_and_print(f"Cleaned text: {detail['cleaned_text']}")
        write_and_print(f"Tokens before stopwords ({detail['token_count_before_stopwords']}): {detail['tokens_before_stopwords']}")
        write_and_print(f"Tokens after stopwords ({detail['token_count_after_stopwords']}): {detail['tokens_after_stopwords']}")
        write_and_print(f"Became empty at: {detail['became_empty_at']}")
        write_and_print(f"Reason: {detail['reason']}")
    write_and_print("-" * 80)


def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV into a DataFrame with basic error handling."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    return df



def main() -> None:
    """Run the analysis for both platforms and save output to file."""
    output_dir = "outputs/results/empty_strings/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "empty_string_analysis.txt")

    with open(output_path, "w", encoding="utf-8") as out:
        def write_and_print(text):
            print(text)
            out.write(text + '\n')

        write_and_print("=" * 80)
        write_and_print("EMPTY STRING ANALYSIS - PREPROCESSING PIPELINE INVESTIGATION")
        write_and_print("=" * 80)

        app_path = "c:/Users/Lenovo/Downloads/hasil-tesis/data/processed/lex_labeled_review_app.csv"
        play_path = "c:/Users/Lenovo/Downloads/hasil-tesis/data/processed/lex_labeled_review_play.csv"

        try:
            df_app = load_dataset(app_path)
            write_and_print(f"[OK] Loaded App Store data: {len(df_app)} reviews")
            df_play = load_dataset(play_path)
            write_and_print(f"[OK] Loaded Play Store data: {len(df_play)} reviews")
        except FileNotFoundError as error:
            write_and_print(f"[ERROR] {error}")
            return

        app_results = analyze_preprocessing_pipeline(df_app, "text", "App Store")
        play_results = analyze_preprocessing_pipeline(df_play, "content", "Play Store")

        print_analysis_results(app_results, file=out)
        print_analysis_results(play_results, file=out)

        write_and_print("=" * 80)
        write_and_print("CONCLUSION")
        write_and_print("=" * 80)
        write_and_print("Some reviews contain only stopwords after cleaning, leaving no tokens to model.")
        write_and_print("Filter these rows before training to avoid empty strings in the processed data.")


if __name__ == "__main__":
    main()
