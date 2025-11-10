import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Disney+ Hotstar Review Dashboard", layout="wide")
st.title("Disney+ Hotstar Sentiment Analysis (Minimal)")
st.caption("Versi minimal untuk memastikan dashboard bisa berjalan di Streamlit.")

# Coba cari file data di beberapa lokasi umum
def find_data_file():
    candidates = [
        Path("data/processed/lex_labeled_review_app.csv"),
        Path("data/processed/lex_labeled_review_play.csv"),
        Path("data/lex_labeled_review_app.csv"),
        Path("data/lex_labeled_review_play.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None

data_file = find_data_file()
if not data_file:
    st.error("File data review tidak ditemukan. Pastikan file CSV sudah di-upload ke repository.")
    st.stop()

# Load data
try:
    df = pd.read_csv(data_file)
except Exception as e:
    st.error(f"Gagal membaca file: {e}")
    st.stop()

# Tampilkan beberapa kolom utama
st.subheader("Contoh Data Review")
st.dataframe(df.head(50), use_container_width=True)

# Filter rating jika kolom tersedia
if "rating_score" in df.columns:
    min_rating = int(df["rating_score"].min())
    max_rating = int(df["rating_score"].max())
    rating = st.slider("Filter rating", min_value=min_rating, max_value=max_rating, value=(min_rating, max_rating))
    df = df[df["rating_score"].between(*rating)]

# Filter sentiment jika kolom tersedia
if "sentiment_label" in df.columns:
    sentiments = sorted(df["sentiment_label"].dropna().unique())
    selected_sentiments = st.multiselect("Filter sentiment", sentiments, default=sentiments)
    df = df[df["sentiment_label"].isin(selected_sentiments)]

# Grafik distribusi rating
if "rating_score" in df.columns:
    st.subheader("Distribusi Rating")
    fig = px.histogram(df, x="rating_score", nbins=10, title="Distribusi Rating Review")
    st.plotly_chart(fig, use_container_width=True)

# Grafik distribusi sentiment
if "sentiment_label" in df.columns:
    st.subheader("Distribusi Sentimen")
    fig = px.histogram(df, x="sentiment_label", title="Distribusi Sentimen Review")
    st.plotly_chart(fig, use_container_width=True)

st.success("Dashboard minimal berhasil dijalankan!")
