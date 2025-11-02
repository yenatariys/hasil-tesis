from __future__ import annotations

import ast
import json
import re
import string
from pathlib import Path
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:  # Optional dependency for loading persisted scikit-learn pipelines
    import joblib
except ImportError:  # pragma: no cover - handled gracefully in UI
    joblib = None

try:  # Optional translator for preprocessing playground
    from googletrans import Translator as _GoogleTranslator  # type: ignore
except ImportError:  # pragma: no cover - translator optional
    _GoogleTranslator = None

try:  # Optional NLP utilities
    import nltk
    from nltk.corpus import stopwords as _nltk_stopwords
    from nltk.tokenize import word_tokenize as _word_tokenize
except ImportError:  # pragma: no cover
    nltk = None
    _nltk_stopwords = None
    _word_tokenize = None

try:  # Optional Indonesian stemmer
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory  # type: ignore
except ImportError:  # pragma: no cover
    StemmerFactory = None

try:  # Optional heavy dependencies for transformer-based embeddings
    import torch
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover - handled gracefully in UI
    torch = None
    AutoModel = AutoTokenizer = None

try:  # Optional WordCloud dependency
    from wordcloud import STOPWORDS, WordCloud
except ImportError:  # pragma: no cover - handled gracefully in UI
    WordCloud = None
    STOPWORDS = set()

# Caches for optional resources shared across helpers
_LEXICON_CACHE: Optional[Dict[str, Any]] = None
_SIMULATED_STOPWORDS: Optional[set[str]] = None
_SIMULATED_STEMMER: Any = None

st.set_page_config(
    page_title="Sentiment & Pre-processing Dashboard",
    page_icon="📊",
    layout="wide",
)

CUSTOM_STYLES = """
<style>
div[data-testid="metric-container"] {
    background: #f8f9fb;
    border: 1px solid #e6e9ef;
    padding: 16px;
    border-radius: 12px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.04);
}
div[data-testid="metric-container"] .stMetric-value {
    font-size: 22px !important;
}
button[kind="secondary"] {
    border-radius: 999px !important;
}
</style>
"""

# Prefer data/ and outputs/ layout but fall back to repo root for backwards compatibility
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"


def _resolve_csv(filename: str) -> Path:
    candidates = [DATA_DIR / filename, REPO_ROOT / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return DATA_DIR / filename  # default so the error message names expected location


def _discover_exported_jsons() -> List[Path]:
    seen: Dict[Path, None] = {}
    candidates: List[Path] = []
    for directory in (OUTPUTS_DIR, REPO_ROOT):
        if not directory.exists():
            continue
        for path in directory.glob("exported_model_results_*.json"):
            resolved = path.resolve()
            if resolved not in seen:
                seen[resolved] = None
                candidates.append(path)
    return sorted(candidates)


def _discover_saved_pipelines() -> List[Path]:
    if joblib is None:
        return []

    seen: Dict[Path, None] = {}
    candidates: List[Path] = []
    for directory in (OUTPUTS_DIR, REPO_ROOT):
        if not directory.exists():
            continue
        for extension in ("*.joblib", "*.pkl"):
            for path in directory.glob(f"exported_model_*{Path(extension).suffix}"):
                resolved = path.resolve()
                if resolved not in seen:
                    seen[resolved] = None
                    candidates.append(path)
            for path in directory.glob(extension):
                resolved = path.resolve()
                if resolved not in seen:
                    seen[resolved] = None
                    candidates.append(path)
    return sorted(candidates)


def _label_saved_pipeline(path: Path) -> str:
    name = path.stem.lower()
    label_parts: List[str] = []
    if "tfidf" in name:
        label_parts.append("TF-IDF")
    elif "bert" in name:
        label_parts.append("IndoBERT SVM")
    else:
        label_parts.append("Pipeline")

    if "app" in name:
        label_parts.append("App Store")
    elif "play" in name or "playstore" in name:
        label_parts.append("Play Store")

    label = " | ".join(label_parts)
    return label if label else path.name


def _format_result_label(slot: str, source_name: str) -> str:
    base = Path(source_name).stem if Path(source_name).suffix else source_name
    descriptor = base.replace("_", " ")
    slot_prefix = "TF-IDF" if slot == "tfidf" else "IndoBERT"

    lowered = base.lower()
    if "app" in lowered:
        suffix = "App Store"
    elif "play" in lowered:
        suffix = "Play Store"
    else:
        suffix = descriptor

    return f"{slot_prefix} | {suffix}"


def _get_results_repository(slot: str) -> Dict[str, Dict[str, Any]]:
    key = f"{slot}_results_repository"
    repo = st.session_state.get(key)
    if not isinstance(repo, dict):
        repo = {}
    st.session_state[key] = repo
    return repo


def _store_loaded_result(slot: str, label: str, payload: Dict[str, Any]) -> None:
    repo = _get_results_repository(slot).copy()
    final_label = label
    if final_label not in repo:
        counter = 2
        while final_label in repo:
            final_label = f"{label} ({counter})"
            counter += 1
    repo[final_label] = payload
    st.session_state[f"{slot}_results_repository"] = repo
    st.session_state[f"{slot}_results_loaded"] = payload
    st.session_state[f"{slot}_results"] = payload
    st.session_state[f"{slot}_loaded_label"] = final_label
    st.session_state[f"{slot}_active_source"] = "loaded"
    st.session_state[f"{slot}_selected_loaded_label"] = final_label
    if slot == "bert":
        st.session_state["bert_loaded_config"] = payload.get("best_params")


def _infer_pipeline_slot(name: str, pipeline: Any | None = None) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("tfidf", "tf-idf", "tf_idf")):
        return "tfidf"
    if "bert" in lowered:
        return "bert"

    if pipeline is not None and hasattr(pipeline, "named_steps"):
        try:
            step_names = [str(step).lower() for step in pipeline.named_steps.keys()]
            if any("tfidf" in step for step in step_names):
                return "tfidf"
            if any("bert" in step or "transformer" in step for step in step_names):
                return "bert"
        except Exception:  # pragma: no cover - best effort classification
            pass

    return "unknown"


def _get_pipeline_registry() -> Dict[str, Dict[str, Any]]:
    registry = st.session_state.get("uploaded_pipelines")
    if not isinstance(registry, dict):
        registry = {}

    # Backward compatibility: previously stored as {name: pipeline}
    if registry and any(not isinstance(value, dict) for value in registry.values()):
        registry = {"unknown": registry}

    for slot in ("tfidf", "bert", "unknown"):
        if slot not in registry or not isinstance(registry[slot], dict):
            registry[slot] = {}

    st.session_state["uploaded_pipelines"] = registry
    return registry


def _classify_result_slot(content: Dict[str, Any]) -> str:
    best_params = None
    if isinstance(content, dict):
        best_params = content.get("best_params")
        if best_params is None and isinstance(content.get("grid"), dict):
            best_params = content["grid"].get("best_params")

    best_params = best_params or {}
    if any("model_name" in key.lower() for key in best_params.keys()) or "model_name" in best_params:
        return "bert"
    if any(key.startswith("param_tfidf") for key in content.keys()):
        return "tfidf"
    if any("ngram" in str(key).lower() for key in best_params.keys()):
        return "tfidf"
    if any("kernel" in str(key).lower() for key in best_params.keys()):
        return "tfidf"
    return "tfidf"


def _ensure_dataframe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value
    try:
        return pd.DataFrame(value)
    except Exception:  # noqa: BLE001
        return value


def _normalize_loaded_results(content: Dict[str, Any]) -> Dict[str, Any]:
    if content is None:
        return {}

    if "grid" in content and isinstance(content["grid"], dict):
        grid = content["grid"]
        content.setdefault("best_score", grid.get("best_score"))
        content.setdefault("best_params", grid.get("best_params"))
        cv = grid.get("cv_results")
        if isinstance(cv, dict):
            content.setdefault("cv_results", cv)

    if "classification_report" in content:
        content["classification_report"] = content["classification_report"]
    if "confusion_matrix" in content:
        content["confusion_matrix"] = np.array(content["confusion_matrix"])
    if "ngram_summary" in content:
        content["ngram_summary"] = _ensure_dataframe(content["ngram_summary"])
    if "c_search" in content:
        content["c_search"] = _ensure_dataframe(content["c_search"])
    return content


def _autoload_workspace_models() -> None:
    if st.session_state.get("_workspace_models_loaded"):
        return

    loaded_any = False

    repo_tfidf = _get_results_repository("tfidf")
    repo_bert = _get_results_repository("bert")

    for path in _discover_exported_jsons():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                content = json.load(fh)
        except Exception:  # noqa: BLE001
            continue

        normalized = _normalize_loaded_results(content if isinstance(content, dict) else {})
        slot = _classify_result_slot(normalized)

        if slot == "tfidf":
            if "tfidf_results_loaded" not in st.session_state:
                label = _format_result_label("tfidf", path.name)
                _store_loaded_result("tfidf", label, normalized)
                loaded_any = True
            elif not repo_tfidf:
                label = st.session_state.get("tfidf_loaded_label") or _format_result_label("tfidf", path.name)
                repo_tfidf[label] = st.session_state.get("tfidf_results_loaded", normalized)
                st.session_state["tfidf_results_repository"] = repo_tfidf
        elif slot == "bert":
            if "bert_results_loaded" not in st.session_state:
                label = _format_result_label("bert", path.name)
                _store_loaded_result("bert", label, normalized)
                st.session_state["bert_loaded_config"] = normalized.get("best_params")
                loaded_any = True
            elif not repo_bert:
                label = st.session_state.get("bert_loaded_label") or _format_result_label("bert", path.name)
                repo_bert[label] = st.session_state.get("bert_results_loaded", normalized)
                st.session_state["bert_results_repository"] = repo_bert

    if joblib is not None:
        registry = _get_pipeline_registry()
        for path in _discover_saved_pipelines():
            label = _label_saved_pipeline(path)
            if any(label in bucket for bucket in registry.values()):
                continue
            try:
                pipeline_obj = joblib.load(path)
                slot = _infer_pipeline_slot(path.name, pipeline_obj)
                registry[slot][label] = pipeline_obj
                st.session_state["uploaded_pipelines"] = registry
                loaded_any = True
            except Exception:  # noqa: BLE001
                continue

    if loaded_any:
        st.session_state["_workspace_models_loaded"] = True



APP_FILE = _resolve_csv("lex_labeled_review_app.csv")
PLAY_FILE = _resolve_csv("lex_labeled_review_play.csv")


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load and merge App Store and Play Store review datasets."""
    if not APP_FILE.exists() or not PLAY_FILE.exists():
        raise FileNotFoundError("Source CSV files are missing. Expected under `data/` or repository root.")

    app_df = pd.read_csv(APP_FILE)
    play_df = pd.read_csv(PLAY_FILE)

    app_df["Platform"] = "App Store"
    play_df["Platform"] = "Play Store"

    app_df = app_df.rename(
        columns={
            "date": "review_date",
            "text": "original_text",
            "rating": "rating_score",
            "translated_text": "translated_text",
            "cleaned_text": "cleaned_text",
            "tokenized_text": "tokenized_text",
            "initial_token_count": "initial_token_count",
            "stopword_removed_text": "stopword_removed_text",
            "token_count": "token_count",
            "stemmed_text": "stemmed_text",
            "ulasan_bersih": "ulasan_bersih",
            "skor_lexicon": "lexicon_score",
            "skor_lexicon_ke_rating": "lexicon_to_rating",
            "sentimen_multiclass": "sentiment_label",
        }
    )

    play_df = play_df.rename(
        columns={
            "at": "review_date",
            "content": "original_text",
            "score": "rating_score",
            "translated_content": "translated_text",
            "cleaned_content": "cleaned_text",
            "tokenized_content": "tokenized_text",
            "initial_token_count": "initial_token_count",
            "stopword_removed_content": "stopword_removed_text",
            "token_count": "token_count",
            "stemmed_content": "stemmed_text",
            "ulasan_bersih": "ulasan_bersih",
            "skor_lexicon": "lexicon_score",
            "skor_lexicon_ke_score": "lexicon_to_rating",
            "sentimen_multiclass": "sentiment_label",
        }
    )

    combined = pd.concat([app_df, play_df], ignore_index=True, sort=False)
    combined["review_date"] = _parse_review_dates(combined["review_date"])
    combined["rating_score"] = pd.to_numeric(combined["rating_score"], errors="coerce")
    combined["sentiment_label"] = combined.get("sentiment_label", "").astype(str).str.title()
    combined = combined.dropna(subset=["review_date"]).reset_index(drop=True)

    return combined


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Create sidebar filters and return the filtered dataframe."""
    with st.sidebar:
        st.header("Filters")

        platform_choices = sorted(df["Platform"].unique())
        sentiment_choices = sorted(df["sentiment_label"].dropna().unique())

        min_date = df["review_date"].min().date()
        max_date = df["review_date"].max().date()

        rating_min = int(df["rating_score"].min())
        rating_max = int(df["rating_score"].max())

        selected_platforms = st.multiselect(
            "Platform",
            platform_choices,
            default=platform_choices,
            key="platform_filter",
        )

        selected_sentiments = st.multiselect(
            "Sentiment",
            sentiment_choices,
            default=sentiment_choices,
            key="sentiment_filter",
        )

        selected_range = st.date_input(
            "Review Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_filter",
        )

        selected_ratings = st.slider(
            "Rating",
            min_value=rating_min,
            max_value=rating_max,
            value=(rating_min, rating_max),
            key="rating_filter",
        )

        keyword = st.text_input(
            "Keyword search",
            help="Filter reviews containing specific words.",
            key="keyword_filter",
        )

    mask = df["Platform"].isin(selected_platforms)
    mask &= df["sentiment_label"].isin(selected_sentiments)
    mask &= df["rating_score"].between(*selected_ratings)

    if isinstance(selected_range, tuple):
        start_date, end_date = selected_range
    else:
        start_date = end_date = selected_range
    # Compare date-only values to avoid mismatches when some rows include time components
    mask &= df["review_date"].dt.date.between(start_date, end_date)

    if keyword:
        keyword_lower = keyword.lower()
        mask &= (
            df["original_text"].astype(str).str.contains(keyword_lower, case=False, na=False)
            | df["translated_text"].astype(str).str.contains(keyword_lower, case=False, na=False)
            | df["ulasan_bersih"].astype(str).str.contains(keyword_lower, case=False, na=False)
        )

    return df.loc[mask]


def draw_summary_metrics(df: pd.DataFrame) -> None:
    """Render key metrics cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🧾 Total Reviews", f"{len(df):,}")
    with col2:
        st.metric("⭐ Average Rating", f"{df['rating_score'].mean():.2f}")
    with col3:
        sentiment_counts = df["sentiment_label"].value_counts()
        positive_share = sentiment_counts.get("Positif", 0) / max(len(df), 1) * 100
        st.metric("😊 Positive Share", f"{positive_share:.1f}%")
    with col4:
        st.metric("🗓️ Latest Review", df["review_date"].max().strftime("%d %b %Y"))


def sentiment_distribution(df: pd.DataFrame) -> None:
    sentiment_summary = (
        df.groupby(["sentiment_label", "Platform"], as_index=False)
        .size()
        .rename(columns={"size": "review_count"})
    )

    sentiment_summary["platform_total"] = sentiment_summary.groupby("Platform")["review_count"].transform("sum")
    sentiment_summary["percentage"] = np.where(
        sentiment_summary["platform_total"] > 0,
        (sentiment_summary["review_count"] / sentiment_summary["platform_total"]) * 100,
        np.nan,
    )
    sentiment_summary["percentage"] = sentiment_summary["percentage"].round(0)

    def _format_sentiment_annotation(row: pd.Series) -> str:
        if pd.isna(row["percentage"]):
            return f"{int(row['review_count'])} reviews"
        rounded_pct = int(round(float(row["percentage"])))
        return f"{int(row['review_count'])} reviews<br>{rounded_pct}%"

    sentiment_summary["annotation"] = sentiment_summary.apply(_format_sentiment_annotation, axis=1)

    fig = px.bar(
        sentiment_summary,
        x="Platform",
        y="review_count",
        color="sentiment_label",
        text="annotation",
        barmode="stack",
        title="Sentiment Distribution by Platform",
        hover_data={"review_count": True, "percentage": True},
        color_discrete_map={
            "Negatif": "#d62728",
            "Netral": "#6c757d",
            "Positif": "#2ca02c",
        },
    )
    fig.update_traces(texttemplate="%{text}", textposition="inside", textfont_size=12)
    fig.update_layout(
        xaxis_title="Platform",
        yaxis_title="Number of Reviews",
        legend_title_text="Sentiment",
    )
    st.plotly_chart(fig, use_container_width=True)


def sentiment_timeline(df: pd.DataFrame) -> None:
    monthly = (
        df.groupby([pd.Grouper(key="review_date", freq="M"), "sentiment_label"])
        .size()
        .reset_index(name="review_count")
    )

    def _label_period(timestamp: pd.Timestamp) -> Optional[str]:
        if pd.isna(timestamp):
            return None
        year = timestamp.year
        if 2020 <= year <= 2022:
            return "2020-2022"
        if 2023 <= year <= 2025:
            return "2023-2025"
        return None

    period_df = df.dropna(subset=["review_date"]).copy()
    period_df["period_bucket"] = period_df["review_date"].apply(_label_period)
    period_df = period_df.dropna(subset=["period_bucket"])
    period_summary = (
        period_df.groupby(["period_bucket", "sentiment_label"], as_index=False)
        .size()
        .rename(columns={"size": "review_count"})
    )

    if not period_summary.empty:
        period_summary["period_total"] = period_summary.groupby("period_bucket")["review_count"].transform("sum")
        period_summary["percentage"] = np.where(
            period_summary["period_total"] > 0,
            (period_summary["review_count"] / period_summary["period_total"]) * 100,
            np.nan,
        )
        period_summary["percentage"] = period_summary["percentage"].round(1)

        def _format_period_text(row: pd.Series) -> str:
            if pd.isna(row["percentage"]):
                return f"{int(row['review_count'])} reviews"
            rounded_pct = int(round(float(row["percentage"])))
            return f"{int(row['review_count'])} reviews ({rounded_pct}%)"

        period_summary["annotation"] = period_summary.apply(_format_period_text, axis=1)

    tab_monthly, tab_periodic = st.tabs(["Monthly trend", "Period comparison"])

    with tab_monthly:
        fig = px.line(
            monthly,
            x="review_date",
            y="review_count",
            color="sentiment_label",
            title="Monthly Sentiment Trend",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_periodic:
        if period_summary.empty:
            st.info("No reviews fall within the defined 2020-2022 or 2023-2025 periods.")
        else:
            fig_period = px.bar(
                period_summary,
                x="period_bucket",
                y="review_count",
                color="sentiment_label",
                barmode="stack",
                title="Sentiment Comparison by Period",
                category_orders={"period_bucket": ["2020-2022", "2023-2025"]},
                hover_data={"review_count": True, "percentage": True},
                text="annotation",
            )
            fig_period.update_traces(texttemplate="%{text}", textposition="inside", textfont_size=12)
            fig_period.update_layout(xaxis_title="Period", yaxis_title="Number of Reviews", legend_title_text="Sentiment")
            st.plotly_chart(fig_period, use_container_width=True)


def rating_distribution(df: pd.DataFrame) -> None:
    category_levels = sorted(df["rating_score"].dropna().unique())
    summary = (
        df.groupby(["rating_score", "Platform"], as_index=False)
        .size()
        .rename(columns={"size": "review_count"})
    )

    summary["platform_total"] = summary.groupby("Platform")["review_count"].transform("sum")
    summary["percentage"] = np.where(
        summary["platform_total"] > 0,
        (summary["review_count"] / summary["platform_total"]) * 100,
        np.nan,
    )
    summary["percentage"] = summary["percentage"].round(1)

    fig = px.bar(
        summary,
        x="rating_score",
        y="review_count",
        color="Platform",
        barmode="group",
        category_orders={"rating_score": category_levels},
        labels={"rating_score": "Rating", "review_count": "Number of Reviews"},
        title="Rating Breakdown by Platform",
        hover_data={"review_count": True, "percentage": True},
    )
    fig.update_layout(legend_title_text="Platform", uniformtext_minsize=10, uniformtext_mode="hide")
    fig.update_xaxes(type="category")
    st.plotly_chart(fig, use_container_width=True)


def _round_metric(value: float, digits: int = 3) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return float("nan")
        return round(float(value), digits)
    except Exception:
        return float("nan")


METRIC_ORDER = ["MAE", "RMSE", "Pearson r", "Spearman ρ"]


def _compute_platform_metrics(platform_df: pd.DataFrame) -> Dict[str, float]:
    mae = np.mean(np.abs(platform_df["lexicon_to_rating"] - platform_df["rating_score"]))
    rmse = np.sqrt(np.mean((platform_df["lexicon_to_rating"] - platform_df["rating_score"]) ** 2))
    pearson = platform_df["rating_score"].corr(platform_df["lexicon_to_rating"], method="pearson")
    spearman = platform_df["rating_score"].corr(platform_df["lexicon_to_rating"], method="spearman")

    metrics = {
        "MAE": _round_metric(mae),
        "RMSE": _round_metric(rmse),
        "Pearson r": _round_metric(pearson),
        "Spearman ρ": _round_metric(spearman),
    }
    return metrics


def _render_platform_evaluation(
    platform: str,
    platform_df: pd.DataFrame,
    expand_visuals: bool = False,
    metrics_summary: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    st.subheader(platform)

    if metrics_summary is None:
        metrics_summary = _compute_platform_metrics(platform_df)
    metrics_df = pd.DataFrame({"Metric": METRIC_ORDER, "Value": [metrics_summary[m] for m in METRIC_ORDER]})
    st.dataframe(
        metrics_df.style.format({"Value": "{:.3f}"}),
        hide_index=True,
        use_container_width=True,
    )

    try:
        actual = platform_df["rating_score"].round().astype(int)
        lexicon = platform_df["lexicon_to_rating"].round().astype(int)

        pivot = pd.crosstab(actual, lexicon).reindex(index=range(1, 6), columns=range(1, 6), fill_value=0)

        fig_counts = px.imshow(
            pivot.values,
            x=pivot.columns.astype(str),
            y=pivot.index.astype(str),
            text_auto=True,
            color_continuous_scale="Blues",
            labels={"x": "Lexicon-derived Rating", "y": "Original Rating"},
            title=f"Rating Consistency — Counts ({platform})",
        )
        fig_counts.update_layout(
            xaxis_title="Lexicon-derived Rating",
            yaxis_title="Original Rating",
            coloraxis_colorbar=dict(title="Count"),
        )
        fig_counts.update_xaxes(type="category", categoryorder="array", categoryarray=[str(v) for v in pivot.columns])
        fig_counts.update_yaxes(type="category", categoryorder="array", categoryarray=[str(v) for v in pivot.index])
        fig_counts.data[0].text = pivot.astype(int).astype(str).values
        fig_counts.update_traces(texttemplate="%{text}", textfont_size=14)

        if expand_visuals:
            fig_counts.update_layout(height=650)
        st.plotly_chart(fig_counts, use_container_width=True)
    except Exception:
        heatmap = px.density_heatmap(
            platform_df,
            x="rating_score",
            y="lexicon_to_rating",
            nbinsx=5,
            nbinsy=5,
            color_continuous_scale="Blues",
            title=f"Rating Consistency Heatmap ({platform})",
        )
        if expand_visuals:
            heatmap.update_layout(height=650)
        st.plotly_chart(heatmap, use_container_width=True)

    return metrics_summary


def platform_evaluation_insights(df: pd.DataFrame) -> None:
    eligible = {}
    for platform, platform_df in df.groupby("Platform"):
        scoped = platform_df.dropna(subset=["rating_score", "lexicon_to_rating"])
        if not scoped.empty:
            eligible[platform] = scoped

    if not eligible:
        return

    st.header("Lexicon vs Rating Evaluation Summary")

    metrics_cache = {platform: _compute_platform_metrics(platform_df) for platform, platform_df in eligible.items()}

    if len(eligible) == 1:
        platform, platform_df = next(iter(eligible.items()))
        metrics_summary = _render_platform_evaluation(platform, platform_df, expand_visuals=True, metrics_summary=metrics_cache[platform])

        export_df = pd.DataFrame({"Metric": METRIC_ORDER, "Value": [metrics_summary[m] for m in METRIC_ORDER]})
        st.download_button(
            label="Download metrics (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{platform.lower().replace(' ', '_')}_lexicon_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )
        return

    tab = st.tabs(["All platforms"])[0]
    with tab:
        st.markdown("Both App Store and Play Store are selected – compare the visualisations side by side.")

        baseline_platform = sorted(metrics_cache.keys())[0]
        baseline_metrics = metrics_cache[baseline_platform]

        comparison_cols = st.columns(len(metrics_cache))
        for col, platform in zip(comparison_cols, sorted(metrics_cache.keys())):
            col.markdown(f"**{platform}**")
            for metric_name in METRIC_ORDER:
                metric_value = metrics_cache[platform][metric_name]
                if platform == baseline_platform:
                    col.metric(metric_name, f"{metric_value:.3f}")
                else:
                    delta = metric_value - baseline_metrics[metric_name]
                    col.metric(metric_name, f"{metric_value:.3f}", delta=f"{delta:+.3f}")

        columns = st.columns(len(eligible))
        for col, (platform, platform_df) in zip(columns, sorted(eligible.items())):
            with col:
                _render_platform_evaluation(platform, platform_df, metrics_summary=metrics_cache[platform])


def wordcloud_section(df: pd.DataFrame) -> None:
    st.header("WordCloud Explorer")

    if WordCloud is None:
        st.info("Install the `wordcloud` package to generate word clouds.")
        return

    selectable_columns = {
        "Final cleaned text (ulasan_bersih)": "ulasan_bersih",
        "Translated text": "translated_text",
        "Cleaned text": "cleaned_text",
        "Original text": "original_text",
    }

    available_columns = {label: col for label, col in selectable_columns.items() if col in df.columns}
    if not available_columns:
        st.warning("No suitable text columns available for generating a word cloud.")
        return

    with st.container():
        col_select, col_platform, col_sentiment, col_maxwords = st.columns([2, 1.5, 1.5, 1])
        text_label = col_select.selectbox("Text column", list(available_columns.keys()))
        text_column = available_columns[text_label]

        platforms = sorted(df["Platform"].unique())
        selected_platforms = col_platform.multiselect("Platforms", platforms, default=platforms)

        sentiment_options = ["All sentiments"] + sorted(df["sentiment_label"].dropna().unique())
        selected_sentiment = col_sentiment.selectbox("Sentiment", sentiment_options)

        max_words = col_maxwords.slider("Max words", min_value=50, max_value=400, value=200, step=50)

    scope = df[df["Platform"].isin(selected_platforms)] if selected_platforms else df.copy()
    if selected_sentiment != "All sentiments":
        scope = scope[scope["sentiment_label"] == selected_sentiment]

    text_series = scope[text_column].dropna().astype(str)
    combined_text = " ".join(text_series)

    if not combined_text.strip():
        st.warning("No text available with the selected filters. Adjust your selections to generate a word cloud.")
        return

    stop_words = set(STOPWORDS)
    custom_stopwords = st.text_input("Additional stopwords (comma separated)", help="Optional extra words to exclude from the cloud.")
    if custom_stopwords.strip():
        stop_words.update({word.strip().lower() for word in custom_stopwords.split(",") if word.strip()})

    try:
        cloud = WordCloud(
            width=1000,
            height=500,
            background_color="white",
            max_words=max_words,
            stopwords=stop_words,
            collocations=False,
        ).generate(combined_text)
    except ValueError:
        st.warning("Failed to generate the word cloud. Try expanding your text selection.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

    top_words = (
        pd.Series(cloud.words_)
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
        .rename(columns={"index": "word", 0: "weight"})
    )
    st.caption("Top words contributing to the cloud")
    st.dataframe(top_words, hide_index=True, use_container_width=True)


def preprocessing_explorer(df: pd.DataFrame) -> None:
    st.subheader("Pre-processing Explorer")
    st.markdown(
        "> **Pipeline insight:** Follow a single review as it moves from raw text through translation, cleaning, tokenisation, stopword removal, stemming, and finally sentiment labelling."
    )
    with st.expander("Try the pipeline on custom text", expanded=False):
        sample_input = st.text_area(
            "Type a sample review (max 500 characters)",
            value="",
            max_chars=500,
            help="See how the pre-processing steps transform your input.",
        )

        if sample_input.strip():
            preview = _simulate_preprocessing(sample_input)
            for step_name, content in preview.items():
                st.markdown(f"**{step_name}**")
                st.code(content, language="text")
        else:
            st.caption("Enter text above to generate the step-by-step transformation.")

    if df.empty:
        st.info("Adjust the filters to view pre-processing details.")
        return

    sorted_df = df.sort_values("review_date", ascending=False).reset_index(drop=True)
    selectable_columns = ["review_date", "Platform", "sentiment_label", "rating_score", "original_text"]
    preview_df = sorted_df[selectable_columns]

    index = st.slider(
        "Select review",
        min_value=0,
        max_value=len(preview_df) - 1,
        value=0,
        format="Row #%d",
    )

    selected_row = sorted_df.iloc[index]

    meta_cols = st.columns(4)
    meta_cols[0].metric("Review Date", selected_row["review_date"].strftime("%Y-%m-%d"))
    meta_cols[1].metric("Platform", selected_row["Platform"])
    meta_cols[2].metric("Rating", f"{selected_row['rating_score']:.0f}")
    meta_cols[3].metric("Sentiment", selected_row["sentiment_label"])

    steps = {
        "Original": selected_row.get("original_text", ""),
        "Translated": selected_row.get("translated_text", ""),
        "Cleaned": selected_row.get("cleaned_text", ""),
        "Tokenized": selected_row.get("tokenized_text", ""),
        "Stopword Removed": selected_row.get("stopword_removed_text", ""),
        "Stemmed": selected_row.get("stemmed_text", ""),
        "Final (ulasan_bersih)": selected_row.get("ulasan_bersih", ""),
    }

    for step_name, content in steps.items():
        if pd.isna(content) or str(content).strip() == "":
            continue
        st.markdown(f"**{step_name}**")
        st.code(str(content), language="text")

    token_counts = []
    for step_name, content in steps.items():
        if pd.isna(content) or str(content).strip() == "":
            continue
        if step_name == "Tokenized":
            try:
                values = pd.Series(ast.literal_eval(str(content)))
            except Exception:  # noqa: BLE001
                values = pd.Series(str(content).split())
            token_counts.append((step_name, len(values)))
        else:
            token_counts.append((step_name, len(str(content).split())))

    if token_counts:
        token_df = pd.DataFrame(token_counts, columns=["step", "count"])
        fig = px.line(token_df, x="step", y="count", markers=True, title="Token Count Across Steps")
        st.plotly_chart(fig, use_container_width=True)


def _simulate_preprocessing(text: str) -> Dict[str, str]:
    """Produce a lightweight preview of the cleaning pipeline for ad-hoc text."""

    original = text.strip()
    translated = _translate_to_indonesian(original)

    cleaned = _clean_text_notebook_style(translated)
    tokens = _tokenize_notebook_style(cleaned)
    tokens_no_stop = _remove_stopwords_notebook_style(tokens)
    stemmed_tokens = _stem_tokens_notebook_style(tokens_no_stop)
    approximated_final = " ".join(stemmed_tokens)
    sentiment_label, sentiment_score = _approximate_sentiment_label(stemmed_tokens)

    return {
        "Original": original or "(empty)",
        "Translated (notebook-style)": translated or "(empty)",
        "Cleaned": cleaned or "(empty)",
        "Tokenized": str(tokens) if tokens else "[]",
        "Stopword Removed": str(tokens_no_stop) if tokens_no_stop else "[]",
        "Stemmed": str(stemmed_tokens) if stemmed_tokens else "[]",
        "Final (approx. ulasan_bersih)": approximated_final or "(empty)",
        "Lexicon label (approx.)": sentiment_label,
        "Lexicon score (approx.)": f"{sentiment_score:+.2f}",
    }


def _load_lexicon_scores() -> Dict[str, Any]:
    global _LEXICON_CACHE
    if _LEXICON_CACHE is not None:
        return _LEXICON_CACHE

    scores: Dict[str, float] = {}
    max_len = 1
    file_names = ("positive.tsv", "negative.tsv")

    for filename in file_names:
        resolved: Optional[Path] = None
        for candidate in (DATA_DIR / filename, REPO_ROOT / filename):
            if candidate.exists():
                resolved = candidate
                break
        if resolved is None:
            continue

        try:
            df = pd.read_csv(resolved, sep="\t", dtype={"word": str, "weight": float})
        except Exception:
            continue

        for _, row in df.iterrows():
            word = str(row.get("word", "")).strip()
            if not word:
                continue
            try:
                weight = float(row.get("weight", 0.0))
            except Exception:
                continue
            key = word.lower()
            scores[key] = weight
            max_len = max(max_len, len(key.split()))

    _LEXICON_CACHE = {"scores": scores, "max_len": max_len}
    return _LEXICON_CACHE


def _translate_to_indonesian(text: str) -> str:
    if not text:
        return ""

    if _GoogleTranslator is not None:
        try:
            translator = _GoogleTranslator()
            return translator.translate(text, dest="id").text
        except Exception:
            pass

    return _basic_translate_to_indonesian(text)


def _clean_text_notebook_style(text: str) -> str:
    if text is None:
        return ""

    content = str(text)
    content = content.lower()
    content = content.replace("\n", " ")
    content = re.sub(r"\[.*?\]", "", content)
    content = re.sub(r"https?://\S+|www\.\S+", " ", content)
    content = content.translate(str.maketrans("", "", string.punctuation))
    content = re.sub(r"\d+", " ", content)
    content = re.sub(r"\s+", " ", content).strip()
    return content


def _ensure_nltk_resource(resource: str) -> None:
    if nltk is None:
        return
    try:
        if resource == "punkt":
            nltk.data.find("tokenizers/punkt")
        elif resource == "stopwords":
            nltk.data.find("corpora/stopwords")
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass


def _tokenize_notebook_style(text: str) -> List[str]:
    if not text:
        return []

    if _word_tokenize is not None:
        _ensure_nltk_resource("punkt")
        try:
            return _word_tokenize(text)
        except LookupError:
            return text.split()
    return text.split()


def _get_notebook_stopwords() -> set:
    global _SIMULATED_STOPWORDS
    if _SIMULATED_STOPWORDS is not None:
        return _SIMULATED_STOPWORDS

    stopword_set: set = set()

    if _nltk_stopwords is not None:
        _ensure_nltk_resource("stopwords")
        try:
            stopword_set.update(_nltk_stopwords.words("indonesian"))
        except LookupError:
            pass

    stopword_set.update(
        {
            "ga",
            "aplikasi",
            "disney",
            "nya",
            "gak",
            "aja",
            "tolong",
            "udah",
            "banget",
            "yg",
            "ya",
            "udh",
            "bagus",
            "jelek",
            "pas",
        }
    )

    if STOPWORDS:
        stopword_set.update(word.lower() for word in STOPWORDS)

    _SIMULATED_STOPWORDS = stopword_set
    return stopword_set


def _remove_stopwords_notebook_style(tokens: List[str]) -> List[str]:
    if not tokens:
        return []
    stopword_set = _get_notebook_stopwords()
    return [token for token in tokens if token not in stopword_set]


def _get_notebook_stemmer():
    global _SIMULATED_STEMMER
    if _SIMULATED_STEMMER is not None:
        return _SIMULATED_STEMMER
    if StemmerFactory is not None:
        try:
            _SIMULATED_STEMMER = StemmerFactory().create_stemmer()
            return _SIMULATED_STEMMER
        except Exception:
            _SIMULATED_STEMMER = None
    return None


def _stem_tokens_notebook_style(tokens: List[str]) -> List[str]:
    if not tokens:
        return []

    stemmer = _get_notebook_stemmer()
    if stemmer is not None:
        try:
            return [stemmer.stem(token) for token in tokens]
        except Exception:
            pass

    # Fallback simple suffix stripping if stemmer unavailable
    suffixes = ("nya", "lah", "kah", "pun", "kan", "an", "ing", "es", "s")
    stemmed: List[str] = []
    for token in tokens:
        stem = token
        for suffix in suffixes:
            if stem.endswith(suffix) and len(stem) - len(suffix) >= 3:
                stem = stem[: -len(suffix)]
                break
        stemmed.append(stem)
    return stemmed


def _approximate_sentiment_label(tokens: List[str]) -> Tuple[str, float]:
    if not tokens:
        return "Netral", 0.0

    lexicon = _load_lexicon_scores()
    scores = lexicon.get("scores", {}) if isinstance(lexicon, dict) else {}

    lowered_tokens = [token.lower() for token in tokens if token]

    if scores:
        max_len = int(lexicon.get("max_len", 1) or 1)
        total = 0.0
        idx = 0
        while idx < len(lowered_tokens):
            matched = False
            max_span = min(max_len, len(lowered_tokens) - idx)
            for span in range(max_span, 0, -1):
                phrase = " ".join(lowered_tokens[idx : idx + span])
                if phrase in scores:
                    total += scores[phrase]
                    idx += span
                    matched = True
                    break
            if not matched:
                idx += 1

        if total > 0:
            return "Positif", total
        if total < 0:
            return "Negatif", total
        return "Netral", 0.0

    # Fallback heuristic if lexicon tables are unavailable
    positive_words = {
        "bagus",
        "baik",
        "hebat",
        "mantap",
        "suka",
        "cinta",
        "senang",
        "puas",
        "lancar",
        "cepat",
        "mudah",
        "great",
        "awesome",
        "love",
        "nice",
        "amazing",
        "keren",
        "rekomendasi",
        "terbaik",
        "nyaman",
    }

    negative_words = {
        "buruk",
        "jelek",
        "benci",
        "parah",
        "lambat",
        "lemot",
        "macet",
        "crash",
        "error",
        "masalah",
        "payah",
        "susah",
        "hate",
        "bad",
        "worse",
        "terburuk",
        "mengecewakan",
        "ribet",
        "gagal",
    }

    score = 0
    for token in lowered_tokens:
        if token in positive_words:
            score += 1
        elif token in negative_words:
            score -= 1

    if score > 0:
        return "Positif", float(score)
    if score < 0:
        return "Negatif", float(score)
    return "Netral", 0.0


def _basic_translate_to_indonesian(text: str) -> str:
    """Fallback word-level translator for common English terms."""

    if not text:
        return ""

    dictionary = {
        "why": "kenapa",
        "what": "apa",
        "app": "aplikasi",
        "apps": "aplikasi",
        "always": "selalu",
        "crash": "macet",
        "crashes": "macet",
        "crashing": "macet",
        "bad": "buruk",
        "experience": "pengalaman",
        "hate": "benci",
        "love": "suka",
        "good": "bagus",
        "great": "hebat",
        "not": "tidak",
        "working": "berfungsi",
        "work": "bekerja",
        "please": "tolong",
        "fix": "perbaiki",
        "fun": "seru",
        "boring": "membosankan",
        "never": "tidak pernah",
        "loading": "memuat",
        "slow": "lambat",
        "fast": "cepat",
        "update": "pembaruan",
        "problem": "masalah",
        "error": "kesalahan",
        "issue": "masalah",
        "help": "bantuan",
        "great": "hebat",
        "amazing": "mengesankan",
        "terrible": "mengerikan",
        "awesome": "keren",
        "recommend": "merekomendasikan",
        "service": "layanan",
        "support": "dukungan",
    }

    tokens = re.findall(r"\w+|[^\w\s]", text)
    translated_tokens: List[str] = []
    for token in tokens:
        lower = token.lower()
        if lower in dictionary:
            translated = dictionary[lower]
            translated_tokens.append(translated)
        else:
            translated_tokens.append(token)

    return " ".join(translated_tokens)


def _prepare_text_and_labels(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    text_column = "ulasan_bersih" if "ulasan_bersih" in df.columns else None
    if text_column and df[text_column].dropna().str.strip().any():
        texts = df[text_column].fillna("").astype(str)
    else:
        fallback = "cleaned_text" if "cleaned_text" in df.columns else "original_text"
        texts = df[fallback].fillna("").astype(str)

    labels = df["sentiment_label"].fillna("Unknown").astype(str)
    return texts, labels


def _class_balance_ok(labels: Iterable[str], min_per_class: int = 3) -> bool:
    counts = pd.Series(labels).value_counts()
    if counts.empty:
        return False
    if len(counts) < 2:
        return False
    return (counts >= min_per_class).all()


def _format_classification_report(report_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df_report = pd.DataFrame(report_dict).transpose()
    df_report = df_report.rename(columns={"f1-score": "f1"})
    df_report = df_report[[col for col in ["precision", "recall", "f1", "support"] if col in df_report.columns]]
    numeric_cols = [col for col in df_report.columns if col != "support"]
    df_report[numeric_cols] = df_report[numeric_cols].applymap(lambda x: round(x, 3))
    return df_report


def _plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str) -> None:
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        title=title,
    )
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig, use_container_width=True)


def _get_model_classes(estimator: Any) -> Optional[List[str]]:
    if estimator is None:
        return None
    try:
        if hasattr(estimator, "classes_"):
            classes = getattr(estimator, "classes_")
            return list(classes) if classes is not None else None
        if hasattr(estimator, "named_steps"):
            named_steps = getattr(estimator, "named_steps")
            if isinstance(named_steps, dict) and named_steps:
                return _get_model_classes(list(named_steps.values())[-1])
        if hasattr(estimator, "steps"):
            steps = getattr(estimator, "steps")
            if isinstance(steps, list) and steps:
                return _get_model_classes(steps[-1][1])
    except Exception:  # noqa: BLE001
        return None
    return None


def _predict_with_pipeline(pipeline: Any, text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"label": None, "scores": None, "margin": None}

    prediction = pipeline.predict([text])[0]
    result["label"] = prediction

    classes = _get_model_classes(pipeline)

    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba([text])
            arr = np.asarray(proba)[0]
            if classes and arr.shape[0] == len(classes):
                df = (
                    pd.DataFrame({"class": classes, "score": arr})
                    .sort_values("score", ascending=False)
                    .reset_index(drop=True)
                )
                result["scores"] = df
        except Exception:  # noqa: BLE001
            pass

    if result["scores"] is None and hasattr(pipeline, "decision_function"):
        try:
            decision = pipeline.decision_function([text])
            arr = np.asarray(decision)
            if arr.ndim == 2 and classes and arr.shape[1] == len(classes):
                df = (
                    pd.DataFrame({"class": classes, "score": arr[0]})
                    .sort_values("score", ascending=False)
                    .reset_index(drop=True)
                )
                result["scores"] = df
            elif arr.ndim == 1 and classes and arr.shape[0] == len(classes):
                df = (
                    pd.DataFrame({"class": classes, "score": arr})
                    .sort_values("score", ascending=False)
                    .reset_index(drop=True)
                )
                result["scores"] = df
            elif arr.size == 1:
                result["margin"] = float(np.squeeze(arr))
        except Exception:  # noqa: BLE001
            pass

    return result


def _predict_with_bert_classifier(text: str, classifier: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    if torch is None:
        raise ImportError("transformers and torch are required for IndoBERT predictions.")

    model_name = config.get("model_name", "indobenchmark/indobert-base-p1")
    max_length = int(config.get("max_length", 128) or 128)

    tokenizer, transformer = _load_transformer(model_name)
    device = torch.device("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    transformer = transformer.to(device)

    inputs = tokenizer([text], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = transformer(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    prediction = classifier.predict(embedding)[0]

    result: Dict[str, Any] = {"label": prediction, "scores": None, "margin": None}
    classes = _get_model_classes(classifier)

    if hasattr(classifier, "predict_proba"):
        try:
            proba = classifier.predict_proba(embedding)
            arr = np.asarray(proba)[0]
            if classes and arr.shape[0] == len(classes):
                df = (
                    pd.DataFrame({"class": classes, "score": arr})
                    .sort_values("score", ascending=False)
                    .reset_index(drop=True)
                )
                result["scores"] = df
        except Exception:  # noqa: BLE001
            pass

    if result["scores"] is None and hasattr(classifier, "decision_function"):
        try:
            decision = classifier.decision_function(embedding)
            arr = np.asarray(decision)
            if arr.ndim == 2 and classes and arr.shape[1] == len(classes):
                df = (
                    pd.DataFrame({"class": classes, "score": arr[0]})
                    .sort_values("score", ascending=False)
                    .reset_index(drop=True)
                )
                result["scores"] = df
            elif arr.ndim == 1 and classes and arr.shape[0] == len(classes):
                df = (
                    pd.DataFrame({"class": classes, "score": arr})
                    .sort_values("score", ascending=False)
                    .reset_index(drop=True)
                )
                result["scores"] = df
            elif arr.size == 1:
                result["margin"] = float(np.squeeze(arr))
        except Exception:  # noqa: BLE001
            pass

    return result


@st.cache_resource(show_spinner=False)
def _train_tfidf_svm(texts: Tuple[str, ...], labels: Tuple[str, ...], random_state: int = 42) -> Dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        list(texts),
        list(labels),
        test_size=0.2,
        stratify=labels,
        random_state=random_state,
    )

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("svm", LinearSVC(class_weight="balanced", random_state=random_state)),
        ]
    )

    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "svm__C": [0.1, 1, 10],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    y_pred = grid.best_estimator_.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(labels)))

    cv_results = pd.DataFrame(grid.cv_results_)
    ngram_summary = (
        cv_results.groupby(cv_results["param_tfidf__ngram_range"].astype(str))["mean_test_score"].max().reset_index()
    )
    ngram_summary = ngram_summary.rename(
        columns={"param_tfidf__ngram_range": "ngram", "mean_test_score": "f1_macro"}
    )

    return {
        "best_params": grid.best_params_,
        "best_score": grid.best_score_,
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": sorted(set(labels)),
        "ngram_summary": ngram_summary,
        "model": grid.best_estimator_,
    }


@st.cache_resource(show_spinner=False)
def _load_transformer(model_name: str):
    if AutoTokenizer is None or AutoModel is None or torch is None:
        raise ImportError("transformers and torch are required for IndoBERT embeddings.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


@st.cache_data(show_spinner=False)
def _compute_bert_embeddings(
    texts: Tuple[str, ...],
    model_name: str,
    max_length: int = 128,
    batch_size: int = 16,
) -> np.ndarray:
    tokenizer, model = _load_transformer(model_name)
    device = torch.device("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    embeddings: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start : start + batch_size])
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)

    return np.vstack(embeddings)


@st.cache_resource(show_spinner=False)
def _train_bert_svm(
    texts: Tuple[str, ...],
    labels: Tuple[str, ...],
    model_name: str = "indobenchmark/indobert-base-p1",
    max_length: int = 128,
    batch_size: int = 16,
    random_state: int = 42,
) -> Dict[str, Any]:
    features = _compute_bert_embeddings(
        texts,
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        list(labels),
        test_size=0.2,
        stratify=labels,
        random_state=random_state,
    )

    estimator = LinearSVC(class_weight="balanced", random_state=random_state)
    param_grid = {"C": [0.1, 1, 5]}

    grid = GridSearchCV(
        estimator,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    y_pred = grid.best_estimator_.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(labels)))

    cv_results = pd.DataFrame(grid.cv_results_)[["param_C", "mean_test_score"]]
    cv_results = cv_results.rename(columns={"param_C": "C", "mean_test_score": "f1_macro"})

    return {
        "best_params": {
            "C": grid.best_params_["C"],
            "model_name": model_name,
            "max_length": max_length,
            "batch_size": batch_size,
        },
        "best_score": grid.best_score_,
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": sorted(set(labels)),
        "c_search": cv_results,
        "model": grid.best_estimator_,
    }


def prediction_playground() -> None:
    st.subheader("Sentiment Prediction Playground")

    models_catalog: List[Dict[str, Any]] = []

    tfidf_trained = st.session_state.get("tfidf_results_trained")
    if isinstance(tfidf_trained, dict) and tfidf_trained.get("model") is not None:
        display_name = st.session_state.get("tfidf_trained_label") or "TF-IDF (trained this session)"
        models_catalog.append(
            {
                "display": display_name,
                "type": "pipeline",
                "object": tfidf_trained["model"],
            }
        )

    registry = _get_pipeline_registry()
    for slot, entries in registry.items():
        for name, pipeline in entries.items():
            descriptor = f"Uploaded {slot} pipeline" if slot != "unknown" else "Uploaded pipeline"
            models_catalog.append(
                {
                    "display": f"{descriptor}: {name}",
                    "type": "pipeline",
                    "object": pipeline,
                }
            )

    bert_trained = st.session_state.get("bert_results_trained")
    if (
        isinstance(bert_trained, dict)
        and bert_trained.get("model") is not None
        and AutoTokenizer is not None
        and AutoModel is not None
        and torch is not None
    ):
        config = bert_trained.get("best_params", {})
        default_label = f"IndoBERT ({config.get('model_name', 'indobenchmark/indobert-base-p1')})"
        display_name = st.session_state.get("bert_trained_label") or default_label
        models_catalog.append(
            {
                "display": display_name,
                "type": "bert",
                "object": bert_trained["model"],
                "config": config,
            }
        )

    if not models_catalog:
        st.info("Train a model in this session or upload a saved pipeline to enable quick sentiment checks.")
        return

    options = [entry["display"] for entry in models_catalog]

    with st.form("sentiment_playground_form"):
        sample_text = st.text_area("Enter a sentence or review", height=120)
        selected_label = st.selectbox("Inference model", options)
        submitted = st.form_submit_button("Predict sentiment")

    if not submitted:
        return

    if not sample_text.strip():
        st.warning("Please enter some text before requesting a prediction.")
        return

    entry = next(item for item in models_catalog if item["display"] == selected_label)

    try:
        if entry["type"] == "pipeline":
            result = _predict_with_pipeline(entry["object"], sample_text)
        elif entry["type"] == "bert":
            result = _predict_with_bert_classifier(sample_text, entry["object"], entry.get("config", {}))
        else:
            st.error("Unsupported model selection.")
            return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed: {exc}")
        return

    st.success(f"Predicted sentiment: {result['label']}")

    if isinstance(result.get("scores"), pd.DataFrame):
        st.caption("Model confidence (sorted descending).")
        st.dataframe(result["scores"], hide_index=True, use_container_width=True)
    elif result.get("margin") is not None:
        st.caption(f"Decision margin: {result['margin']:.3f} (positive implies the predicted class is favoured).")


def model_performance_section(df: pd.DataFrame) -> None:
    st.header("Model Performance Comparison")
    st.markdown(
        "> **Model insight:** Compare TF-IDF + SVM against IndoBERT embeddings + SVM to understand classification quality across feature representations."
    )

    texts, labels = _prepare_text_and_labels(df)
    label_tuple = tuple(labels)
    if not _class_balance_ok(label_tuple):
        st.info("Need at least two sentiment classes with three or more samples each to run model evaluations.")
        return

    text_tuple = tuple(texts)
    data_signature = hash((text_tuple, label_tuple))
    if st.session_state.get("model_data_signature") != data_signature:
        st.session_state.pop("tfidf_results", None)
        st.session_state.pop("tfidf_results_loaded", None)
        st.session_state.pop("tfidf_results_trained", None)
        st.session_state.pop("tfidf_loaded_label", None)
        st.session_state.pop("tfidf_trained_label", None)
        st.session_state.pop("tfidf_active_source", None)
        st.session_state.pop("tfidf_results_repository", None)
        st.session_state.pop("tfidf_selected_loaded_label", None)
        st.session_state.pop("bert_results", None)
        st.session_state.pop("bert_results_loaded", None)
        st.session_state.pop("bert_results_trained", None)
        st.session_state.pop("bert_loaded_label", None)
        st.session_state.pop("bert_trained_label", None)
        st.session_state.pop("bert_loaded_config", None)
        st.session_state.pop("bert_trained_config", None)
        st.session_state.pop("bert_active_source", None)
        st.session_state.pop("bert_results_repository", None)
        st.session_state.pop("bert_selected_loaded_label", None)
        st.session_state["model_data_signature"] = data_signature

    _autoload_workspace_models()

    with st.expander("Import precomputed model results (JSON)", expanded=False):
        st.write(
            "Upload JSON files exported from the notebooks containing precomputed CV / report results to avoid retraining heavy models."
        )
        tfidf_file = st.file_uploader("TF-IDF results (JSON)", type=["json"], key="upload_tfidf")
        bert_file = st.file_uploader("IndoBERT results (JSON)", type=["json"], key="upload_bert")
        prefer_import = st.checkbox("Prefer imported results over retraining when present", value=True)

        if tfidf_file is not None:
            try:
                tfidf_payload = json.load(tfidf_file)
                if isinstance(tfidf_payload, dict) and "best_score" in tfidf_payload:
                    label = _format_result_label("tfidf", tfidf_file.name)
                    _store_loaded_result("tfidf", label, tfidf_payload)
                    st.success(f"Loaded TF-IDF results: {label}")
                else:
                    st.error("Invalid TF-IDF JSON structure: expected a dict containing 'best_score'.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to parse TF-IDF JSON: {exc}")

        if bert_file is not None:
            try:
                bert_payload = json.load(bert_file)
                if isinstance(bert_payload, dict) and "best_score" in bert_payload:
                    label = _format_result_label("bert", bert_file.name)
                    _store_loaded_result("bert", label, bert_payload)
                    st.session_state["bert_loaded_config"] = bert_payload.get("best_params", None)
                    st.success(f"Loaded IndoBERT results: {label}")
                else:
                    st.error("Invalid IndoBERT JSON structure: expected a dict containing 'best_score'.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to parse IndoBERT JSON: {exc}")

        workspace_candidates = _discover_exported_jsons()

        if workspace_candidates:
            st.markdown("**Workspace: detected exported JSON files**")
            for path in workspace_candidates:
                cols = st.columns([3, 1, 1])
                cols[0].write(Path(path).relative_to(REPO_ROOT))
                if cols[1].button("Preview", key=f"preview_ws_{path.stem}"):
                    try:
                        with open(path, "r", encoding="utf-8") as fh:
                            content = json.load(fh)
                        st.json(content)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Failed to read {path.name}: {exc}")

                if cols[2].button("Load into dashboard", key=f"load_ws_{path.stem}"):
                    try:
                        with open(path, "r", encoding="utf-8") as fh:
                            content = json.load(fh)

                        def _normalize_export(obj: dict) -> dict:
                            if isinstance(obj, dict) and ("best_score" in obj or "classification_report" in obj):
                                return obj

                            normalized: Dict[str, Any] = {}
                            grid = obj.get("grid") if isinstance(obj, dict) else None
                            if isinstance(grid, dict):
                                if "best_score" in grid:
                                    normalized["best_score"] = grid.get("best_score")
                                if "best_params" in grid:
                                    normalized["best_params"] = grid.get("best_params")

                                cv = grid.get("cv_results")
                                try:
                                    if isinstance(cv, dict):
                                        cv_df = pd.DataFrame(cv)
                                        if "param_tfidf__ngram_range" in cv_df.columns:
                                            ngram_summary = (
                                                cv_df.groupby(cv_df["param_tfidf__ngram_range"].astype(str))["mean_test_score"]
                                                .max()
                                                .reset_index()
                                                .rename(
                                                    columns={
                                                        "param_tfidf__ngram_range": "ngram",
                                                        "mean_test_score": "f1_macro",
                                                    }
                                                )
                                            )
                                            normalized["ngram_summary"] = ngram_summary

                                        if "param_C" in cv_df.columns:
                                            c_search = cv_df[["param_C", "mean_test_score"]].rename(
                                                columns={"param_C": "C", "mean_test_score": "f1_macro"}
                                            )
                                            normalized["c_search"] = c_search

                                        normalized.setdefault("cv_results", cv)
                                except Exception:
                                    normalized.setdefault("cv_results", cv)

                            if isinstance(obj, dict):
                                if "classification_report" in obj:
                                    normalized["classification_report"] = obj.get("classification_report")
                                if "confusion_matrix" in obj:
                                    normalized["confusion_matrix"] = obj.get("confusion_matrix")

                            return normalized if normalized else obj

                        norm = _normalize_export(content)

                        assigned = False
                        if isinstance(norm, dict):
                            best_params = norm.get("best_params", {})
                            if "ngram_summary" in norm or any(
                                key.startswith("tfidf") or "ngram" in str(key).lower() for key in best_params.keys()
                            ):
                                label = _format_result_label("tfidf", path.name)
                                _store_loaded_result("tfidf", label, norm)
                                st.success(f"Loaded TF-IDF results: {label}")
                                assigned = True
                            elif "c_search" in norm and isinstance(best_params, dict) and best_params.get("model_name"):
                                label = _format_result_label("bert", path.name)
                                _store_loaded_result("bert", label, norm)
                                st.session_state["bert_loaded_config"] = best_params
                                st.success(f"Loaded IndoBERT results: {label}")
                                assigned = True
                            elif "c_search" in norm or (
                                isinstance(norm.get("cv_results"), dict)
                                and any(col.startswith("param_kernel") for col in norm["cv_results"].keys())
                            ):
                                label = _format_result_label("tfidf", path.name)
                                _store_loaded_result("tfidf", label, norm)
                                st.success(f"Loaded TF-IDF results: {label}")
                                assigned = True

                        if not assigned:
                            label = _format_result_label("tfidf", path.name)
                            _store_loaded_result("tfidf", label, norm)
                            st.info(
                                f"Loaded {path.name} into TF-IDF slot (fallback). Use retrain buttons to recompute if needed."
                            )

                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Failed to load {path.name}: {exc}")

        st.session_state["prefer_import"] = prefer_import

    with st.expander("Load saved scikit-learn pipelines (joblib/pkl)", expanded=False):
        if joblib is None:
            st.info("Install the `joblib` package to enable loading persisted pipelines.")
        else:
            registry = _get_pipeline_registry()
            workspace_pipelines = _discover_saved_pipelines()
            if workspace_pipelines:
                st.markdown("**Workspace: detected saved pipelines**")
                for path in workspace_pipelines:
                    cols = st.columns([3, 1])
                    cols[0].write(Path(path).relative_to(REPO_ROOT))
                    if cols[1].button("Load", key=f"load_pipeline_{path.stem}"):
                        try:
                            pipeline_obj = joblib.load(path)
                            slot = _infer_pipeline_slot(path.name, pipeline_obj)
                            registry = _get_pipeline_registry()
                            registry[slot][path.name] = pipeline_obj
                            st.session_state["uploaded_pipelines"] = registry
                            st.success(f"Loaded `{path.name}` ({slot}) for inference.")
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Failed to load {path.name}: {exc}")

            uploaded = st.file_uploader(
                "Upload a saved pipeline for inference",
                type=["pkl", "joblib"],
                key="uploaded_pipeline_file",
                help="Upload a scikit-learn Pipeline (e.g., TF-IDF + SVM) exported with joblib.",
            )

            if uploaded is not None:
                label = _label_saved_pipeline(Path(uploaded.name))
                try:
                    pipeline_obj = joblib.load(uploaded)
                    slot = _infer_pipeline_slot(uploaded.name, pipeline_obj)
                    registry = _get_pipeline_registry()
                    if label not in registry[slot]:
                        registry[slot][label] = pipeline_obj
                        st.session_state["uploaded_pipelines"] = registry
                        st.success(f"Loaded `{label}` ({slot}) for interactive predictions.")
                    else:
                        st.info(f"Pipeline `{label}` is already loaded under the {slot} bucket.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to load pipeline: {exc}")

            registry = _get_pipeline_registry()
            summary_bits: List[str] = []
            for bucket, entries in registry.items():
                if entries:
                    summary_bits.append(f"{bucket}: {', '.join(entries.keys())}")
            if summary_bits:
                st.caption("Available pipelines → " + " | ".join(summary_bits))
                if st.button("Clear uploaded pipelines", key="clear_uploaded_pipelines"):
                    st.session_state["uploaded_pipelines"] = {"tfidf": {}, "bert": {}, "unknown": {}}
                    st.info("Cleared uploaded pipelines.")

    tfidf_tab, bert_tab = st.tabs(["SVM + TF-IDF", "SVM + IndoBERT"])

    with tfidf_tab:
        st.markdown("Run hyperparameter search across n-grams and C values for TF-IDF features.")
        st.caption(
            "The app uses `LinearSVC`, which applies a linear kernel under the hood. "
            "That is why the search space focuses on the regularisation strength (`C`) and TF-IDF n-gram ranges."
        )

        tfidf_trained = st.session_state.get("tfidf_results_trained")

        tfidf_options: List[str] = []
        if tfidf_trained is not None:
            tfidf_options.append("Trained results")
        tfidf_options.append("Train new model")

        active_source = st.session_state.get("tfidf_active_source")
        if active_source == "trained" and "Trained results" in tfidf_options:
            default_choice = "Trained results"
        else:
            default_choice = tfidf_options[0]

        selected_option = st.radio(
            "TF-IDF result source",
            tfidf_options,
            index=tfidf_options.index(default_choice),
            key="tfidf_source_choice",
        )

        if selected_option == "Trained results":
            if tfidf_trained is None:
                st.warning("Train the TF-IDF model first to view these results.")
            else:
                if st.session_state.get("tfidf_active_source") != "trained":
                    st.session_state["tfidf_results"] = tfidf_trained
                    st.session_state["tfidf_active_source"] = "trained"
        else:
            if st.button("Train / Refresh TF-IDF Model", key="train_tfidf"):
                with st.spinner("Training Linear SVM with TF-IDF features..."):
                    trained = _train_tfidf_svm(text_tuple, label_tuple)
                st.session_state["tfidf_results_trained"] = trained
                st.session_state["tfidf_results"] = trained
                st.session_state["tfidf_trained_label"] = f"Trained on {pd.Timestamp.now():%Y-%m-%d %H:%M}"
                st.session_state["tfidf_active_source"] = "trained"

        tfidf_results = st.session_state.get("tfidf_results")
        if tfidf_results:
            source_label = None
            if st.session_state.get("tfidf_active_source") == "trained":
                source_label = st.session_state.get("tfidf_trained_label")
            if source_label:
                st.caption(f"Viewing: {source_label}")

            st.success(
                f"Best F1 (macro): {tfidf_results.get('best_score', float('nan')):.3f} | Best Params: {tfidf_results.get('best_params', {})}"
            )

            if "classification_report" in tfidf_results and tfidf_results.get("classification_report"):
                st.subheader("Classification Report")
                try:
                    st.dataframe(
                        _format_classification_report(tfidf_results["classification_report"]),
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(f"Failed to render classification report: {exc}")
            else:
                st.warning(
                    "No classification report found in the selected TF-IDF results. "
                    "This JSON appears to contain only the grid/cv_results (hyperparameter search) but no held-out evaluation. "
                    "You can: (1) switch to 'Train new model' to compute evaluation on a local split, or (2) load an evaluation JSON that includes 'classification_report' and 'confusion_matrix'."
                )

            cm_values = tfidf_results.get("confusion_matrix")
            label_values = tfidf_results.get("labels")
            if cm_values is not None and label_values is not None:
                st.subheader("Confusion Matrix")
                try:
                    cm_array = np.asarray(cm_values)
                    if cm_array.size:
                        _plot_confusion_matrix(cm_array, list(label_values), "TF-IDF Model")
                    else:
                        st.info("Confusion matrix is empty for the selected TF-IDF results.")
                except Exception as exc:
                    st.error(f"Failed to render confusion matrix: {exc}")
            else:
                st.info("Confusion matrix not included in this export. Train locally to generate evaluation diagnostics.")

            if "ngram_summary" in tfidf_results and tfidf_results.get("ngram_summary") is not None:
                st.subheader("N-gram Search Summary")
                try:
                    ngram_summary = tfidf_results["ngram_summary"]
                    if hasattr(ngram_summary, "assign"):
                        st.dataframe(
                            ngram_summary.assign(f1_macro=lambda x: x["f1_macro"].round(3)),
                            use_container_width=True,
                        )
                    else:
                        df_summary = pd.DataFrame(ngram_summary)
                        if "f1_macro" in df_summary.columns:
                            df_summary["f1_macro"] = df_summary["f1_macro"].round(3)
                        st.dataframe(df_summary, use_container_width=True)
                except Exception as exc:
                    st.error(f"Failed to render n-gram summary: {exc}")
        else:
            st.info("Train the TF-IDF model to generate evaluation metrics.")

        tfidf_registry = _get_pipeline_registry().get("tfidf", {})
        if tfidf_registry:
            st.caption("TF-IDF pipelines available for inference: " + ", ".join(tfidf_registry.keys()))

    with bert_tab:
        if AutoTokenizer is None or AutoModel is None or torch is None:
            st.warning("Install transformers and torch to evaluate the IndoBERT-based model.")
            return

        st.markdown("Fine-tune C over CLS embeddings extracted from IndoBERT.")
        model_name = st.selectbox(
            "IndoBERT checkpoint",
            options=["indobenchmark/indobert-base-p1", "indobenchmark/indobert-base-p2"],
            index=0,
        )
        max_length = st.slider("Max token length", min_value=64, max_value=256, value=128, step=32)
        batch_size = st.selectbox("Batch size", options=[8, 16, 32], index=1)

        bert_config_tuple = (model_name, max_length, batch_size)

        bert_trained_config = st.session_state.get("bert_trained_config")
        if bert_trained_config is not None:
            trained_tuple = (
                bert_trained_config.get("model_name"),
                bert_trained_config.get("max_length"),
                bert_trained_config.get("batch_size"),
            )
            if trained_tuple != bert_config_tuple:
                st.session_state.pop("bert_results_trained", None)
                st.session_state.pop("bert_trained_config", None)
                st.session_state.pop("bert_trained_label", None)
                if st.session_state.get("bert_active_source") == "trained":
                    st.session_state["bert_active_source"] = "loaded" if st.session_state.get("bert_results_loaded") else None

        bert_trained = st.session_state.get("bert_results_trained")
        prefer_import = st.session_state.get("prefer_import", True)

        bert_options: List[str] = []
        if bert_trained is not None:
            bert_options.append("Trained results")
        bert_options.append("Train new model")

        bert_active_source = st.session_state.get("bert_active_source")
        if bert_active_source == "trained" and "Trained results" in bert_options:
            default_choice = "Trained results"
        else:
            default_choice = bert_options[0]

        selected_option = st.radio(
            "IndoBERT result source",
            bert_options,
            index=bert_options.index(default_choice),
            key="bert_source_choice",
        )

        if selected_option == "Trained results":
            if bert_trained is None:
                st.warning("Train the IndoBERT model first to view these results.")
            else:
                if st.session_state.get("bert_active_source") != "trained":
                    st.session_state["bert_results"] = bert_trained
                    st.session_state["bert_active_source"] = "trained"
        else:
            trigger_key = f"train_bert_{model_name}_{max_length}_{batch_size}"
            if st.button("Train / Refresh IndoBERT Model", key=trigger_key):
                with st.spinner("Embedding texts with IndoBERT and training Linear SVM..."):
                    trained = _train_bert_svm(
                        text_tuple,
                        label_tuple,
                        model_name=model_name,
                        max_length=max_length,
                        batch_size=batch_size,
                    )
                st.session_state["bert_results_trained"] = trained
                st.session_state["bert_results"] = trained
                st.session_state["bert_trained_config"] = {
                    "model_name": model_name,
                    "max_length": max_length,
                    "batch_size": batch_size,
                }
                st.session_state["bert_trained_label"] = (
                    f"Trained on {pd.Timestamp.now():%Y-%m-%d %H:%M} | {model_name}, max_len={max_length}, batch={batch_size}"
                )
                st.session_state["bert_active_source"] = "trained"

        bert_results = st.session_state.get("bert_results")
        if bert_results:
            source_label = None
            if st.session_state.get("bert_active_source") == "trained":
                source_label = st.session_state.get("bert_trained_label")
                trained_cfg = st.session_state.get("bert_trained_config")
                if trained_cfg:
                    st.caption(f"Training config: {trained_cfg}")
            if source_label:
                st.caption(f"Viewing: {source_label}")

            st.success(
                f"Best F1 (macro): {bert_results['best_score']:.3f} | Best Params: {bert_results['best_params']}"
            )
            st.subheader("Classification Report")
            st.dataframe(
                _format_classification_report(bert_results["classification_report"]),
                use_container_width=True,
            )

            cm_values = bert_results.get("confusion_matrix")
            label_values = bert_results.get("labels")
            if cm_values is not None and label_values is not None:
                st.subheader("Confusion Matrix")
                cm_array = np.asarray(cm_values)
                if cm_array.size:
                    _plot_confusion_matrix(cm_array, list(label_values), "IndoBERT Model")
                else:
                    st.info("Confusion matrix is empty for the IndoBERT results.")

            st.subheader("C Search Results")
            st.dataframe(
                bert_results["c_search"].assign(f1_macro=lambda x: x["f1_macro"].round(3)),
                use_container_width=True,
            )
        else:
            st.info("Configure parameters and train the IndoBERT model to view evaluation metrics.")

        bert_registry = _get_pipeline_registry().get("bert", {})
        if bert_registry:
            st.caption("IndoBERT pipelines available for inference: " + ", ".join(bert_registry.keys()))

def _parse_review_dates(series: pd.Series) -> pd.Series:
    """Parse date strings that may mix date-only and timestamp formats."""

    def _parse_val(value):
        if pd.isna(value):
            return pd.NaT
        if isinstance(value, pd.Timestamp):
            return value
        text = str(value).strip()
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return pd.to_datetime(text, format=fmt)
            except (ValueError, TypeError):
                continue
        try:
            return pd.to_datetime(text, errors="coerce")
        except Exception:
            return pd.NaT

    return series.apply(_parse_val)


def main() -> None:
    st.title("Disney+ Hotstar Review Insights")
    st.caption("Explore sentiment analysis outputs and inspect the text pre-processing pipeline.")
    st.markdown(CUSTOM_STYLES, unsafe_allow_html=True)

    try:
        df = load_dataset()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    filtered_df = apply_filters(df)

    if filtered_df.empty:
        st.warning("No reviews match the selected filters.")
        return

    draw_summary_metrics(filtered_df)

    st.subheader("Sentiment Overview")
    st.markdown(
        "> **Sentiment insight:** Use the filters to observe how sentiment mixes differ between App Store and Play Store reviews."
    )
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            sentiment_distribution(filtered_df)
        with col2:
            rating_distribution(filtered_df)

    st.subheader("Sentiment Trends")
    st.markdown(
        "> **Trend insight:** Use the tabs below to switch between month-over-month sentiment movements and the broader 2020-2022 vs. 2023-2025 period comparison."
    )
    sentiment_timeline(filtered_df)

    platform_evaluation_insights(filtered_df)

    st.subheader("Exploration Playground")
    tab_preprocess, tab_wordcloud, tab_prediction = st.tabs(
        ["Preprocessing Explorer", "Wordcloud Explorer", "Sentiment Prediction Playground"]
    )

    with tab_preprocess:
        preprocessing_explorer(filtered_df)

    with tab_wordcloud:
        wordcloud_section(filtered_df)

    with tab_prediction:
        prediction_playground()

    model_performance_section(filtered_df)

    st.subheader("Filtered Reviews")
    review_table = filtered_df.copy()
    if "review_date" in review_table.columns:
        review_table = review_table.sort_values("review_date", ascending=False)
    for col in ("predicted_sentiment_tfidf", "predicted_sentiment_bert"):
        if col not in review_table.columns:
            review_table[col] = np.nan

    desired_order = [
        "original_text",
        "translated_text",
        "cleaned_text",
        "tokenized_text",
        "stemmed_text",
        "ulasan_bersih",
        "rating_score",
        "sentiment_label",
        "predicted_sentiment_tfidf",
        "predicted_sentiment_bert",
    ]
    existing_columns = [col for col in desired_order if col in review_table.columns]
    review_display = review_table[existing_columns].copy()
    rename_map = {
        "original_text": "Original review",
        "translated_text": "Translated",
        "cleaned_text": "Cleaned",
        "tokenized_text": "Tokenized",
        "stemmed_text": "Stemmed",
        "ulasan_bersih": "Cleaned review",
        "rating_score": "Rating",
        "sentiment_label": "Sentiment label",
        "predicted_sentiment_tfidf": "Predicted (TF-IDF)",
        "predicted_sentiment_bert": "Predicted (IndoBERT)",
    }
    review_display = review_display.rename(columns={k: v for k, v in rename_map.items() if k in review_display.columns})
    st.dataframe(review_display, use_container_width=True)


if __name__ == "__main__":
    main()
