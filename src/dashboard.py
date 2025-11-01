from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:  # Optional heavy dependencies for transformer-based embeddings
    import torch
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover - handled gracefully in UI
    torch = None
    AutoModel = AutoTokenizer = None

st.set_page_config(
    page_title="Sentiment & Pre-processing Dashboard",
    page_icon="📊",
    layout="wide",
)

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
        selected_platforms = st.multiselect(
            "Platform",
            platform_choices,
            default=platform_choices,
        )

        sentiment_choices = sorted(df["sentiment_label"].dropna().unique())
        selected_sentiments = st.multiselect(
            "Sentiment",
            sentiment_choices,
            default=sentiment_choices,
        )

        min_date = df["review_date"].min().date()
        max_date = df["review_date"].max().date()
        selected_range = st.date_input(
            "Review Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        rating_min = int(df["rating_score"].min())
        rating_max = int(df["rating_score"].max())
        selected_ratings = st.slider(
            "Rating",
            min_value=rating_min,
            max_value=rating_max,
            value=(rating_min, rating_max),
        )

        keyword = st.text_input("Keyword search", help="Filter reviews containing specific words.")

    mask = df["Platform"].isin(selected_platforms)
    mask &= df["sentiment_label"].isin(selected_sentiments)
    mask &= df["rating_score"].between(*selected_ratings)

    start_date, end_date = selected_range
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
        st.metric("Total Reviews", f"{len(df):,}")
    with col2:
        st.metric("Average Rating", f"{df['rating_score'].mean():.2f}")
    with col3:
        sentiment_counts = df["sentiment_label"].value_counts()
        positive_share = sentiment_counts.get("Positif", 0) / max(len(df), 1) * 100
        st.metric("Positive Share", f"{positive_share:.1f}%")
    with col4:
        st.metric("Latest Review", df["review_date"].max().strftime("%Y-%m-%d"))


def sentiment_distribution(df: pd.DataFrame) -> None:
    sentiment_summary = (
        df.groupby(["sentiment_label", "Platform"], as_index=False)
        .size()
        .rename(columns={"size": "review_count"})
    )
    fig = px.bar(
        sentiment_summary,
        x="sentiment_label",
        y="review_count",
        color="Platform",
        barmode="group",
        title="Sentiment Distribution by Platform",
    )
    fig.update_layout(xaxis_title="Sentiment", yaxis_title="Number of Reviews")
    st.plotly_chart(fig, use_container_width=True)


def sentiment_timeline(df: pd.DataFrame) -> None:
    monthly = (
        df.groupby([pd.Grouper(key="review_date", freq="M"), "sentiment_label"])
        .size()
        .reset_index(name="review_count")
    )
    fig = px.line(
        monthly,
        x="review_date",
        y="review_count",
        color="sentiment_label",
        title="Monthly Sentiment Trend",
        markers=True,
    )
    st.plotly_chart(fig, use_container_width=True)


def rating_distribution(df: pd.DataFrame) -> None:
    category_levels = sorted(df["rating_score"].dropna().unique())
    summary = (
        df.groupby(["rating_score", "Platform"], as_index=False)
        .size()
        .rename(columns={"size": "review_count"})
    )

    fig = px.bar(
        summary,
        x="rating_score",
        y="review_count",
        color="Platform",
        barmode="group",
        category_orders={"rating_score": category_levels},
        labels={"rating_score": "Rating", "review_count": "Number of Reviews"},
        title="Rating Breakdown by Platform",
    )
    fig.update_layout(legend_title_text="Platform")
    fig.update_xaxes(type="category")
    st.plotly_chart(fig, use_container_width=True)


def _round_metric(value: float, digits: int = 3) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return float("nan")
        return round(float(value), digits)
    except Exception:
        return float("nan")


def _render_platform_evaluation(platform: str, platform_df: pd.DataFrame) -> None:
    st.subheader(platform)

    mae = np.mean(np.abs(platform_df["lexicon_to_rating"] - platform_df["rating_score"]))
    rmse = np.sqrt(np.mean((platform_df["lexicon_to_rating"] - platform_df["rating_score"]) ** 2))
    pearson = platform_df["rating_score"].corr(platform_df["lexicon_to_rating"], method="pearson")
    spearman = platform_df["rating_score"].corr(platform_df["lexicon_to_rating"], method="spearman")

    metrics_df = pd.DataFrame(
        {
            "Metric": ["MAE", "RMSE", "Pearson r", "Spearman ρ"],
            "Value": [_round_metric(mae), _round_metric(rmse), _round_metric(pearson), _round_metric(spearman)],
        }
    )
    st.dataframe(metrics_df, hide_index=True)

    scatter_fig = px.scatter(
        platform_df,
        x="rating_score",
        y="lexicon_to_rating",
        labels={"rating_score": "Original Rating", "lexicon_to_rating": "Lexicon Rating"},
        title=f"Original vs Lexicon Rating ({platform})",
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

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
        fig_counts.update_layout(xaxis_title="Lexicon-derived Rating", yaxis_title="Original Rating")
        fig_counts.update_traces(texttemplate="%{text}", textfont_size=14)

        percent = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0) * 100
        fig_pct = px.imshow(
            percent.values,
            x=percent.columns.astype(str),
            y=percent.index.astype(str),
            text_auto=True,
            color_continuous_scale="Viridis",
            labels={"x": "Lexicon-derived Rating", "y": "Original Rating"},
            title=f"Rating Consistency — % within Original Rating ({platform})",
        )
        fig_pct.update_layout(xaxis_title="Lexicon-derived Rating", yaxis_title="Original Rating")
        percent_text = percent.round(1).astype(str) + "%"
        fig_pct.data[0].text = percent_text.values
        fig_pct.update_traces(texttemplate="%{text}", textfont_size=12)

        st.plotly_chart(fig_counts, use_container_width=True)
        st.plotly_chart(fig_pct, use_container_width=True)
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
        st.plotly_chart(heatmap, use_container_width=True)


def platform_evaluation_insights(df: pd.DataFrame) -> None:
    eligible = {}
    for platform, platform_df in df.groupby("Platform"):
        scoped = platform_df.dropna(subset=["rating_score", "lexicon_to_rating"])
        if not scoped.empty:
            eligible[platform] = scoped

    if not eligible:
        return

    st.header("Lexicon vs Rating Evaluation Summary")

    tabs = st.tabs(list(eligible.keys()))
    for tab, (platform, platform_df) in zip(tabs, eligible.items()):
        with tab:
            _render_platform_evaluation(platform, platform_df)


def preprocessing_explorer(df: pd.DataFrame) -> None:
    st.subheader("Pre-processing Explorer")
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
    }


def model_performance_section(df: pd.DataFrame) -> None:
    st.header("Model Performance Comparison")

    texts, labels = _prepare_text_and_labels(df)
    label_tuple = tuple(labels)
    if not _class_balance_ok(label_tuple):
        st.info("Need at least three samples per sentiment class to run model evaluations.")
        return

    text_tuple = tuple(texts)
    data_signature = hash((text_tuple, label_tuple))
    if st.session_state.get("model_data_signature") != data_signature:
        st.session_state.pop("tfidf_results", None)
        st.session_state.pop("bert_results", None)
        st.session_state.pop("bert_results_config", None)
        st.session_state["model_data_signature"] = data_signature

    with st.expander("Import precomputed model results (JSON)", expanded=False):
        st.write(
            "Upload JSON files exported from the notebooks containing precomputed CV / report results to avoid retraining heavy models."
        )
        tfidf_file = st.file_uploader("TF-IDF results (JSON)", type=["json"], key="upload_tfidf")
        bert_file = st.file_uploader("IndoBERT results (JSON)", type=["json"], key="upload_bert")
        prefer_import = st.checkbox("Prefer imported results over retraining when present", value=True)

        tfidf_preview = None
        bert_preview = None

        if tfidf_file is not None:
            try:
                tfidf_preview = json.load(tfidf_file)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to parse TF-IDF JSON: {exc}")

        if bert_file is not None:
            try:
                bert_preview = json.load(bert_file)
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
                                st.session_state["tfidf_results"] = norm
                                st.success(f"Loaded {path.name} as TF-IDF results.")
                                assigned = True
                            elif "c_search" in norm and isinstance(best_params, dict) and best_params.get("model_name"):
                                st.session_state["bert_results"] = norm
                                st.session_state["bert_results_config"] = best_params
                                st.success(f"Loaded {path.name} as IndoBERT results.")
                                assigned = True
                            elif "c_search" in norm or (
                                isinstance(norm.get("cv_results"), dict)
                                and any(col.startswith("param_kernel") for col in norm["cv_results"].keys())
                            ):
                                st.session_state["tfidf_results"] = norm
                                st.success(f"Loaded {path.name} into TF-IDF slot (SVM/grid heuristic).")
                                assigned = True

                        if not assigned:
                            st.session_state["tfidf_results"] = norm
                            st.info(
                                f"Loaded {path.name} into TF-IDF slot (fallback). Use retrain buttons to recompute if needed."
                            )

                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Failed to load {path.name}: {exc}")

        if tfidf_preview is not None:
            st.markdown("**TF-IDF JSON preview**")
            try:
                st.json(tfidf_preview)
            except Exception:
                st.write(tfidf_preview)
            if st.button("Load TF-IDF results into dashboard", key="load_tfidf"):
                if isinstance(tfidf_preview, dict) and "best_score" in tfidf_preview:
                    st.session_state["tfidf_results"] = tfidf_preview
                    st.success("TF-IDF results loaded into session.")
                else:
                    st.error("Invalid TF-IDF JSON structure: expected a dict with 'best_score'.")

        if bert_preview is not None:
            st.markdown("**IndoBERT JSON preview**")
            try:
                st.json(bert_preview)
            except Exception:
                st.write(bert_preview)
            if st.button("Load IndoBERT results into dashboard", key="load_bert"):
                if isinstance(bert_preview, dict) and "best_score" in bert_preview:
                    st.session_state["bert_results"] = bert_preview
                    st.session_state["bert_results_config"] = bert_preview.get("best_params", None)
                    st.success("IndoBERT results loaded into session.")
                else:
                    st.error("Invalid IndoBERT JSON structure: expected a dict with 'best_score'.")

        st.session_state["prefer_import"] = prefer_import

    tfidf_tab, bert_tab = st.tabs(["SVM + TF-IDF", "SVM + IndoBERT"])

    with tfidf_tab:
        st.markdown("Run hyperparameter search across n-grams and C values for TF-IDF features.")
        if st.button("Train / Refresh TF-IDF Model", key="train_tfidf"):
            with st.spinner("Training Linear SVM with TF-IDF features..."):
                st.session_state["tfidf_results"] = _train_tfidf_svm(text_tuple, label_tuple)

        tfidf_results = st.session_state.get("tfidf_results")
        if tfidf_results:
            st.success(
                f"Best F1 (macro): {tfidf_results.get('best_score', float('nan')):.3f} | Best Params: {tfidf_results.get('best_params', {})}"
            )

            if "classification_report" in tfidf_results and tfidf_results.get("classification_report"):
                st.subheader("Classification Report")
                try:
                    st.dataframe(_format_classification_report(tfidf_results["classification_report"]))
                except Exception as exc:
                    st.error(f"Failed to render classification report: {exc}")
            else:
                st.warning(
                    "No classification report found in the loaded TF-IDF results. "
                    "This JSON appears to contain only the grid/cv_results (hyperparameter search) but no held-out evaluation. "
                    "You can: (1) press 'Train / Refresh TF-IDF Model' to compute evaluation on a local split, or (2) load an evaluation JSON that includes 'classification_report' and 'confusion_matrix'."
                )

            if "confusion_matrix" in tfidf_results and tfidf_results.get("confusion_matrix") and "labels" in tfidf_results:
                st.subheader("Confusion Matrix")
                try:
                    _plot_confusion_matrix(tfidf_results["confusion_matrix"], tfidf_results["labels"], "TF-IDF Model")
                except Exception as exc:
                    st.error(f"Failed to render confusion matrix: {exc}")
            else:
                if "cv_results" in tfidf_results and tfidf_results.get("cv_results"):
                    st.subheader("Cross-validation results (preview)")
                    try:
                        st.write("Grid / CV results are present but no confusion matrix:")
                        st.json(tfidf_results.get("cv_results"))
                    except Exception:
                        st.write(tfidf_results.get("cv_results"))

            if "ngram_summary" in tfidf_results and tfidf_results.get("ngram_summary") is not None:
                st.subheader("N-gram Search Summary")
                try:
                    ngram_summary = tfidf_results["ngram_summary"]
                    if hasattr(ngram_summary, "assign"):
                        st.dataframe(ngram_summary.assign(f1_macro=lambda x: x["f1_macro"].round(3)))
                    else:
                        df_summary = pd.DataFrame(ngram_summary)
                        if "f1_macro" in df_summary.columns:
                            df_summary["f1_macro"] = df_summary["f1_macro"].round(3)
                        st.dataframe(df_summary)
                except Exception as exc:
                    st.error(f"Failed to render n-gram summary: {exc}")
        else:
            if st.session_state.get("prefer_import"):
                st.info("Load precomputed TF-IDF JSON results or click the button above to train.")
            else:
                st.info("Click the button above to train the TF-IDF + SVM model.")

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

        bert_config = (model_name, max_length, batch_size)
        if st.session_state.get("bert_results_config") != bert_config:
            st.session_state.pop("bert_results", None)

        trigger_key = f"train_bert_{model_name}_{max_length}_{batch_size}"
        if st.button("Train / Refresh IndoBERT Model", key=trigger_key):
            with st.spinner("Embedding texts with IndoBERT and training Linear SVM..."):
                st.session_state["bert_results"] = _train_bert_svm(
                    text_tuple,
                    label_tuple,
                    model_name=model_name,
                    max_length=max_length,
                    batch_size=batch_size,
                )
                st.session_state["bert_results_config"] = bert_config

        bert_results = st.session_state.get("bert_results")
        if bert_results:
            st.success(
                f"Best F1 (macro): {bert_results['best_score']:.3f} | Best Params: {bert_results['best_params']}"
            )
            st.subheader("Classification Report")
            st.dataframe(_format_classification_report(bert_results["classification_report"]))
            st.subheader("Confusion Matrix")
            _plot_confusion_matrix(bert_results["confusion_matrix"], bert_results["labels"], "IndoBERT Model")
            st.subheader("C Search Results")
            st.dataframe(bert_results["c_search"].assign(f1_macro=lambda x: x["f1_macro"].round(3)))
        else:
            if st.session_state.get("prefer_import"):
                st.info("Load a precomputed IndoBERT JSON or configure parameters and run the model.")
            else:
                st.info("Configure parameters and run the IndoBERT model to see results.")


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

    col1, col2 = st.columns(2)
    with col1:
        sentiment_distribution(filtered_df)
    with col2:
        rating_distribution(filtered_df)

    sentiment_timeline(filtered_df)

    platform_evaluation_insights(filtered_df)

    preprocessing_explorer(filtered_df)

    model_performance_section(filtered_df)

    st.subheader("Filtered Reviews")
    st.dataframe(
        filtered_df[
            [
                "review_date",
                "Platform",
                "rating_score",
                "sentiment_label",
                "original_text",
                "translated_text",
                "ulasan_bersih",
            ]
        ].sort_values("review_date", ascending=False),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
