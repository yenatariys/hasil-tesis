# Disney+ Hotstar Sentiment Analysis Dashboard: Code Logic Overview with References & Explanations

This document explains the main logic and structure of the Streamlit dashboard in `dashboard.py`, with direct code references and explanations for each step.

---

## 1. Project Structure
- **dashboard.py**: Main dashboard logic and UI.
- **data/**: Contains review datasets and lexicon files.
- **outputs/**: Stores model results and trained pipelines.

---

## 2. Dashboard Initialization
- **Set page config:**
  ```python
  st.set_page_config(
      page_title="Disney+ Hotstar Sentiment Analysis Dashboard",
      page_icon="📊",
      layout="wide",
  )
  ```
  *Explanation*: Sets the dashboard's title, icon, and layout for a wide, modern look.

- **Custom styles:**
  ```python
  CUSTOM_STYLES = """
  <style>
  ...
  </style>
  """
  st.markdown(CUSTOM_STYLES, unsafe_allow_html=True)
  ```
  *Explanation*: Adds custom CSS for metrics and buttons to improve UI clarity and aesthetics.

---

## 3. Data Loading & Preparation
- **Load datasets:**
  ```python
  @st.cache_data(show_spinner=False)
  def load_dataset() -> pd.DataFrame:
      ...
      app_df = pd.read_csv(APP_FILE)
      play_df = pd.read_csv(PLAY_FILE)
      ...
      combined = pd.concat([app_df, play_df], ignore_index=True, sort=False)
      ...
      return combined
  ```
  *Explanation*: Loads and merges App Store and Play Store reviews, standardizes columns, parses dates, and cleans data for analysis.

- **Apply filters:**
  ```python
  def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
      with st.sidebar:
          ...
          selected_platforms = st.multiselect(...)
          selected_sentiments = st.multiselect(...)
          selected_range = st.date_input(...)
          selected_ratings = st.slider(...)
          keyword = st.text_input(...)
      ...
      return df.loc[mask]
  ```
  *Explanation*: Provides sidebar controls for filtering reviews by platform, sentiment, date, rating, and keyword.

---

## 4. Main Dashboard Sections
### a. Summary Metrics
- **draw_summary_metrics(df):**
  ```python
  def draw_summary_metrics(df: pd.DataFrame) -> None:
      col1, col2, col3, col4 = st.columns(4)
      ...
      st.metric("🧾 Total Reviews", f"{len(df):,}")
      st.metric("⭐ Average Rating", f"{df['rating_score'].mean():.2f}")
      ...
  ```
  *Explanation*: Displays key metrics (total reviews, average rating, positive share, latest review) for quick insights.

### b. Sentiment Distribution & Trends
- **sentiment_distribution(df):**
  ```python
  def sentiment_distribution(df: pd.DataFrame) -> None:
      ...
      fig = px.bar(...)
      st.plotly_chart(fig, use_container_width=True)
  ```
  *Explanation*: Visualizes sentiment breakdown by platform using a stacked bar chart.

- **sentiment_timeline(df):**
  ```python
  def sentiment_timeline(df: pd.DataFrame) -> None:
      ...
      fig = px.line(...)
      st.plotly_chart(fig, use_container_width=True)
  ```
  *Explanation*: Shows monthly sentiment trends and period comparisons to reveal changes over time.

### c. Rating Distribution
- **rating_distribution(df):**
  ```python
  def rating_distribution(df: pd.DataFrame) -> None:
      ...
      fig = px.bar(...)
      st.plotly_chart(fig, use_container_width=True)
  ```
  *Explanation*: Presents rating distribution by platform for further context on user feedback.

### d. Lexicon Evaluation
- **platform_evaluation_insights(df):**
  ```python
  def platform_evaluation_insights(df: pd.DataFrame) -> None:
      ...
      metrics_cache = {platform: _compute_platform_metrics(platform_df) for platform, platform_df in eligible.items()}
      ...
      st.download_button(...)
  ```
  *Explanation*: Compares lexicon-derived sentiment scores to actual ratings, computes metrics, and allows CSV export.

- **_compute_platform_metrics(platform_df):**
  ```python
  def _compute_platform_metrics(platform_df: pd.DataFrame) -> Dict[str, float]:
      ...
      return metrics
  ```
  *Explanation*: Calculates MAE, RMSE, Pearson r, and Spearman ρ for platform-level evaluation.

### e. WordCloud Explorer
- **wordcloud_section(df):**
  ```python
  def wordcloud_section(df: pd.DataFrame) -> None:
      ...
      cloud = WordCloud(...).generate(combined_text)
      ...
      st.pyplot(fig)
      ...
      st.dataframe(top_words, hide_index=True, use_container_width=True)
  ```
  *Explanation*: Generates word clouds for selected text columns, platforms, and sentiments, highlighting frequent terms.

### f. Preprocessing Explorer
- **preprocessing_explorer(df):**
  ```python
  def preprocessing_explorer(df: pd.DataFrame) -> None:
      ...
      sample_input = st.text_area(...)
      ...
      for step_name, content in steps.items():
          st.markdown(f"**{step_name}**")
          st.code(str(content), language="text")
      ...
  ```
  *Explanation*: Shows each step of the text preprocessing pipeline for a selected review or custom input.

### g. Sentiment Prediction Playground
- **prediction_playground():**
  ```python
  def prediction_playground() -> None:
      ...
      with st.form("sentiment_playground_form"):
          ...
      st.success(f"Predicted sentiment: {result['label']}")
      ...
  ```
  *Explanation*: Lets users test trained models or uploaded pipelines on custom text, displaying predictions and confidence.

### h. Model Performance Comparison
- **model_performance_section(df):**
  ```python
  def model_performance_section(df: pd.DataFrame) -> None:
      ...
      tab_compare, tab_tfidf, tab_bert = st.tabs([...])
      ...
      st.dataframe(_format_classification_report(tfidf_results["classification_report"]), ...)
      ...
      _plot_confusion_matrix(cm_array, list(tfidf_results["labels"]), "TF-IDF Model")
      ...
  ```
  *Explanation*: Compares TF-IDF and IndoBERT models side-by-side, showing scores, params, reports, and confusion matrices.

### i. Filtered Reviews Table
- **Display table:**
  ```python
  review_display = review_table[existing_columns].copy()
  ...
  st.dataframe(review_display, use_container_width=True)
  ```
  *Explanation*: Displays the filtered reviews with all relevant columns and predicted sentiments for user inspection.

---

## 5. Model Training & Loading
- **_train_tfidf_svm(texts, labels):**
  ```python
  def _train_tfidf_svm(texts: Tuple[str, ...], labels: Tuple[str, ...], ...):
      ...
      grid = GridSearchCV(...)
      grid.fit(X_train, y_train)
      ...
      return {...}
  ```
  *Explanation*: Trains a TF-IDF + SVM model, performs grid search, and returns metrics and the trained model.

- **_train_bert_svm(texts, labels, ...):**
  ```python
  def _train_bert_svm(...):
      ...
  ```
  *Explanation*: Trains an IndoBERT + SVM model using transformer embeddings for advanced sentiment classification.

- **_get_pipeline_registry():**
  ```python
  def _get_pipeline_registry() -> Dict[str, Dict[str, Any]]:
      ...
      return registry
  ```
  *Explanation*: Manages uploaded pipelines for inference and model comparison.

- **_normalize_loaded_results(content):**
  ```python
  def _normalize_loaded_results(content: Dict[str, Any]) -> Dict[str, Any]:
      ...
      return content
  ```
  *Explanation*: Standardizes loaded model results for consistent display and comparison.

---

## 6. Utility Functions
- **_prepare_text_and_labels(df):**
  ```python
  def _prepare_text_and_labels(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
      ...
      return texts, labels
  ```
  *Explanation*: Extracts texts and labels from the DataFrame for model training and evaluation.

- **_plot_confusion_matrix(cm, labels, title):**
  ```python
  def _plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str) -> None:
      ...
      st.plotly_chart(fig, use_container_width=True)
  ```
  *Explanation*: Visualizes confusion matrices for model evaluation.

- **_format_classification_report(report_dict):**
  ```python
  def _format_classification_report(report_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
      ...
      return df_report
  ```
  *Explanation*: Formats classification reports for clear tabular display in the dashboard.

---

## 7. Main Entry Point
- **main():**
  ```python
  def main() -> None:
      ...
      filtered_df = apply_filters(df)
      ...
      draw_summary_metrics(filtered_df)
      ...
      sentiment_distribution(filtered_df)
      ...
      platform_evaluation_insights(filtered_df)
      ...
      tab_preprocess, tab_wordcloud, tab_prediction = st.tabs([...])
      ...
      model_performance_section(filtered_df)
      ...
      st.dataframe(review_display, use_container_width=True)
  ```
  *Explanation*: Orchestrates the dashboard workflow—loads data, applies filters, renders all sections, and manages user interactions.

---

**Note:**
- Each section above is directly referenced by the function or code block in `dashboard.py` and explained for clarity.
- For full details, see the actual function implementations in the code.
- This document is designed to help you quickly map dashboard features to their code logic and understand their purpose.

For further details, see inline comments in `dashboard.py` or request a deeper dive into any function or section.
