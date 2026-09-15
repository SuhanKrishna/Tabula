# ◰ Tabula

**Any CSV. No assumptions.**

A general-purpose tabular data studio built with Streamlit. Upload any CSV and get adaptive profiling, a data-quality dashboard, exploratory analysis, leakage-safe preprocessing, model training, and scoring of new data.

The name comes from *tabula rasa*, the blank slate. No column name is hardcoded anywhere in the codebase. Column types, encoding strategy, model selection, and every metric are inferred at runtime from your file.

```bash
pip install -r requirements.txt
streamlit run app.py
```

No CSV to hand? Three synthetic sample datasets are built in, covering classification, regression, and mixed column types.

---

## Why

Most EDA and modeling dashboards are quietly hardcoded to a single dataset. Swap the CSV and they break, or worse, they keep running and silently produce nonsense.

Tabula inverts that. You upload a file, pick a target column, and everything downstream adapts.

---

## The six sections

### ◰ Dashboard

The landing view once data is loaded. Answers "what am I dealing with, and should I trust it?" at a glance.

- **KPI row** — rows, columns, missing cells, duplicate rows, memory footprint, and an overall data-quality score.
- **Quality score (0–100)** — a transparent deduction model, not a black box. Every point lost traces to a named penalty: missing data, duplicate rows, constant columns, columns over 50% missing, duplicated column pairs, and small-sample penalties. Graded A through D.
- **Column composition** — donut chart of the detected type breakdown.
- **Prioritised issue feed** — critical / warning / info items with concrete detail, sorted by severity.
- **Target snapshot** — task type, class count and imbalance ratio, or mean and skew for regression.
- **Live model summary** — once trained: best model, headline metrics, model comparison, and top feature drivers.

### Data & Profiling

- **Robust CSV parsing** — comma, semicolon, tab and pipe delimiters across UTF-8 and Latin-1, with a binary-content guard that rejects non-text uploads before pandas mangles them.
- **Statistical column type inference** — `numeric`, `categorical`, `high_cardinality_categorical`, `datetime`, `boolean`, `id_like`. Based on dtype, cardinality and parse rate. Never on the column's name.
- **Numeric summary tab** — mean, quartiles, skew, and IQR-based outlier counts per column.
- **Target selection** with automatic classification-vs-regression detection, overridable when the inference is wrong.
- **Leakage warnings at selection time** (see below).

### Explore

Five tabs, every chart driven by detected columns.

- **Target** — distribution plus a feature drilldown that picks the right chart for the type pairing (box, stacked composition, scatter with OLS trendline).
- **Missing data** — per-column missingness and a co-occurrence matrix showing which columns go missing *together*.
- **Correlation** — Pearson or Spearman, selectable columns, with automatic multicollinearity warnings.
- **Distributions** — histograms and category bars with live skew and outlier annotations.
- **Relationships** — free-form X/Y/colour explorer across scatter, box, line, violin, and density heatmap.

Every narrative insight is computed from real statistics via f-strings. Nothing is a hardcoded claim about what a column "means."

### Preprocessing

- Configurable imputation — numeric (median / mean / constant), categorical (mode / `"Missing"` category).
- One-hot for low-cardinality categoricals; **frequency or ordinal encoding** for high-cardinality ones, with an adjustable threshold.
- **Encoded matrix width estimate** before you train, with a warning when one-hot encoding is about to explode.
- Optional `StandardScaler`, optional duplicate-row removal, manual column exclusion.
- **Per-column routing table** showing exactly which branch each column takes.
- Exportable config as JSON.

Implemented as an sklearn `ColumnTransformer` + `Pipeline` **fit only on the training split**.

### Modeling & Results

| Task | Models | Metrics |
| --- | --- | --- |
| Classification | RandomForest, XGBoost, optional LogisticRegression | Accuracy, macro F1, precision, recall, full report, count and row-normalised confusion matrices, ROC/AUC, precision-recall/AP, threshold tuning |
| Regression | RandomForest, XGBoost, optional LinearRegression | R², RMSE, MAE, predicted-vs-actual, residuals vs predicted, residual distribution |

- **Cross-validation** option with automatic fold reduction when a class is too small, plus a warning when the single-split score and CV mean diverge by more than 0.10.
- **Interactive decision threshold tuning** for binary targets, with precision and recall updating live.
- **Permutation importance** as an alternative to impurity importance, which is biased toward high-cardinality features.
- **Balanced class weights** for imbalanced targets; **log transform** for skewed regression targets.
- Feature importances mapped back through the `ColumnTransformer` to readable names.
- Export the pipeline, test predictions, or the metrics report.

### Score New Data

Apply the trained pipeline to a fresh CSV.

- Validates required columns and lists anything missing.
- Downloadable blank template with the expected schema.
- Replays the exact feature engineering from training via a shared `resolve_feature_roles` function, so training and scoring cannot drift apart.
- Handles unseen categories and new missing values.
- Prediction confidence scores and a low-confidence row count for classification.
- Inverts the log transform automatically when one was applied.
- If the uploaded file happens to contain the target column, reports accuracy against it.

---

## Design notes

**Leakage detection.** Numeric features are checked for absolute correlation above 0.95 with the encoded target; categorical features are checked for whether their categories map near-perfectly onto target classes. Either pattern usually means the column encodes the outcome. Flagged at target-selection time and on the dashboard, before you waste a training run believing a 0.99 F1.

**Datetime handling.** Date columns expand into year / month / day / day-of-week numeric features rather than being dropped or one-hot encoded into thousands of columns.

**Custom `FrequencyEncoder`.** A small sklearn-compatible transformer encoding each category by its relative frequency in the *training* data. Unseen values map to `0`. Implements `get_feature_names_out`, so charts show `region_code_freq` rather than an unlabeled index.

**Caching.** `@st.cache_data` for loading and profiling, `@st.cache_resource` for trained pipelines, keyed on an explicit SHA-256 hash of data + target + preprocessing config + hyperparameters. Large objects are underscore-prefixed so Streamlit doesn't redundantly rehash them.

**Graceful degradation.** `xgboost` is optional; if absent the app warns and skips those models. Every edge case surfaces through `st.error` / `st.warning` rather than a traceback.

---

## Two bugs worth documenting

Both were caught during testing, and both were the silent kind.

**1. pandas 3.0 string dtype.** Plain text columns no longer report `dtype == object` under pandas 3.0's native string dtype. This silently broke datetime detection: date columns fell through to the ID-like branch and were dropped from modeling entirely. No error, no crash, just quietly worse predictions. The check now tests `is_string_dtype` and `is_object_dtype` together.

**2. Delimiter sniffing corrupts single-column files.** Falling back to `pd.read_csv(sep=None)` on a one-column file lets the sniffer pick a letter from the header as the delimiter. A file whose header reads `only` sniffs to `sep="y"` and parses into a mangled `onl` column plus a phantom empty one. The fallback chain now tries a plain default read before sniffing, so genuine single-column files survive intact.

Neither raised an exception. Both would have shipped.

---

## Testing

Validated against synthetic datasets built to be deliberately hostile:

- Classification, regression, and mixed-type targets
- Both high-cardinality encoding strategies, and a 400-category explosion case
- Native booleans and string booleans (`yes`/`no`, `true`/`false`)
- Datetime-only feature sets
- Heavy missingness, target NaNs, duplicate rows and duplicate columns
- Rare classes triggering the non-stratified split fallback
- All-object dataframes and 6-row datasets
- Injected leakage columns, both numeric and categorical
- Scoring files containing unseen categories and new missing values
- Comma / semicolon / tab / pipe delimiters, UTF-8 and Latin-1, quoted fields, single-column files, and binary payloads

Plus a live headless boot of the Streamlit server.

Verified on streamlit 1.63, scikit-learn 1.8, xgboost 3.4, plotly 7.0, pandas 3.0.2.

---

## Project structure

```
.
├── app.py              # Single-file Streamlit application
├── requirements.txt
└── README.md
```

---

## License

MIT
