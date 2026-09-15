import io
import json
import time
import pickle
import hashlib
 
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    r2_score, mean_squared_error, mean_absolute_error,
)
 
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:  # xgboost is optional; the app degrades gracefully without it
    XGBOOST_AVAILABLE = False
 
 
APP_NAME = "Tabula"
APP_TAGLINE = "Any CSV. No assumptions."
 
# =====================================================================================
# PAGE CONFIG & THEME
# =====================================================================================
st.set_page_config(
    page_title=f"{APP_NAME} - {APP_TAGLINE}",
    page_icon="\u25F0",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
CUSTOM_CSS = """
<style>
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; }
 
    .tabula-hero {
        padding: 1.4rem 1.6rem;
        border-radius: 14px;
        background: linear-gradient(120deg, #1e3a5f 0%, #2d6a7a 100%);
        color: #ffffff;
        margin-bottom: 1.4rem;
    }
    .tabula-hero h1 { margin: 0; font-size: 2.0rem; letter-spacing: -0.5px; color: #ffffff; }
    .tabula-hero p  { margin: 0.25rem 0 0 0; opacity: 0.85; font-size: 0.98rem; }
 
    .kpi-card {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        height: 100%;
    }
    .kpi-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.6px; opacity: 0.65; }
    .kpi-value { font-size: 1.7rem; font-weight: 650; line-height: 1.25; margin-top: 0.2rem; }
    .kpi-sub   { font-size: 0.8rem; opacity: 0.6; }
 
    .issue-row {
        border-left: 4px solid var(--sev-color, #888);
        background: rgba(128, 128, 128, 0.07);
        padding: 0.6rem 0.9rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    .issue-title { font-weight: 600; font-size: 0.92rem; }
    .issue-detail { font-size: 0.83rem; opacity: 0.75; }
 
    .step-done    { color: #1a9850; font-weight: 600; }
    .step-pending { opacity: 0.45; }
 
    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
</style>
"""
 
SEVERITY_COLORS = {"critical": "#d73027", "warning": "#fc8d59", "info": "#4575b4"}
 
 
def render_hero(title, subtitle):
    st.markdown(
        f'<div class="tabula-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )
 
 
def kpi_card(label, value, sub=""):
    return (
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>'
    )
 
 
# =====================================================================================
# CUSTOM TRANSFORMER - frequency encoding for high-cardinality categoricals
# =====================================================================================
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encodes each categorical column as the relative frequency of its value in the
    TRAINING data (fit only on train, so no leakage). Unseen values map to 0.
    """
 
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.n_features_in_ = X.shape[1]
        self.freq_maps_ = []
        for i in range(X.shape[1]):
            vc = X.iloc[:, i].astype(str).value_counts(normalize=True)
            self.freq_maps_.append(vc.to_dict())
        return self
 
    def transform(self, X):
        X = pd.DataFrame(X)
        out = np.zeros((len(X), X.shape[1]))
        for i in range(X.shape[1]):
            out[:, i] = X.iloc[:, i].astype(str).map(self.freq_maps_[i]).fillna(0.0).values
        return out
 
    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array([f"{f}_freq" for f in input_features])
        return np.array([f"feature_{i}_freq" for i in range(self.n_features_in_)])
 
 
# =====================================================================================
# DATA LOADING
# =====================================================================================
@st.cache_data(show_spinner=False)
def load_data(file_bytes, filename):
    """Robustly parse an uploaded CSV across delimiters and encodings.
    Returns (dataframe, encoding_used, delimiter_used).
 
    Resolution order matters. Pandas' own sniffer (sep=None) is tried LAST, because
    on a genuine single-column file it will happily guess a letter from the header as
    the delimiter - e.g. "only\\n1\\n2" sniffs to sep="y" and yields a mangled
    'onl' column plus a phantom empty one. Trying the plain default read first means
    single-column files parse correctly instead of being silently corrupted.
    """
    delimiters = [",", ";", "\t", "|"]
    encodings = ["utf-8", "latin-1"]
    last_err = None
 
    # Reject binary payloads before pandas tries to coerce them into a frame
    sample = file_bytes[:4096]
    if sample:
        printable = sum(1 for b in sample if 9 <= b <= 13 or 32 <= b <= 126 or b >= 128)
        if printable / len(sample) < 0.85:
            raise ValueError("This file does not look like text - it appears to be binary data.")
 
    # Pass 1: explicit delimiters, accepting only a genuine multi-column parse
    for enc in encodings:
        for delim in delimiters:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), sep=delim, encoding=enc, engine="python")
                if df.shape[1] > 1:
                    return df, enc, delim
            except Exception as e:  # noqa: BLE001 - deliberately try every combination
                last_err = e
                continue
 
    # Pass 2: plain default read - the correct answer for real single-column files
    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            if df.shape[1] >= 1:
                return df, enc, ","
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
 
    # Pass 3: sniffing, only once everything better has failed
    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python", encoding=enc)
            if df.shape[1] >= 1:
                return df, enc, "auto-detected"
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
 
    raise ValueError(f"Unable to parse this file as CSV. Last error: {last_err}")
 
 
# =====================================================================================
# SAMPLE DATASETS - let users explore the app without their own file
# =====================================================================================
@st.cache_data(show_spinner=False)
def generate_sample_dataset(kind):
    """Synthetic but realistically messy datasets covering both task types and every
    column archetype the app is designed to detect.
    """
    rng = np.random.default_rng(42)
 
    if kind == "Customer churn (classification)":
        n = 900
        tenure = rng.integers(1, 72, n)
        monthly = rng.normal(70, 25, n).clip(15, 160)
        df = pd.DataFrame({
            "customer_id": [f"CUS-{100000 + i}" for i in range(n)],
            "tenure_months": tenure.astype(float),
            "monthly_charges": monthly.round(2),
            "total_charges": (tenure * monthly).round(2),
            "contract_type": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]),
            "payment_method": rng.choice(["Card", "Bank transfer", "Electronic check", "Mailed check"], n),
            "support_tickets": rng.poisson(1.5, n).astype(float),
            "has_paperless_billing": rng.choice(["yes", "no"], n),
            "signup_date": pd.to_datetime("2019-01-01") + pd.to_timedelta(rng.integers(0, 1800, n), unit="D"),
            "region_code": [f"R{rng.integers(1, 180)}" for _ in range(n)],
        })
        risk = (
            (df["contract_type"] == "Month-to-month").astype(float) * 1.6
            + (df["support_tickets"] / 4.0)
            + (df["monthly_charges"] / 120.0)
            - (df["tenure_months"] / 50.0)
            + rng.normal(0, 0.55, n)
        )
        df["churned"] = np.where(risk > 1.5, "Yes", "No")
        df.loc[rng.choice(n, 70, replace=False), "total_charges"] = np.nan
        df.loc[rng.choice(n, 45, replace=False), "payment_method"] = np.nan
 
    elif kind == "Housing prices (regression)":
        n = 850
        area = rng.normal(1650, 520, n).clip(420, 4200)
        rooms = rng.integers(1, 7, n)
        age = rng.integers(0, 85, n)
        df = pd.DataFrame({
            "listing_ref": [f"LST{200000 + i}" for i in range(n)],
            "area_sqft": area.round(0),
            "bedrooms": rooms.astype(float),
            "bathrooms": rng.integers(1, 5, n).astype(float),
            "property_age_years": age.astype(float),
            "neighbourhood": rng.choice(
                ["Riverside", "Old Town", "Hillcrest", "Eastgate", "Parkview", "Docklands"], n
            ),
            "has_garage": rng.choice([True, False], n),
            "listed_on": pd.to_datetime("2021-06-01") + pd.to_timedelta(rng.integers(0, 1100, n), unit="D"),
        })
        df["sale_price"] = (
            area * 185 + rooms * 11000 - age * 750
            + df["has_garage"].astype(float) * 16000
            + rng.normal(0, 26000, n)
        ).round(0).clip(45000, None)
        df.loc[rng.choice(n, 60, replace=False), "property_age_years"] = np.nan
 
    else:  # "Employee attrition (mixed types)"
        n = 700
        years = rng.integers(0, 25, n)
        df = pd.DataFrame({
            "employee_uuid": [f"EMP-{rng.integers(10**6, 10**7)}-{i}" for i in range(n)],
            "age_years": rng.integers(21, 63, n).astype(float),
            "years_at_company": years.astype(float),
            "satisfaction_score": rng.uniform(1, 5, n).round(2),
            "department": rng.choice(["Engineering", "Sales", "HR", "Finance", "Support"], n),
            "job_level": rng.choice(["Junior", "Mid", "Senior", "Lead"], n, p=[0.35, 0.35, 0.2, 0.1]),
            "works_remotely": rng.choice(["true", "false"], n),
            "last_review_date": pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 700, n), unit="D"),
            "office_code": [f"OFF-{rng.integers(1, 95)}" for _ in range(n)],
            "constant_flag": ["ACTIVE"] * n,  # deliberately constant, to exercise detection
        })
        score = (
            -(df["satisfaction_score"] / 2.0)
            - (years / 14.0)
            + (df["job_level"] == "Junior").astype(float) * 0.9
            + rng.normal(0, 0.5, n)
        )
        df["left_company"] = np.where(score > -0.3, 1, 0)
        df.loc[rng.choice(n, 55, replace=False), "satisfaction_score"] = np.nan
 
    return df
 
 
# =====================================================================================
# COLUMN TYPE INFERENCE
# =====================================================================================
@st.cache_data(show_spinner=False)
def infer_column_types(df, cardinality_threshold=50):
    """Classify every column as numeric / categorical / high_cardinality_categorical /
    datetime / boolean / id_like. Purely statistical - never looks at the column name.
    """
    col_types = {}
    n_rows = len(df)
 
    for col in df.columns:
        s = df[col]
        n_unique = s.nunique(dropna=True)
 
        # --- boolean ---------------------------------------------------------------
        if pd.api.types.is_bool_dtype(s):
            col_types[col] = "boolean"
            continue
 
        non_null = s.dropna()
        if len(non_null) > 0 and n_unique <= 2:
            sample_vals = {str(v).strip().lower() for v in non_null.unique()[:20]}
            if sample_vals.issubset({"true", "false", "yes", "no", "y", "n"}):
                col_types[col] = "boolean"
                continue
 
        # --- datetime --------------------------------------------------------------
        if pd.api.types.is_datetime64_any_dtype(s):
            col_types[col] = "datetime"
            continue
 
        # NOTE: pandas 3.0 introduced a native string dtype, so plain text columns no
        # longer report dtype == object. Check both, or datetime detection silently
        # fails and date columns get misclassified as id_like.
        is_text_dtype = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
        if is_text_dtype and n_rows > 0:
            parsed = pd.to_datetime(s, errors="coerce")
            if parsed.notna().mean() > 0.9 and n_unique > 2:
                col_types[col] = "datetime"
                continue
 
        # --- numeric (including numeric IDs) ----------------------------------------
        if pd.api.types.is_numeric_dtype(s):
            try:
                is_integer_like = pd.api.types.is_integer_dtype(s) or (non_null.mod(1) == 0).all()
            except Exception:  # noqa: BLE001
                is_integer_like = False
            if n_rows > 20 and n_unique == n_rows and is_integer_like:
                col_types[col] = "id_like"
            else:
                col_types[col] = "numeric"
            continue
 
        # --- remaining text columns ---------------------------------------------------
        uniqueness_ratio = n_unique / n_rows if n_rows > 0 else 0
        if n_rows > 20 and uniqueness_ratio > 0.9 and n_unique > 20:
            col_types[col] = "id_like"
        elif n_unique > cardinality_threshold:
            col_types[col] = "high_cardinality_categorical"
        else:
            col_types[col] = "categorical"
 
    return col_types
 
 
@st.cache_data(show_spinner=False)
def profile_dataframe(df):
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "unique_count": df.nunique(),
        "unique_pct": (df.nunique() / max(len(df), 1) * 100).round(2),
    })
 
 
@st.cache_data(show_spinner=False)
def numeric_summary(df, numeric_cols):
    """Descriptive stats plus skew and IQR-based outlier counts for numeric columns."""
    rows = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = int(((s < lo) | (s > hi)).sum())
        rows.append({
            "column": col, "mean": s.mean(), "std": s.std(), "min": s.min(),
            "q1": q1, "median": s.median(), "q3": q3, "max": s.max(),
            "skew": s.skew(), "outliers_iqr": outliers,
            "outlier_pct": round(outliers / len(s) * 100, 2),
        })
    return pd.DataFrame(rows).set_index("column").round(3) if rows else pd.DataFrame()
 
 
def find_duplicate_columns(df, max_check_cols=300):
    """Groups of columns whose values are identical across every row."""
    seen = {}
    for col in df.columns[:max_check_cols]:
        try:
            key = hashlib.md5(
                pd.util.hash_pandas_object(df[col], index=False).values.tobytes()
            ).hexdigest()
        except Exception:  # noqa: BLE001
            continue
        seen.setdefault(key, []).append(col)
    return [g for g in seen.values() if len(g) > 1]
 
 
def detect_problem_type(y_non_null):
    """Numeric target with many distinct values -> regression; otherwise classification."""
    if pd.api.types.is_numeric_dtype(y_non_null) and not pd.api.types.is_bool_dtype(y_non_null):
        n_unique = y_non_null.nunique()
        if n_unique <= 20 and (n_unique / max(len(y_non_null), 1)) < 0.05:
            return "classification"
        return "regression"
    return "classification"
 
 
def rebucket_column_types(df, feature_cols, column_types, cardinality_threshold):
    """Re-apply the user's cardinality threshold (initial inference used a default)."""
    out = {}
    for c in feature_cols:
        t = column_types.get(c)
        if t in ("categorical", "high_cardinality_categorical"):
            n_unique = df[c].nunique(dropna=True) if c in df.columns else 0
            out[c] = "high_cardinality_categorical" if n_unique > cardinality_threshold else "categorical"
        else:
            out[c] = t
    return out
 
 
# =====================================================================================
# DATA QUALITY ENGINE
# =====================================================================================
def detect_leakage_candidates(df, target, column_types, problem_type, max_cols=60):
    """Flags features suspiciously predictive of the target on their own - a classic
    sign of leakage (a column recorded after, or derived from, the outcome).
 
    Numeric features: absolute Pearson correlation with the (encoded) target.
    Categorical features: how purely each category maps to a single target class.
    """
    suspects = []
    y = df[target]
    mask = y.notna()
    if mask.sum() < 10:
        return suspects
 
    if problem_type == "classification":
        y_codes = pd.Series(pd.factorize(y[mask].astype(str))[0], index=y[mask].index)
    else:
        y_codes = pd.to_numeric(y[mask], errors="coerce")
 
    for col in list(df.columns)[:max_cols]:
        if col == target:
            continue
        ctype = column_types.get(col)
        try:
            if ctype in ("numeric", "boolean"):
                s = pd.to_numeric(df.loc[mask, col], errors="coerce")
                if s.nunique() < 2:
                    continue
                corr = s.corr(y_codes)
                if pd.notna(corr) and abs(corr) > 0.95:
                    suspects.append((col, f"|correlation| with target = {abs(corr):.3f}"))
 
            elif ctype == "categorical" and problem_type == "classification":
                grouped = pd.DataFrame({"f": df.loc[mask, col].astype(str), "y": y[mask].astype(str)})
                purity = grouped.groupby("f")["y"].agg(lambda g: g.value_counts(normalize=True).iloc[0])
                counts = grouped["f"].value_counts()
                # Only consider groups large enough for purity to be meaningful
                valid = purity[counts.reindex(purity.index) >= 5]
                if len(valid) >= 2 and valid.mean() > 0.98:
                    suspects.append((col, f"categories map almost 1:1 to target classes ({valid.mean():.1%} pure)"))
        except Exception:  # noqa: BLE001 - never let a diagnostic break the app
            continue
 
    return suspects
 
 
@st.cache_data(show_spinner=False)
def assess_data_quality(df, column_types, target=None, problem_type=None):
    """Computes a 0-100 data-quality score plus a prioritised list of concrete issues.
 
    The score is a transparent deduction model rather than a black box: every point
    lost is traceable to one of the penalties below.
    """
    n_rows, n_cols = df.shape
    issues = []
 
    missing_frac = float(df.isna().mean().mean()) if n_cols else 0.0
    dup_rows = int(df.duplicated().sum())
    dup_frac = dup_rows / n_rows if n_rows else 0.0
 
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    high_missing_cols = [c for c in df.columns if df[c].isna().mean() > 0.5]
    id_like_cols = [c for c, t in column_types.items() if t == "id_like"]
    high_card_cols = [c for c, t in column_types.items() if t == "high_cardinality_categorical"]
    dup_col_groups = find_duplicate_columns(df)
 
    # ---- transparent deduction model --------------------------------------------------
    score = 100.0
    score -= missing_frac * 40.0
    score -= dup_frac * 25.0
    score -= (len(constant_cols) / n_cols * 15.0) if n_cols else 0
    score -= (len(high_missing_cols) / n_cols * 15.0) if n_cols else 0
    score -= min(len(dup_col_groups) * 3.0, 10.0)
    if n_rows < 50:
        score -= 12.0
    elif n_rows < 200:
        score -= 5.0
    score = float(max(0.0, min(100.0, score)))
 
    # ---- issue list ----------------------------------------------------------------------
    if missing_frac > 0.30:
        issues.append(("critical", "Heavy missing data",
                       f"{missing_frac:.1%} of all cells are empty. Imputation will carry a lot of weight."))
    elif missing_frac > 0.05:
        issues.append(("warning", "Missing data present",
                       f"{missing_frac:.1%} of all cells are empty across the dataset."))
 
    if high_missing_cols:
        issues.append(("warning", f"{len(high_missing_cols)} column(s) over 50% missing",
                       ", ".join(high_missing_cols[:6]) + ("..." if len(high_missing_cols) > 6 else "")))
 
    if dup_rows > 0:
        issues.append(("warning" if dup_frac < 0.10 else "critical", f"{dup_rows:,} duplicate row(s)",
                       f"{dup_frac:.1%} of rows are exact duplicates. These can inflate test scores."))
 
    if constant_cols:
        issues.append(("info", f"{len(constant_cols)} constant column(s)",
                       ", ".join(constant_cols[:6]) + " - zero predictive value, auto-excluded."))
 
    if dup_col_groups:
        issues.append(("warning", f"{len(dup_col_groups)} duplicated column pair(s)",
                       "; ".join(" == ".join(g) for g in dup_col_groups[:4])))
 
    if id_like_cols:
        issues.append(("info", f"{len(id_like_cols)} ID-like column(s) detected",
                       ", ".join(id_like_cols[:6]) + " - auto-excluded from modeling."))
 
    if high_card_cols:
        issues.append(("info", f"{len(high_card_cols)} high-cardinality column(s)",
                       ", ".join(high_card_cols[:6]) + " - routed to frequency/ordinal encoding."))
 
    if n_rows < 50:
        issues.append(("critical", "Very small dataset",
                       f"Only {n_rows} rows. Metrics will be unstable and easily overfit."))
    elif n_rows < 200:
        issues.append(("warning", "Small dataset",
                       f"{n_rows} rows. Consider cross-validation over a single split."))
 
    # ---- target-specific checks --------------------------------------------------------
    if target is not None and target in df.columns:
        t_missing = df[target].isna().mean()
        if t_missing > 0:
            issues.append(("warning" if t_missing < 0.2 else "critical",
                           f"Target is {t_missing:.1%} missing",
                           "Those rows are dropped before training."))
 
        if problem_type == "classification":
            vc = df[target].dropna().astype(str).value_counts()
            if len(vc) >= 2:
                ratio = vc.max() / vc.min()
                if ratio > 10:
                    issues.append(("critical", "Severe class imbalance",
                                   f"Most-to-least frequent ratio is {ratio:.0f}x. Accuracy will be misleading; use macro F1 and consider balanced class weights."))
                elif ratio > 3:
                    issues.append(("warning", "Class imbalance",
                                   f"Most-to-least frequent ratio is {ratio:.1f}x. Prefer macro F1 over accuracy."))
            if len(vc) > 0 and (vc < 2).any():
                rare = vc[vc < 2].index.tolist()
                issues.append(("warning", "Rare class(es) with <2 samples",
                               f"{', '.join(map(str, rare[:5]))} - stratified splitting will be disabled."))
        elif problem_type == "regression":
            s = pd.to_numeric(df[target], errors="coerce").dropna()
            if not s.empty and abs(s.skew()) > 2:
                issues.append(("warning", "Heavily skewed target",
                               f"Skew = {s.skew():.2f}. A log transform may improve model fit."))
 
        leaks = detect_leakage_candidates(df, target, column_types, problem_type)
        for col, reason in leaks:
            issues.append(("critical", f"Possible target leakage: {col}", reason))
 
    if not issues:
        issues.append(("info", "No significant issues detected", "This dataset looks clean and ready to model."))
 
    severity_rank = {"critical": 0, "warning": 1, "info": 2}
    issues.sort(key=lambda x: severity_rank[x[0]])
 
    return {
        "score": score,
        "grade": "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D",
        "issues": issues,
        "missing_frac": missing_frac,
        "duplicate_rows": dup_rows,
        "constant_cols": constant_cols,
        "high_missing_cols": high_missing_cols,
        "id_like_cols": id_like_cols,
        "high_card_cols": high_card_cols,
        "duplicate_col_groups": dup_col_groups,
    }
 
 
def estimate_onehot_width(df, categorical_cols):
    """Predicts how wide one-hot encoding would make the feature matrix."""
    return int(sum(df[c].nunique(dropna=True) for c in categorical_cols if c in df.columns))
 
 
# =====================================================================================
# FEATURE ENGINEERING (deterministic - safe to apply before the train/test split)
# =====================================================================================
def engineer_datetime_features(df, datetime_cols):
    """Expand each datetime column into numeric calendar parts, then drop the raw column."""
    df = df.copy()
    new_cols = []
    for col in datetime_cols:
        dt = pd.to_datetime(df[col], errors="coerce")
        for part, values in [
            ("year", dt.dt.year), ("month", dt.dt.month),
            ("day", dt.dt.day), ("dayofweek", dt.dt.dayofweek),
        ]:
            name = f"{col}_{part}"
            df[name] = values.astype("float")
            new_cols.append(name)
        df.drop(columns=[col], inplace=True)
    return df, new_cols
 
 
def engineer_boolean_features(df, boolean_cols):
    """Map boolean-like columns to 0/1 floats so they flow through the numeric branch."""
    df = df.copy()
    bool_map = {"true": 1.0, "yes": 1.0, "y": 1.0, "1": 1.0, "1.0": 1.0,
                "false": 0.0, "no": 0.0, "n": 0.0, "0": 0.0, "0.0": 0.0}
    for col in boolean_cols:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(float)
        else:
            df[col] = df[col].astype(str).str.strip().str.lower().map(bool_map)
    return df
 
 
def resolve_feature_roles(df, feature_cols, column_types, cardinality_threshold):
    """Single source of truth for which column goes down which preprocessing branch.
    Used by both training and new-data scoring so the two can never drift apart.
    """
    types = rebucket_column_types(df, feature_cols, column_types, cardinality_threshold)
    datetime_cols = [c for c in feature_cols if types.get(c) == "datetime"]
    boolean_cols = [c for c in feature_cols if types.get(c) == "boolean"]
    numeric_base = [c for c in feature_cols if types.get(c) == "numeric"]
    categorical_cols = [c for c in feature_cols if types.get(c) == "categorical"]
    high_card_cols = [c for c in feature_cols if types.get(c) == "high_cardinality_categorical"]
 
    known = set(numeric_base) | set(categorical_cols) | set(high_card_cols) | set(datetime_cols) | set(boolean_cols)
    leftover = [c for c in feature_cols if c not in known]
    categorical_cols = categorical_cols + leftover  # defensive catch-all
 
    return {
        "datetime_cols": datetime_cols,
        "boolean_cols": boolean_cols,
        "numeric_base": numeric_base,
        "categorical_cols": categorical_cols,
        "high_card_cols": high_card_cols,
    }
 
 
def apply_feature_engineering(df, roles):
    """Replays datetime expansion + boolean mapping, returning the frame and the final
    numeric column list. Deterministic and row-independent, so it is leakage-safe.
    """
    out, new_dt_cols = engineer_datetime_features(df, roles["datetime_cols"])
    out = engineer_boolean_features(out, roles["boolean_cols"])
    numeric_cols = roles["numeric_base"] + roles["boolean_cols"] + new_dt_cols
    return out, numeric_cols
 
 
# =====================================================================================
# PREPROCESSING PIPELINE
# =====================================================================================
def build_preprocessor(numeric_cols, categorical_cols, high_card_cols, config):
    transformers = []
 
    if numeric_cols:
        steps = []
        strategy = config["numeric_impute"]
        if strategy in ("mean", "median"):
            steps.append(("imputer", SimpleImputer(strategy=strategy)))
        else:
            steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=0)))
        if config["scale"]:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps), numeric_cols))
 
    if categorical_cols:
        imputer = (SimpleImputer(strategy="most_frequent") if config["categorical_impute"] == "mode"
                   else SimpleImputer(strategy="constant", fill_value="Missing"))
        transformers.append(("cat", Pipeline([
            ("imputer", imputer),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_cols))
 
    if high_card_cols:
        encoder = (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                   if config["high_card_encoding"] == "ordinal" else FrequencyEncoder())
        transformers.append(("hc", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", encoder),
        ]), high_card_cols))
 
    return ColumnTransformer(transformers=transformers, remainder="drop")
 
 
def get_readable_feature_names(preprocessor):
    """Strip sklearn's 'num__' / 'cat__' prefixes so charts show human-readable names."""
    return [n.split("__", 1)[1] if "__" in n else n for n in preprocessor.get_feature_names_out()]
 
 
def compute_cache_key(df, target, problem_type, preprocess_config, model_config):
    h = hashlib.sha256()
    try:
        h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    except Exception:  # noqa: BLE001
        h.update(str(df.shape).encode())
    h.update(target.encode())
    h.update(problem_type.encode())
    h.update(json.dumps(preprocess_config, sort_keys=True, default=str).encode())
    h.update(json.dumps(model_config, sort_keys=True, default=str).encode())
    return h.hexdigest()
 
 
# =====================================================================================
# MODEL TRAINING
# =====================================================================================
@st.cache_resource(show_spinner=False)
def train_models(_df, target, problem_type, _feature_cols, _column_types,
                 _preprocess_config, _model_config, cache_key):
    """Trains the model set for the detected task type. Cached on an explicit key so
    retraining only happens when data, target, preprocessing, or hyperparameters change.
    """
    df, feature_cols = _df, _feature_cols
    column_types = _column_types
    pre_cfg, mdl_cfg = _preprocess_config, _model_config
 
    work = df[feature_cols + [target]].copy()
    work = work.dropna(subset=[target])
    if pre_cfg.get("drop_duplicate_rows"):
        work = work.drop_duplicates()
 
    roles = resolve_feature_roles(work, feature_cols, column_types, pre_cfg["cardinality_threshold"])
    work, numeric_cols = apply_feature_engineering(work, roles)
    categorical_cols, high_card_cols = roles["categorical_cols"], roles["high_card_cols"]
 
    X = work[numeric_cols + categorical_cols + high_card_cols]
    y_raw = work[target]
 
    # ---- target encoding / transform -------------------------------------------------
    label_encoder, target_log = None, False
    if problem_type == "classification":
        classes = sorted(y_raw.astype(str).unique())
        label_map = {c: i for i, c in enumerate(classes)}
        y = y_raw.astype(str).map(label_map)
        label_encoder = {"classes": classes, "map": label_map}
    else:
        y = pd.to_numeric(y_raw, errors="coerce").astype(float)
        if mdl_cfg.get("log_transform_target") and (y > 0).all():
            y = np.log1p(y)
            target_log = True
 
    # ---- train / test split ------------------------------------------------------------
    stratify_arg, split_warning = None, None
    if problem_type == "classification":
        vc = y.value_counts()
        if vc.min() >= 2:
            stratify_arg = y
        else:
            split_warning = "At least one class has fewer than 2 samples - falling back to a non-stratified split."
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=mdl_cfg["test_size"],
        random_state=mdl_cfg["random_state"], stratify=stratify_arg,
    )
 
    # ---- preprocessing: fit ONLY on train -----------------------------------------------
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, high_card_cols, pre_cfg)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = get_readable_feature_names(preprocessor)
 
    # ---- model set ------------------------------------------------------------------------
    models = {}
    class_weight = "balanced" if mdl_cfg.get("balance_classes") else None
    if problem_type == "classification":
        models["Random Forest"] = RandomForestClassifier(
            n_estimators=mdl_cfg["rf_n_estimators"], max_depth=mdl_cfg["rf_max_depth"],
            min_samples_leaf=mdl_cfg.get("rf_min_samples_leaf", 1),
            class_weight=class_weight, random_state=mdl_cfg["random_state"], n_jobs=-1,
        )
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBClassifier(
                n_estimators=mdl_cfg["xgb_n_estimators"], max_depth=mdl_cfg["xgb_max_depth"],
                learning_rate=mdl_cfg["xgb_learning_rate"],
                subsample=mdl_cfg.get("xgb_subsample", 1.0),
                random_state=mdl_cfg["random_state"], eval_metric="logloss",
            )
        if mdl_cfg["include_baseline"]:
            models["Logistic Regression"] = LogisticRegression(max_iter=1000, class_weight=class_weight)
    else:
        models["Random Forest"] = RandomForestRegressor(
            n_estimators=mdl_cfg["rf_n_estimators"], max_depth=mdl_cfg["rf_max_depth"],
            min_samples_leaf=mdl_cfg.get("rf_min_samples_leaf", 1),
            random_state=mdl_cfg["random_state"], n_jobs=-1,
        )
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBRegressor(
                n_estimators=mdl_cfg["xgb_n_estimators"], max_depth=mdl_cfg["xgb_max_depth"],
                learning_rate=mdl_cfg["xgb_learning_rate"],
                subsample=mdl_cfg.get("xgb_subsample", 1.0),
                random_state=mdl_cfg["random_state"],
            )
        if mdl_cfg["include_baseline"]:
            models["Linear Regression"] = LinearRegression()
 
    # ---- fit + evaluate ----------------------------------------------------------------------
    results_by_model = {}
    n_classes = len(label_encoder["classes"]) if label_encoder else 0
 
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train_t, y_train)
        train_seconds = time.time() - t0
        preds = model.predict(X_test_t)
        entry = {"model": model, "preds": preds, "train_seconds": train_seconds}
 
        if problem_type == "classification":
            proba = model.predict_proba(X_test_t) if hasattr(model, "predict_proba") else None
            entry.update({
                "proba": proba,
                "accuracy": accuracy_score(y_test, preds),
                "f1_macro": f1_score(y_test, preds, average="macro"),
                "precision_macro": precision_score(y_test, preds, average="macro", zero_division=0),
                "recall_macro": recall_score(y_test, preds, average="macro", zero_division=0),
                "report": classification_report(
                    y_test, preds, labels=list(range(n_classes)),
                    target_names=[str(c) for c in label_encoder["classes"]],
                    output_dict=True, zero_division=0),
                "confusion_matrix": confusion_matrix(y_test, preds, labels=list(range(n_classes))),
            })
            entry["score"] = entry["f1_macro"]
        else:
            entry.update({
                "r2": r2_score(y_test, preds),
                "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
                "mae": mean_absolute_error(y_test, preds),
            })
            entry["score"] = entry["r2"]
 
        # ---- cross-validation (optional; more trustworthy on small data) -----------------
        if mdl_cfg.get("use_cv"):
            try:
                folds = mdl_cfg.get("cv_folds", 5)
                if problem_type == "classification":
                    min_class = y_train.value_counts().min()
                    folds = max(2, min(folds, int(min_class)))
                    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=mdl_cfg["random_state"])
                    scoring = "f1_macro"
                else:
                    cv = KFold(n_splits=folds, shuffle=True, random_state=mdl_cfg["random_state"])
                    scoring = "r2"
                cv_scores = cross_val_score(model, X_train_t, y_train, cv=cv, scoring=scoring, n_jobs=-1)
                entry["cv_mean"] = float(np.mean(cv_scores))
                entry["cv_std"] = float(np.std(cv_scores))
                entry["cv_folds_used"] = folds
            except Exception as e:  # noqa: BLE001 - CV is a nice-to-have, never fatal
                entry["cv_error"] = str(e)
 
        # ---- feature importance ---------------------------------------------------------
        if hasattr(model, "feature_importances_"):
            entry["feature_importances"] = dict(zip(feature_names, model.feature_importances_))
            entry["importance_kind"] = "impurity"
        elif hasattr(model, "coef_"):
            coefs = np.asarray(model.coef_)
            coefs = coefs[0] if coefs.ndim > 1 else coefs
            entry["feature_importances"] = dict(zip(feature_names, np.abs(coefs)))
            entry["importance_kind"] = "|coefficient|"
 
        if mdl_cfg.get("use_permutation_importance"):
            try:
                pi = permutation_importance(
                    model, X_test_t, y_test, n_repeats=5,
                    random_state=mdl_cfg["random_state"], n_jobs=-1)
                entry["permutation_importances"] = dict(zip(feature_names, pi.importances_mean))
            except Exception:  # noqa: BLE001
                pass
 
        results_by_model[name] = entry
 
    best_name = max(results_by_model, key=lambda n: results_by_model[n]["score"])
 
    return {
        "model_results": results_by_model,
        "best_name": best_name,
        "y_test": y_test,
        "X_test_raw": X_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "label_encoder": label_encoder,
        "split_warning": split_warning,
        "problem_type": problem_type,
        "target": target,
        "feature_cols": feature_cols,
        "roles": roles,
        "target_log": target_log,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "preprocess_config": pre_cfg,
        "model_config": mdl_cfg,
    }
 
 
# =====================================================================================
# SHARED UI HELPERS
# =====================================================================================
def workflow_state():
    """Which stages of the workflow are complete - drives the sidebar + dashboard."""
    return {
        "uploaded": "df" in st.session_state,
        "target_set": "target" in st.session_state,
        "configured": "preprocess_config" in st.session_state,
        "trained": st.session_state.get("results") is not None,
    }
 
 
def render_workflow_status():
    s = workflow_state()
    steps = [("Data loaded", s["uploaded"]), ("Target selected", s["target_set"]),
             ("Preprocessing set", s["configured"]), ("Model trained", s["trained"])]
    html = ""
    for label, done in steps:
        cls = "step-done" if done else "step-pending"
        mark = "\u2713" if done else "\u25CB"
        html += f'<div class="{cls}">{mark} {label}</div>'
    st.sidebar.markdown(html, unsafe_allow_html=True)
 
 
def require_data():
    if "df" not in st.session_state:
        st.warning("No dataset loaded yet. Head to **Data & Profiling** to upload a CSV or load a sample dataset.")
        return False
    return True
 
 
def require_target():
    if not require_data():
        return False
    if "target" not in st.session_state:
        st.warning("No target column selected yet. Choose one in **Data & Profiling**.")
        return False
    return True
 
 
def reset_downstream():
    """Invalidate anything that depended on the previous data/target selection."""
    for key in ("results", "preprocess_config", "feature_cols", "trained", "quality"):
        st.session_state.pop(key, None)
 
 
# =====================================================================================
# PAGE - DASHBOARD
# =====================================================================================
def page_dashboard():
    render_hero(f"\u25F0 {APP_NAME}", APP_TAGLINE)
 
    if "df" not in st.session_state:
        st.markdown(
            "### Welcome\n"
            f"**{APP_NAME}** profiles, explores, and models **any** tabular CSV. "
            "Nothing is hardcoded to a particular dataset: column types, encoding strategy, "
            "model choice, and every metric are inferred at runtime from your file."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### \U0001F4C1 Profile\nType inference, quality scoring, and issue detection before you model.")
        with c2:
            st.markdown("#### \U0001F4C8 Explore\nFully dynamic charts built from your detected columns.")
        with c3:
            st.markdown("#### \U0001F916 Model\nLeakage-safe pipelines, adaptive metrics, exportable artifacts.")
        st.divider()
        st.info("Open **Data & Profiling** in the sidebar to upload a CSV, or load one of the built-in sample datasets to explore the app immediately.")
        return
 
    df = st.session_state.df
    column_types = st.session_state.column_types
    target = st.session_state.get("target")
    problem_type = st.session_state.get("problem_type")
 
    quality = assess_data_quality(df, column_types, target, problem_type)
    st.session_state.quality = quality
 
    # ---- KPI row -----------------------------------------------------------------------
    grade_color = {"A": "#1a9850", "B": "#66bd63", "C": "#fc8d59", "D": "#d73027"}[quality["grade"]]
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(kpi_card("Rows", f"{df.shape[0]:,}", f"{df.shape[1]} columns"), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card("Missing cells", f"{quality['missing_frac']:.1%}",
                             f"{int(df.isna().sum().sum()):,} empty values"), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card("Duplicate rows", f"{quality['duplicate_rows']:,}",
                             f"{quality['duplicate_rows'] / max(len(df), 1):.1%} of dataset"), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_card("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB",
                             "in-memory footprint"), unsafe_allow_html=True)
    with k5:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Quality score</div>'
            f'<div class="kpi-value" style="color:{grade_color}">{quality["score"]:.0f}'
            f'<span style="font-size:1rem;"> / 100 &nbsp;({quality["grade"]})</span></div>'
            f'<div class="kpi-sub">transparent deduction model</div></div>',
            unsafe_allow_html=True)
 
    st.write("")
 
    # ---- composition + issues ------------------------------------------------------------
    left, right = st.columns([1, 1.35])
 
    with left:
        st.subheader("Column composition")
        type_counts = pd.Series(column_types).value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index, hole=0.55)
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300,
                          legend=dict(orientation="h", y=-0.1))
        fig.update_traces(textposition="inside", textinfo="value+label")
        st.plotly_chart(fig, use_container_width=True, key="dash_types")
 
    with right:
        st.subheader("Detected issues")
        counts = {"critical": 0, "warning": 0, "info": 0}
        for sev, _, _ in quality["issues"]:
            counts[sev] += 1
        b1, b2, b3 = st.columns(3)
        b1.metric("Critical", counts["critical"])
        b2.metric("Warnings", counts["warning"])
        b3.metric("Notes", counts["info"])
 
        with st.container(height=260):
            for sev, title, detail in quality["issues"]:
                st.markdown(
                    f'<div class="issue-row" style="--sev-color:{SEVERITY_COLORS[sev]}">'
                    f'<div class="issue-title">{title}</div>'
                    f'<div class="issue-detail">{detail}</div></div>',
                    unsafe_allow_html=True)
 
    st.divider()
 
    # ---- target snapshot -------------------------------------------------------------------
    if target:
        st.subheader(f"Target snapshot: `{target}`")
        tcol1, tcol2 = st.columns([1, 1.4])
        series = df[target].dropna()
        with tcol1:
            if problem_type == "classification":
                vc = series.astype(str).value_counts()
                ratio = vc.max() / vc.min() if vc.min() > 0 else float("inf")
                st.metric("Task", "Classification")
                st.metric("Classes", len(vc))
                st.metric("Imbalance ratio", f"{ratio:.1f}x")
            else:
                st.metric("Task", "Regression")
                st.metric("Mean", f"{series.mean():,.2f}")
                st.metric("Skew", f"{series.skew():.2f}")
        with tcol2:
            if problem_type == "classification":
                vc = series.astype(str).value_counts()
                fig = px.bar(x=vc.index, y=vc.values, labels={"x": target, "y": "count"})
            else:
                fig = px.histogram(series, x=target, nbins=40)
            fig.update_layout(margin=dict(t=20, b=10), height=260, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="dash_target")
    else:
        st.info("Select a target column in **Data & Profiling** to unlock target diagnostics and modeling.")
 
    # ---- live model summary -------------------------------------------------------------------
    results = st.session_state.get("results")
    if results:
        st.divider()
        st.subheader("Latest model run")
        best = results["model_results"][results["best_name"]]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Best model", results["best_name"])
        if results["problem_type"] == "classification":
            m2.metric("Accuracy", f"{best['accuracy']:.3f}")
            m3.metric("Macro F1", f"{best['f1_macro']:.3f}")
        else:
            m2.metric("R\u00b2", f"{best['r2']:.3f}")
            m3.metric("RMSE", f"{best['rmse']:,.2f}")
        m4.metric("Train / test rows", f"{results['n_train']:,} / {results['n_test']:,}")
 
        s1, s2 = st.columns(2)
        with s1:
            comp = build_comparison_frame(results["model_results"], results["problem_type"])
            metric_col = "Macro F1" if results["problem_type"] == "classification" else "R2"
            fig = px.bar(comp.reset_index(), x="index", y=metric_col,
                         labels={"index": "model"}, title=f"Model comparison ({metric_col})")
            fig.update_layout(height=280, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True, key="dash_modelcomp")
        with s2:
            if "feature_importances" in best:
                fi = pd.Series(best["feature_importances"]).sort_values(ascending=True).tail(10)
                fig = px.bar(x=fi.values, y=fi.index, orientation="h",
                             labels={"x": "importance", "y": ""}, title="Top drivers")
                fig.update_layout(height=280, margin=dict(t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, key="dash_fi")
    else:
        st.divider()
        st.info("No model trained yet. Configure preprocessing, then train in **Modeling & Results**.")
 
 
# =====================================================================================
# PAGE - DATA & PROFILING
# =====================================================================================
def page_data_upload():
    render_hero("Data & Profiling", "Load a CSV, inspect what was detected, and choose your target.")
 
    src1, src2 = st.columns([1.3, 1])
    with src1:
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    with src2:
        st.markdown("**...or load a sample dataset**")
        sample_choice = st.selectbox(
            "Sample dataset", ["None",
                               "Customer churn (classification)",
                               "Housing prices (regression)",
                               "Employee attrition (mixed types)"],
            label_visibility="collapsed")
        load_sample = st.button("Load sample", use_container_width=True)
 
    df = None
    source_note = ""
 
    if load_sample and sample_choice != "None":
        df = generate_sample_dataset(sample_choice)
        source_note = f"Sample dataset: {sample_choice}"
        st.session_state.data_source = source_note
        reset_downstream()
    elif uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        if len(file_bytes) == 0:
            st.error("The uploaded file is empty.")
            return
        try:
            df, enc, delim = load_data(file_bytes, uploaded_file.name)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read this file as a CSV: {e}")
            return
        source_note = f"{uploaded_file.name} (encoding `{enc}`, delimiter `{delim}`)"
        if st.session_state.get("data_source") != source_note:
            reset_downstream()
        st.session_state.data_source = source_note
 
    if df is None:
        if "df" in st.session_state:
            df = st.session_state.df
            source_note = st.session_state.get("data_source", "current dataset")
        else:
            st.info("Upload a CSV or load a sample dataset to begin.")
            return
 
    if df.shape[1] == 0:
        st.error("The file appears to have no columns.")
        return
    if df.shape[0] < 2:
        st.error("Dataset has fewer than 2 rows - cannot proceed.")
        return
 
    st.success(f"Loaded **{df.shape[0]:,} rows x {df.shape[1]} columns** - {source_note}")
 
    st.session_state.df = df
    column_types = infer_column_types(df)
    st.session_state.column_types = column_types
    st.session_state.constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
 
    quality = assess_data_quality(df, column_types)
    crit = [i for i in quality["issues"] if i[0] == "critical"]
    if crit:
        st.error(f"**{len(crit)} critical issue(s) detected.** See the Dashboard for the full breakdown. Top item: {crit[0][1]}")
 
    tab_profile, tab_numeric, tab_preview = st.tabs(["Column profile", "Numeric summary", "Data preview"])
 
    with tab_profile:
        profile = profile_dataframe(df).copy()
        profile["detected_type"] = profile.index.map(column_types)
        st.dataframe(profile, use_container_width=True)
        st.caption("Types are inferred statistically from dtype, cardinality, and parse rate - never from column names.")
 
    with tab_numeric:
        numeric_cols = [c for c, t in column_types.items() if t == "numeric"]
        if numeric_cols:
            st.dataframe(numeric_summary(df, numeric_cols), use_container_width=True)
            st.caption("Outliers counted using the 1.5 x IQR rule.")
        else:
            st.info("No numeric columns detected.")
 
    with tab_preview:
        st.dataframe(df.head(50), use_container_width=True)
 
    st.divider()
 
    # ---- target selection -------------------------------------------------------------------
    st.subheader("Target selection")
    candidates = [c for c in df.columns if c not in st.session_state.constant_cols]
    if not candidates:
        st.error("No usable target column - every column is constant.")
        return
 
    prev_target = st.session_state.get("target")
    default_idx = candidates.index(prev_target) if prev_target in candidates else len(candidates) - 1
    target = st.selectbox("Which column do you want to predict?", candidates, index=default_idx)
 
    t_missing = df[target].isna().mean() * 100
    if t_missing > 50:
        st.error(f"Target `{target}` is {t_missing:.1f}% missing. Choose a more complete column.")
        return
    if t_missing > 0:
        st.warning(f"Target has {t_missing:.1f}% missing values - those rows are dropped before training.")
 
    non_null = df[target].dropna()
    auto_type = detect_problem_type(non_null)
    problem_type = st.radio(
        "Problem type", ["classification", "regression"],
        index=0 if auto_type == "classification" else 1, horizontal=True,
        help=f"Auto-detected as **{auto_type}** from the target's dtype and {non_null.nunique()} unique value(s). Override if that's wrong.")
 
    if problem_type == "classification" and non_null.nunique() < 2:
        st.error(f"Target `{target}` has only {non_null.nunique()} class - classification needs at least 2.")
        return
    if problem_type == "regression" and not pd.api.types.is_numeric_dtype(non_null):
        st.error(f"Target `{target}` is not numeric, so it cannot be used for regression.")
        return
 
    if prev_target != target or st.session_state.get("problem_type") != problem_type:
        reset_downstream()
    st.session_state.target = target
    st.session_state.problem_type = problem_type
 
    leaks = detect_leakage_candidates(df, target, column_types, problem_type)
    if leaks:
        st.error("**Possible target leakage detected.** These features predict the target almost perfectly on "
                 "their own, which usually means they encode the outcome:\n\n"
                 + "\n".join(f"- `{c}` - {r}" for c, r in leaks)
                 + "\n\nConsider excluding them in **Preprocessing**.")
 
    st.success(f"Target set to **{target}**, treated as **{problem_type}**. Continue to **Explore** or **Preprocessing**.")
 
 
# =====================================================================================
# PAGE - EXPLORE (EDA)
# =====================================================================================
def page_eda():
    render_hero("Explore", "Every chart below is built from the columns detected in your file.")
    if not require_target():
        return
 
    df = st.session_state.df
    target = st.session_state.target
    problem_type = st.session_state.problem_type
    column_types = st.session_state.column_types
 
    numeric_cols = [c for c, t in column_types.items() if t == "numeric"]
    categorical_cols = [c for c, t in column_types.items() if t in ("categorical", "high_cardinality_categorical")]
 
    t_dist, t_missing, t_corr, t_uni, t_rel = st.tabs(
        ["Target", "Missing data", "Correlation", "Distributions", "Relationships"])
 
    # ---- target ---------------------------------------------------------------------------
    with t_dist:
        series = df[target].dropna()
        if problem_type == "classification":
            vc = series.astype(str).value_counts()
            fig = px.bar(x=vc.index, y=vc.values, labels={"x": target, "y": "count"},
                         title=f"Class distribution of {target}")
            st.plotly_chart(fig, use_container_width=True, key="eda_target_cls")
            ratio = vc.max() / vc.min() if vc.min() > 0 else float("inf")
            note = ("fairly balanced" if ratio < 3 else
                    "noticeably imbalanced - prefer macro F1 over accuracy" if ratio < 10 else
                    "severely imbalanced - consider balanced class weights")
            st.info(f"**{len(vc)} classes.** Most-to-least frequent ratio: **{ratio:.1f}x** ({note}). "
                    f"Majority class `{vc.index[0]}` covers {vc.iloc[0] / vc.sum():.1%} of rows, "
                    f"which is the accuracy a trivial always-predict-majority model would reach.")
        else:
            fig = px.histogram(series, x=target, nbins=40, marginal="box",
                               title=f"Distribution of {target}")
            st.plotly_chart(fig, use_container_width=True, key="eda_target_reg")
            skew = series.skew()
            note = ("roughly symmetric" if abs(skew) < 0.5 else
                    "moderately skewed" if abs(skew) < 1 else "heavily skewed - a log transform may help")
            st.info(f"**Skew {skew:.2f}** ({note}). Mean {series.mean():,.2f}, median {series.median():,.2f}, "
                    f"std {series.std():,.2f}, range {series.min():,.2f} to {series.max():,.2f}.")
 
        # Target-vs-feature drilldown
        st.markdown("**Target against a chosen feature**")
        drill_options = [c for c in df.columns if c != target]
        if drill_options:
            drill = st.selectbox("Feature", drill_options, key="eda_drill")
            try:
                if column_types.get(drill) in ("numeric", "datetime"):
                    if problem_type == "classification":
                        fig = px.box(df, x=df[target].astype(str), y=drill,
                                     labels={"x": target}, title=f"{drill} by {target}")
                    else:
                        fig = px.scatter(df, x=drill, y=target, trendline="ols",
                                         title=f"{target} vs {drill}")
                else:
                    if problem_type == "classification":
                        ct = pd.crosstab(df[drill].astype(str), df[target].astype(str), normalize="index")
                        fig = px.bar(ct, barmode="stack", title=f"{target} composition within {drill}")
                    else:
                        fig = px.box(df, x=df[drill].astype(str), y=target,
                                     labels={"x": drill}, title=f"{target} by {drill}")
                st.plotly_chart(fig, use_container_width=True, key="eda_drill_chart")
            except Exception as e:  # noqa: BLE001
                st.warning(f"Could not render that combination: {e}")
 
    # ---- missing data ------------------------------------------------------------------------
    with t_missing:
        missing = (df.isna().mean() * 100).sort_values(ascending=False)
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values anywhere in this dataset.")
        else:
            fig = px.bar(x=missing.index, y=missing.values,
                         labels={"x": "column", "y": "% missing"}, title="Missing value % by column")
            st.plotly_chart(fig, use_container_width=True, key="eda_missing")
            st.info(f"**{len(missing)} column(s)** contain missing values. Worst: `{missing.index[0]}` "
                    f"at **{missing.iloc[0]:.1f}%**. Rows fully complete: "
                    f"**{df.notna().all(axis=1).mean():.1%}**.")
            if len(missing) > 1:
                st.markdown("**Missingness co-occurrence** - do columns tend to be missing together?")
                miss_matrix = df[missing.index].isna().astype(int)
                fig = px.imshow(miss_matrix.corr(), text_auto=".2f",
                                color_continuous_scale="Purples", zmin=0, zmax=1)
                st.plotly_chart(fig, use_container_width=True, key="eda_missing_corr")
 
    # ---- correlation --------------------------------------------------------------------------
    with t_corr:
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns for a correlation heatmap.")
        else:
            selected = st.multiselect("Numeric columns to include", numeric_cols,
                                      default=numeric_cols[:min(len(numeric_cols), 12)], key="eda_corr_cols")
            method = st.radio("Method", ["pearson", "spearman"], horizontal=True, key="eda_corr_method")
            if len(selected) >= 2:
                corr = df[selected].corr(method=method, numeric_only=True)
                fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                zmin=-1, zmax=1, title=f"{method.title()} correlation matrix")
                st.plotly_chart(fig, use_container_width=True, key="eda_corr")
 
                masked = corr.where(~np.eye(len(corr), dtype=bool))
                if masked.notna().any().any():
                    pair = masked.abs().stack().idxmax()
                    st.info(f"Strongest pair: `{pair[0]}` and `{pair[1]}` at **{corr.loc[pair]:.2f}**.")
                    strong = (masked.abs() > 0.8).sum().sum() / 2
                    if strong > 0:
                        st.warning(f"{int(strong)} pair(s) exceed |0.8| correlation. Multicollinearity inflates "
                                   "the apparent importance of correlated features and destabilises linear models.")
            else:
                st.info("Select at least 2 columns.")
 
    # ---- distributions --------------------------------------------------------------------------
    with t_uni:
        c1, c2 = st.columns(2)
        with c1:
            if numeric_cols:
                choice = st.selectbox("Numeric column", numeric_cols, key="eda_num")
                fig = px.histogram(df, x=choice, nbins=40, marginal="box", title=f"Distribution of {choice}")
                st.plotly_chart(fig, use_container_width=True, key="eda_num_chart")
                s = pd.to_numeric(df[choice], errors="coerce").dropna()
                if not s.empty:
                    q1, q3 = s.quantile(0.25), s.quantile(0.75)
                    iqr = q3 - q1
                    out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
                    st.caption(f"Skew {s.skew():.2f} | {out} IQR outlier(s) ({out / len(s):.1%}) | "
                               f"{s.nunique()} unique values")
            else:
                st.info("No numeric columns detected.")
        with c2:
            if categorical_cols:
                choice = st.selectbox("Categorical column", categorical_cols, key="eda_cat")
                vc = df[choice].astype(str).value_counts().head(25)
                fig = px.bar(x=vc.index, y=vc.values, labels={"x": choice, "y": "count"},
                             title=f"Top categories in {choice}")
                st.plotly_chart(fig, use_container_width=True, key="eda_cat_chart")
                total_unique = df[choice].nunique()
                st.caption(f"{total_unique} unique categories | top value `{vc.index[0]}` covers "
                           f"{vc.iloc[0] / len(df):.1%} of rows")
            else:
                st.info("No categorical columns detected.")
 
    # ---- relationships -------------------------------------------------------------------------
    with t_rel:
        all_cols = list(df.columns)
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            x_col = st.selectbox("X axis", all_cols, key="rel_x")
        with e2:
            y_col = st.selectbox("Y axis", all_cols, index=min(1, len(all_cols) - 1), key="rel_y")
        with e3:
            chart_type = st.selectbox("Chart type", ["scatter", "box", "line", "violin", "density heatmap"], key="rel_type")
        with e4:
            color_options = ["(none)"] + all_cols
            default_color = color_options.index(target) if target in color_options else 0
            color_col = st.selectbox("Colour by", color_options, index=default_color, key="rel_color")
 
        color = None if color_col == "(none)" else color_col
        try:
            if chart_type == "scatter":
                fig = px.scatter(df, x=x_col, y=y_col, color=color, title=f"{y_col} vs {x_col}")
            elif chart_type == "box":
                fig = px.box(df, x=x_col, y=y_col, color=color, title=f"{y_col} by {x_col}")
            elif chart_type == "violin":
                fig = px.violin(df, x=x_col, y=y_col, color=color, box=True, title=f"{y_col} by {x_col}")
            elif chart_type == "density heatmap":
                fig = px.density_heatmap(df, x=x_col, y=y_col, title=f"Density of {y_col} vs {x_col}")
            else:
                fig = px.line(df.sort_values(by=x_col), x=x_col, y=y_col, color=color,
                              title=f"{y_col} over {x_col}")
            st.plotly_chart(fig, use_container_width=True, key="rel_chart")
        except Exception as e:  # noqa: BLE001
            st.warning(f"Could not render this combination: {e}")
 
 
# =====================================================================================
# PAGE - PREPROCESSING
# =====================================================================================
def page_preprocessing():
    render_hero("Preprocessing", "Configure the pipeline. Everything here is fit on the training split only.")
    if not require_target():
        return
 
    df = st.session_state.df
    target = st.session_state.target
    column_types = st.session_state.column_types
    constant_cols = st.session_state.get("constant_cols", [])
 
    candidates = [c for c in df.columns if c != target]
    auto_exclude = [c for c in candidates if column_types.get(c) == "id_like" or c in constant_cols]
 
    st.subheader("Feature selection")
    st.caption("ID-like and constant columns are pre-excluded because they carry no generalisable signal.")
    dropped = st.multiselect("Columns to EXCLUDE from modeling", candidates, default=auto_exclude)
    feature_cols = [c for c in candidates if c not in dropped]
 
    if not feature_cols:
        st.error("At least one feature column is required.")
        return
    st.caption(f"**{len(feature_cols)}** feature(s) will be used.")
 
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Missing values")
        num_label = st.selectbox("Numeric strategy", ["median", "mean", "constant (0)"], index=0)
        numeric_impute = {"mean": "mean", "median": "median", "constant (0)": "constant"}[num_label]
        cat_label = st.selectbox("Categorical strategy", ["mode (most frequent)", "constant ('Missing')"], index=0)
        categorical_impute = "mode" if cat_label.startswith("mode") else "constant"
        st.caption("Median resists outliers; mean preserves the average. A 'Missing' category keeps "
                   "missingness itself as signal, which matters when data is not missing at random.")
 
    with c2:
        st.subheader("Encoding & scaling")
        cardinality_threshold = st.slider(
            "High-cardinality threshold", 5, 200, 50, 5,
            help="Categorical columns with more unique values than this skip one-hot encoding.")
        high_card_encoding = st.radio("High-cardinality encoding", ["frequency", "ordinal"], horizontal=True)
        scale = st.checkbox("Apply StandardScaler to numeric features", value=True)
        drop_duplicate_rows = st.checkbox("Drop exact duplicate rows before splitting", value=False,
                                          help="Duplicates spanning train and test inflate test scores.")
 
    preprocess_config = {
        "dropped_cols": sorted(dropped),
        "numeric_impute": numeric_impute,
        "categorical_impute": categorical_impute,
        "cardinality_threshold": cardinality_threshold,
        "high_card_encoding": high_card_encoding,
        "scale": scale,
        "drop_duplicate_rows": drop_duplicate_rows,
    }
 
    if st.session_state.get("preprocess_config") != preprocess_config or \
       st.session_state.get("feature_cols") != feature_cols:
        st.session_state.pop("results", None)
    st.session_state.feature_cols = feature_cols
    st.session_state.preprocess_config = preprocess_config
 
    st.divider()
    st.subheader("Resulting pipeline")
 
    roles = resolve_feature_roles(df, feature_cols, column_types, cardinality_threshold)
    onehot_width = estimate_onehot_width(df, roles["categorical_cols"])
    est_width = (len(roles["numeric_base"]) + len(roles["boolean_cols"])
                 + len(roles["datetime_cols"]) * 4 + onehot_width + len(roles["high_card_cols"]))
 
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Numeric branch", len(roles["numeric_base"]) + len(roles["boolean_cols"]))
    r2.metric("One-hot branch", f"{len(roles['categorical_cols'])} cols")
    r3.metric("High-cardinality", f"{len(roles['high_card_cols'])} cols")
    r4.metric("Est. matrix width", f"{est_width:,}")
 
    if roles["datetime_cols"]:
        st.caption(f"Datetime column(s) {', '.join(roles['datetime_cols'])} expand into "
                   f"{len(roles['datetime_cols']) * 4} calendar features (year/month/day/day-of-week).")
    if est_width > 500:
        st.warning(f"The encoded matrix would be roughly **{est_width:,} columns** wide. Consider lowering the "
                   "high-cardinality threshold so more columns use frequency encoding instead of one-hot.")
    if onehot_width > 0:
        st.caption(f"One-hot encoding contributes {onehot_width:,} of those columns.")
 
    with st.expander("Per-column routing"):
        routing = pd.DataFrame({
            "column": feature_cols,
            "detected_type": [rebucket_column_types(df, feature_cols, column_types, cardinality_threshold)[c]
                              for c in feature_cols],
            "unique_values": [df[c].nunique(dropna=True) for c in feature_cols],
            "missing_pct": [round(df[c].isna().mean() * 100, 2) for c in feature_cols],
        }).set_index("column")
        st.dataframe(routing, use_container_width=True)
 
    st.download_button("Download this configuration (JSON)",
                       data=json.dumps({"target": target, "features": feature_cols,
                                        "preprocessing": preprocess_config}, indent=2).encode(),
                       file_name="tabula_config.json", mime="application/json")
 
    st.success("Preprocessing configured. Continue to **Modeling & Results**.")
 
 
# =====================================================================================
# PAGE - MODELING & RESULTS
# =====================================================================================
def build_comparison_frame(model_results, problem_type):
    if problem_type == "classification":
        data = {n: {"Accuracy": r["accuracy"], "Macro F1": r["f1_macro"],
                    "Precision": r["precision_macro"], "Recall": r["recall_macro"],
                    "Train (s)": r["train_seconds"]} for n, r in model_results.items()}
    else:
        data = {n: {"R2": r["r2"], "RMSE": r["rmse"], "MAE": r["mae"],
                    "Train (s)": r["train_seconds"]} for n, r in model_results.items()}
    frame = pd.DataFrame(data).T
    if any("cv_mean" in r for r in model_results.values()):
        frame["CV mean"] = [model_results[n].get("cv_mean", np.nan) for n in frame.index]
        frame["CV std"] = [model_results[n].get("cv_std", np.nan) for n in frame.index]
    return frame
 
 
def page_modeling():
    render_hero("Modeling & Results", "Models and metrics adapt to the detected task type.")
    if not require_target():
        return
    if "preprocess_config" not in st.session_state:
        st.warning("Configure the pipeline in **Preprocessing** first.")
        return
 
    df = st.session_state.df
    target = st.session_state.target
    problem_type = st.session_state.problem_type
    column_types = st.session_state.column_types
    feature_cols = st.session_state.feature_cols
    pre_cfg = st.session_state.preprocess_config
 
    if not XGBOOST_AVAILABLE:
        st.warning("`xgboost` is not installed - those models will be skipped. Install it with `pip install xgboost`.")
 
    # ---- sidebar hyperparameters ----------------------------------------------------------
    st.sidebar.divider()
    st.sidebar.subheader("Model settings")
    test_size = st.sidebar.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
    random_state = int(st.sidebar.number_input("Random seed", value=42, step=1))
    baseline_name = "Logistic Regression" if problem_type == "classification" else "Linear Regression"
    include_baseline = st.sidebar.checkbox(f"Include {baseline_name} baseline", value=True)
 
    mdl_cfg = {"test_size": test_size, "random_state": random_state, "include_baseline": include_baseline}
 
    if problem_type == "classification":
        mdl_cfg["balance_classes"] = st.sidebar.checkbox(
            "Use balanced class weights", value=False,
            help="Reweights classes inversely to frequency. Helps on imbalanced targets.")
    else:
        mdl_cfg["log_transform_target"] = st.sidebar.checkbox(
            "Log-transform target", value=False,
            help="Applies log1p to a positive, skewed target. Metrics are reported on the log scale.")
 
    with st.sidebar.expander("Random Forest", expanded=False):
        mdl_cfg["rf_n_estimators"] = st.slider("n_estimators", 50, 600, 200, 50, key="rf_n")
        depth = st.slider("max_depth (0 = unlimited)", 0, 40, 0, 1, key="rf_d")
        mdl_cfg["rf_max_depth"] = depth if depth > 0 else None
        mdl_cfg["rf_min_samples_leaf"] = st.slider("min_samples_leaf", 1, 20, 1, 1, key="rf_l")
 
    if XGBOOST_AVAILABLE:
        with st.sidebar.expander("XGBoost", expanded=False):
            mdl_cfg["xgb_n_estimators"] = st.slider("n_estimators", 50, 600, 200, 50, key="xgb_n")
            mdl_cfg["xgb_max_depth"] = st.slider("max_depth", 2, 15, 6, 1, key="xgb_d")
            mdl_cfg["xgb_learning_rate"] = st.slider("learning_rate", 0.01, 0.5, 0.1, 0.01, key="xgb_lr")
            mdl_cfg["xgb_subsample"] = st.slider("subsample", 0.5, 1.0, 1.0, 0.05, key="xgb_s")
 
    with st.sidebar.expander("Validation", expanded=False):
        mdl_cfg["use_cv"] = st.checkbox("Run cross-validation", value=False,
                                        help="More reliable than a single split, especially on small datasets.")
        mdl_cfg["cv_folds"] = st.slider("CV folds", 2, 10, 5, 1, disabled=not mdl_cfg["use_cv"])
        mdl_cfg["use_permutation_importance"] = st.checkbox(
            "Compute permutation importance", value=False,
            help="Measures importance by shuffling each feature on the test set. Slower, but less biased "
                 "toward high-cardinality features than impurity importance.")
 
    if st.sidebar.button("\U0001F680 Train model(s)", type="primary", use_container_width=True):
        st.session_state.train_requested = True
 
    if not st.session_state.get("train_requested"):
        st.info("Set your hyperparameters in the sidebar, then click **Train model(s)**.")
        return
 
    cache_key = compute_cache_key(df, target, problem_type, pre_cfg, mdl_cfg)
    with st.spinner("Training..."):
        try:
            results = train_models(df, target, problem_type, feature_cols, column_types,
                                   pre_cfg, mdl_cfg, cache_key)
        except Exception as e:  # noqa: BLE001
            st.error(f"Training failed: {e}")
            return
 
    st.session_state.results = results
    render_results(results, problem_type)
 
 
def render_results(results, problem_type):
    model_results = results["model_results"]
    best_name = results["best_name"]
    y_test = results["y_test"]
 
    st.success(f"Best model: **{best_name}** "
               f"(selected on {'macro F1' if problem_type == 'classification' else 'R\u00b2'}) - "
               f"trained on {results['n_train']:,} rows, tested on {results['n_test']:,}.")
    if results.get("split_warning"):
        st.warning(results["split_warning"])
    if results.get("target_log"):
        st.info("Target was log-transformed; regression metrics below are on the log scale.")
 
    st.subheader("Model comparison")
    comp = build_comparison_frame(model_results, problem_type)
    st.dataframe(comp.round(4), use_container_width=True)
 
    if "CV mean" in comp.columns:
        gap_col = "Macro F1" if problem_type == "classification" else "R2"
        gaps = (comp[gap_col] - comp["CV mean"]).abs()
        if (gaps > 0.10).any():
            worst = gaps.idxmax()
            st.warning(f"**{worst}** shows a {gaps.max():.2f} gap between its test score and cross-validated "
                       "mean. That usually means the single split is optimistic - trust the CV figure.")
 
    st.divider()
    tabs = st.tabs(list(model_results.keys()))
    for tab, name in zip(tabs, model_results.keys()):
        with tab:
            render_single_model(name, model_results[name], y_test, results, problem_type)
 
    st.divider()
    st.subheader("Export")
    best = model_results[best_name]
    d1, d2, d3 = st.columns(3)
    with d1:
        payload = pickle.dumps({
            "preprocessor": results["preprocessor"], "model": best["model"],
            "feature_names": results["feature_names"], "label_encoder": results["label_encoder"],
            "problem_type": problem_type, "feature_cols": results["feature_cols"],
            "roles": results["roles"], "target": results["target"],
        })
        st.download_button(f"Trained pipeline ({best_name})", data=payload,
                           file_name="tabula_pipeline.pkl", mime="application/octet-stream",
                           use_container_width=True)
    with d2:
        preds_df = results["X_test_raw"].copy()
        preds_df["actual"] = y_test.values
        preds_df["predicted"] = best["preds"]
        if results["label_encoder"]:
            inv = {v: k for k, v in results["label_encoder"]["map"].items()}
            preds_df["actual"] = preds_df["actual"].map(inv)
            preds_df["predicted"] = preds_df["predicted"].map(inv)
        st.download_button("Test predictions (CSV)", data=preds_df.to_csv(index=False).encode(),
                           file_name="tabula_predictions.csv", mime="text/csv", use_container_width=True)
    with d3:
        st.download_button("Metrics report (CSV)", data=comp.to_csv().encode(),
                           file_name="tabula_metrics.csv", mime="text/csv", use_container_width=True)
 
 
def render_single_model(name, r, y_test, results, problem_type):
    if problem_type == "classification":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{r['accuracy']:.3f}")
        c2.metric("Macro F1", f"{r['f1_macro']:.3f}")
        c3.metric("Precision", f"{r['precision_macro']:.3f}")
        c4.metric("Recall", f"{r['recall_macro']:.3f}")
        if "cv_mean" in r:
            st.caption(f"Cross-validated macro F1: **{r['cv_mean']:.3f} \u00b1 {r['cv_std']:.3f}** "
                       f"over {r.get('cv_folds_used', '?')} folds.")
 
        st.markdown("**Classification report**")
        st.dataframe(pd.DataFrame(r["report"]).T.round(3), use_container_width=True)
 
        classes = results["label_encoder"]["classes"]
        cm = r["confusion_matrix"]
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Confusion matrix (counts)**")
            fig = px.imshow(cm, x=[str(c) for c in classes], y=[str(c) for c in classes],
                            text_auto=True, color_continuous_scale="Blues",
                            labels=dict(x="Predicted", y="Actual", color="Count"))
            st.plotly_chart(fig, use_container_width=True, key=f"cm_{name}")
        with cc2:
            st.markdown("**Confusion matrix (row-normalised)**")
            cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            fig = px.imshow(cm_norm, x=[str(c) for c in classes], y=[str(c) for c in classes],
                            text_auto=".2f", color_continuous_scale="Blues", zmin=0, zmax=1,
                            labels=dict(x="Predicted", y="Actual", color="Rate"))
            st.plotly_chart(fig, use_container_width=True, key=f"cmn_{name}")
 
        if len(classes) == 2 and r.get("proba") is not None:
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.markdown("**ROC curve**")
                fpr, tpr, _ = roc_curve(y_test, r["proba"][:, 1])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {auc(fpr, tpr):.3f})"))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         line=dict(dash="dash"), name="Random"))
                fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate", height=380)
                st.plotly_chart(fig, use_container_width=True, key=f"roc_{name}")
            with pcol2:
                st.markdown("**Precision-recall curve**")
                prec, rec, _ = precision_recall_curve(y_test, r["proba"][:, 1])
                ap = average_precision_score(y_test, r["proba"][:, 1])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"PR (AP = {ap:.3f})"))
                fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=380)
                st.plotly_chart(fig, use_container_width=True, key=f"pr_{name}")
                st.caption("On imbalanced targets the PR curve is more informative than ROC, because "
                           "ROC can look strong even when positives are rarely caught.")
 
            st.markdown("**Decision threshold tuning**")
            thr = st.slider("Classification threshold", 0.05, 0.95, 0.50, 0.01, key=f"thr_{name}")
            tuned = (r["proba"][:, 1] >= thr).astype(int)
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Accuracy", f"{accuracy_score(y_test, tuned):.3f}")
            t2.metric("Macro F1", f"{f1_score(y_test, tuned, average='macro'):.3f}")
            t3.metric("Precision", f"{precision_score(y_test, tuned, zero_division=0):.3f}")
            t4.metric("Recall", f"{recall_score(y_test, tuned, zero_division=0):.3f}")
            st.caption(f"Positive class: `{classes[1]}`. Lower the threshold to catch more positives "
                       "at the cost of false alarms; raise it for higher precision.")
 
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R\u00b2", f"{r['r2']:.3f}")
        c2.metric("RMSE", f"{r['rmse']:,.3f}")
        c3.metric("MAE", f"{r['mae']:,.3f}")
        c4.metric("Train time", f"{r['train_seconds']:.2f}s")
        if "cv_mean" in r:
            st.caption(f"Cross-validated R\u00b2: **{r['cv_mean']:.3f} \u00b1 {r['cv_std']:.3f}** "
                       f"over {r.get('cv_folds_used', '?')} folds.")
 
        y_vals = y_test.values
        resid = y_vals - r["preds"]
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Predicted vs actual**")
            fig = px.scatter(x=y_vals, y=r["preds"], labels={"x": "Actual", "y": "Predicted"},
                             opacity=0.65)
            lo, hi = float(min(y_vals.min(), r["preds"].min())), float(max(y_vals.max(), r["preds"].max()))
            fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="gray"))
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True, key=f"pva_{name}")
        with g2:
            st.markdown("**Residuals vs predicted**")
            fig = px.scatter(x=r["preds"], y=resid, labels={"x": "Predicted", "y": "Residual"}, opacity=0.65)
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True, key=f"resid_{name}")
 
        st.markdown("**Residual distribution**")
        fig = px.histogram(x=resid, nbins=40, labels={"x": "Residual"})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True, key=f"residhist_{name}")
        st.caption(f"Residual mean {resid.mean():,.3f}, std {resid.std():,.3f}. A visible pattern in the "
                   "residual scatter means the model is missing structure; a curve or funnel shape is the "
                   "usual tell.")
 
    # ---- importance ----------------------------------------------------------------------------
    has_impurity = "feature_importances" in r
    has_perm = "permutation_importances" in r
    if has_impurity or has_perm:
        st.markdown("**Feature importance**")
        choices = []
        if has_impurity:
            choices.append(r.get("importance_kind", "impurity"))
        if has_perm:
            choices.append("permutation")
        kind = st.radio("Importance type", choices, horizontal=True, key=f"imp_{name}") if len(choices) > 1 else choices[0]
 
        source = r["permutation_importances"] if kind == "permutation" else r["feature_importances"]
        fi = pd.Series(source).sort_values(ascending=True).tail(20)
        fig = px.bar(x=fi.values, y=fi.index, orientation="h",
                     labels={"x": f"{kind} importance", "y": "Feature"})
        fig.update_layout(height=max(320, 22 * len(fi)))
        st.plotly_chart(fig, use_container_width=True, key=f"fi_{name}_{kind}")
        if kind != "permutation":
            st.caption("Impurity importance is biased toward high-cardinality and continuous features. "
                       "Enable permutation importance in the sidebar for a less biased view.")
 
 
# =====================================================================================
# PAGE - SCORE NEW DATA
# =====================================================================================
def page_scoring():
    render_hero("Score New Data", "Apply your trained pipeline to a fresh CSV.")
    results = st.session_state.get("results")
    if not results:
        st.warning("Train a model first in **Modeling & Results**.")
        return
 
    best_name = results["best_name"]
    feature_cols = results["feature_cols"]
    st.info(f"Using **{best_name}**, trained to predict `{results['target']}` "
            f"from {len(feature_cols)} feature(s).")
 
    with st.expander("Required columns"):
        st.write(feature_cols)
        template = pd.DataFrame(columns=feature_cols)
        st.download_button("Download blank template (CSV)", data=template.to_csv(index=False).encode(),
                           file_name="tabula_scoring_template.csv", mime="text/csv")
 
    new_file = st.file_uploader("Upload a CSV to score", type=["csv"], key="score_upload")
    if new_file is None:
        st.caption("The file needs the same feature columns used in training. The target column is optional; "
                   "if present, accuracy against it is reported.")
        return
 
    try:
        new_df, _, _ = load_data(new_file.getvalue(), new_file.name)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read that file: {e}")
        return
 
    missing_cols = [c for c in feature_cols if c not in new_df.columns]
    if missing_cols:
        st.error(f"This file is missing {len(missing_cols)} required column(s): {', '.join(missing_cols[:10])}")
        return
 
    try:
        scoring_df = new_df[feature_cols].copy()
        engineered, _ = apply_feature_engineering(scoring_df, results["roles"])
        expected = list(results["preprocessor"].feature_names_in_)
        for col in expected:
            if col not in engineered.columns:
                engineered[col] = np.nan
        transformed = results["preprocessor"].transform(engineered[expected])
        model = results["model_results"][best_name]["model"]
        preds = model.predict(transformed)
    except Exception as e:  # noqa: BLE001
        st.error(f"Scoring failed: {e}")
        return
 
    out = new_df.copy()
    if results["label_encoder"]:
        inv = {v: k for k, v in results["label_encoder"]["map"].items()}
        out["prediction"] = pd.Series(preds).map(inv).values
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(transformed)
            out["prediction_confidence"] = proba.max(axis=1).round(4)
    else:
        preds_out = np.expm1(preds) if results.get("target_log") else preds
        out["prediction"] = preds_out
 
    st.success(f"Scored **{len(out):,} rows**.")
 
    s1, s2 = st.columns([1, 1.3])
    with s1:
        if results["label_encoder"]:
            vc = out["prediction"].value_counts()
            st.markdown("**Prediction breakdown**")
            st.dataframe(vc.rename("count").to_frame(), use_container_width=True)
            if "prediction_confidence" in out.columns:
                low_conf = (out["prediction_confidence"] < 0.6).sum()
                st.metric("Low-confidence rows (<0.60)", f"{low_conf:,}")
        else:
            st.markdown("**Prediction summary**")
            st.dataframe(out["prediction"].describe().to_frame().round(3), use_container_width=True)
    with s2:
        if results["label_encoder"]:
            vc = out["prediction"].value_counts()
            fig = px.bar(x=vc.index, y=vc.values, labels={"x": "prediction", "y": "count"})
        else:
            fig = px.histogram(out, x="prediction", nbins=40)
        fig.update_layout(height=300, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True, key="score_dist")
 
    target_name = results["target"]
    if target_name in new_df.columns and new_df[target_name].notna().any():
        st.divider()
        st.subheader("Accuracy against supplied ground truth")
        mask = new_df[target_name].notna()
        try:
            if results["label_encoder"]:
                acc = accuracy_score(new_df.loc[mask, target_name].astype(str), out.loc[mask, "prediction"].astype(str))
                st.metric("Accuracy on this file", f"{acc:.3f}")
            else:
                truth = pd.to_numeric(new_df.loc[mask, target_name], errors="coerce")
                pred = pd.to_numeric(out.loc[mask, "prediction"], errors="coerce")
                ok = truth.notna() & pred.notna()
                e1, e2 = st.columns(2)
                e1.metric("R\u00b2", f"{r2_score(truth[ok], pred[ok]):.3f}")
                e2.metric("RMSE", f"{np.sqrt(mean_squared_error(truth[ok], pred[ok])):,.3f}")
        except Exception as e:  # noqa: BLE001
            st.warning(f"Could not evaluate against the supplied target: {e}")
 
    st.download_button("Download scored file (CSV)", data=out.to_csv(index=False).encode(),
                       file_name="tabula_scored.csv", mime="text/csv", type="primary")
    st.dataframe(out.head(100), use_container_width=True)
 
 
# =====================================================================================
# MAIN
# =====================================================================================
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
 
    st.sidebar.markdown(f"## \u25F0 {APP_NAME}")
    st.sidebar.caption(APP_TAGLINE)
 
    page = st.sidebar.radio("Navigate", [
        "Dashboard", "Data & Profiling", "Explore", "Preprocessing",
        "Modeling & Results", "Score New Data",
    ])
 
    st.sidebar.divider()
    st.sidebar.markdown("**Progress**")
    render_workflow_status()
 
    if "df" in st.session_state:
        st.sidebar.divider()
        df = st.session_state.df
        st.sidebar.caption(f"{df.shape[0]:,} rows x {df.shape[1]} cols")
        if st.session_state.get("target"):
            st.sidebar.caption(f"Target: `{st.session_state.target}` ({st.session_state.problem_type})")
        if st.sidebar.button("Reset session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
 
    pages = {
        "Dashboard": page_dashboard,
        "Data & Profiling": page_data_upload,
        "Explore": page_eda,
        "Preprocessing": page_preprocessing,
        "Modeling & Results": page_modeling,
        "Score New Data": page_scoring,
    }
    pages[page]()
 
 
if __name__ == "__main__":
    main()
 
