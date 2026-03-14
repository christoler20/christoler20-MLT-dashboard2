"""
MLT Career Prep · Offer Prediction Dashboard
=============================================
Single-file Streamlit application for Management Leadership for Tomorrow.

Model  : Further-Reduced LASSO Logistic Regression (L1, balanced weights, 5-fold CV)
Train  : CP 2018 – 2023
Validate: CP 2024
Score  : CP 2025

Author : MLT Analytics Team
"""

# ═══════════════════════════════════════════════════════════════════════
# SECTION 0 — Imports & Page Config
# ═══════════════════════════════════════════════════════════════════════
import os, re, warnings, textwrap
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score,
    f1_score, accuracy_score, confusion_matrix,
)

st.set_page_config(
    page_title="MLT Career Prep · Offer Prediction Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — Constants
# ═══════════════════════════════════════════════════════════════════════
MODEL_VERSION = "Further-Reduced LASSO v2.0"

POSITIVE_STATUSES = [
    "Offered", "Offered & Committed", "Offered & Declined",
    "Offer Rescinded", "My offer has been rescinded.",
]
NEGATIVE_STATUSES = ["Denied", "Pending"]
ALL_OUTCOME_STATUSES = POSITIVE_STATUSES + NEGATIVE_STATUSES

TRAIN_COHORTS = ["CP 2018", "CP 2020", "CP 2021", "CP 2022", "CP 2023"]
VALIDATION_COHORT = "CP 2024"
THRESHOLD = 0.50

FORTUNE_500 = {
    "Amazon", "Target", "Google", "Visa Inc.", "Dell Technologies Inc.",
    "Citi", "AT&T", "Morgan Stanley", "JPMorgan Chase", "Goldman Sachs",
    "Bank of America", "Wells Fargo", "Microsoft", "Apple", "Meta",
    "Meta Platforms", "IBM", "Intel", "Oracle", "Cisco",
    "Johnson & Johnson", "Procter & Gamble", "PepsiCo", "Coca-Cola",
    "Nike", "Walt Disney", "Netflix", "Salesforce", "Adobe", "PayPal",
    "American Express", "Capital One", "T-Mobile", "Verizon",
    "Home Depot", "Walmart", "Costco", "General Electric",
    "Honeywell", "Lockheed Martin", "Boeing", "General Motors", "Ford",
    "ExxonMobil", "Chevron", "Pfizer", "Merck", "Eli Lilly",
    "Accenture", "Uber", "Mastercard", "Starbucks Coffee Company",
    "LinkedIn", "Deloitte", "BlackRock", "Charles Schwab",
    "Kearney", "Boston Consulting", "McKinsey", "Bain",
    "FICO", "Chick-Fil-A",
}

LIKELIHOOD_COLORS = {"Red": "#DC2626", "Yellow": "#F59E0B", "Green": "#059669"}
LIKELIHOOD_LABELS = {
    "Red": "High Support Needed",
    "Yellow": "Moderate Support Needed",
    "Green": "Likely Competitive",
}

OUTPUT_COLS = [
    "Program Enrollment: Enrollment ID",
    "Program Enrollment: Coach",
    "Program Enrollment: Program Track",
    "Related Organization", "Title", "Primary Functional Interest",
    "Predicted_Probability", "Risk_Flag", "Predicted_Label",
    "Predicted_Outcome", "Actual_Label", "Actual_Outcome",
    "Correct", "Suggested_Coach_Action", "Coach_Notes",
    "Likely_Role_Alignment",
]


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — CSS
# ═══════════════════════════════════════════════════════════════════════
CSS = """
<style>
.main .block-container{padding-top:1rem;padding-bottom:2rem}
section[data-testid="stSidebar"]{background:#F8F9FB;border-right:1px solid #E5E7EB}

.dash-header{background:linear-gradient(135deg,#1B2A4A 0%,#2C5F8A 100%);color:#fff;
 padding:1.5rem 2rem;border-radius:14px;margin-bottom:1.2rem}
.dash-header h1{font-size:1.55rem;font-weight:700;margin:0;letter-spacing:-0.3px}
.dash-header p{font-size:.82rem;opacity:.85;margin:.3rem 0 0;font-weight:400}

.kpi-card{background:#fff;border-radius:12px;padding:1rem 1.2rem;
 box-shadow:0 1px 4px rgba(0,0,0,.07);text-align:center;
 border-top:3px solid #2C5F8A;min-height:100px}
.kpi-card.accent-green{border-top-color:#059669}
.kpi-card.accent-red{border-top-color:#DC2626}
.kpi-card.accent-amber{border-top-color:#F59E0B}
.kpi-value{font-size:1.65rem;font-weight:700;color:#1A1A2E;
 margin:.25rem 0 .15rem;line-height:1.2}
.kpi-label{font-size:.7rem;color:#6B7280;text-transform:uppercase;
 letter-spacing:.6px;font-weight:600}

.section-card{background:#fff;border-radius:12px;padding:1.4rem 1.6rem;
 box-shadow:0 1px 3px rgba(0,0,0,.06);margin-bottom:1rem;border:1px solid #F0F0F3}
.section-title{font-size:1.05rem;font-weight:700;color:#1B2A4A;margin-bottom:.6rem}
.section-caption{font-size:.78rem;color:#6B7280;margin-top:-.3rem;margin-bottom:.8rem}

.risk-badge{display:inline-block;padding:3px 10px;border-radius:12px;
 font-size:.72rem;font-weight:600;letter-spacing:.3px}
.risk-badge.red{background:#FEE2E2;color:#DC2626}
.risk-badge.yellow{background:#FEF3C7;color:#D97706}
.risk-badge.green{background:#D1FAE5;color:#059669}

.legend-row{display:flex;gap:1.5rem;flex-wrap:wrap;align-items:center;margin-bottom:.8rem}
.legend-item{display:flex;align-items:center;gap:6px;font-size:.78rem;color:#374151}
.legend-dot{width:12px;height:12px;border-radius:50%;display:inline-block}

.stTabs [data-baseweb="tab-list"] button{font-size:.85rem;font-weight:600;padding:.6rem 1.2rem}
.stDataFrame{font-size:.82rem}

.info-wrap{position:relative;display:inline-block;cursor:pointer;
 vertical-align:middle;margin-left:4px}
.info-icon{display:inline-flex;align-items:center;justify-content:center;
 width:18px;height:18px;border-radius:50%;background:#E5E7EB;color:#6B7280;
 font-size:11px;font-weight:700;font-style:normal;line-height:1;
 transition:background .15s,color .15s}
.info-wrap:hover .info-icon{background:#2C5F8A;color:#fff}
.info-tip{visibility:hidden;opacity:0;position:absolute;z-index:9999;
 bottom:calc(100% + 10px);left:50%;transform:translateX(-50%);width:260px;
 background:#1B2A4A;color:#F3F4F6;padding:10px 14px;border-radius:10px;
 font-size:.74rem;font-weight:400;line-height:1.45;
 box-shadow:0 4px 16px rgba(0,0,0,.18);pointer-events:none;
 transition:opacity .18s,visibility .18s;text-align:left}
.info-tip::after{content:"";position:absolute;top:100%;left:50%;
 transform:translateX(-50%);border-width:6px;border-style:solid;
 border-color:#1B2A4A transparent transparent transparent}
.info-wrap:hover .info-tip{visibility:visible;opacity:1}

.sidebar-brand{font-size:.78rem;color:#6B7280;border-top:1px solid #E5E7EB;
 padding-top:1rem;margin-top:1.5rem}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — Helper Functions
# ═══════════════════════════════════════════════════════════════════════

def safe_col(df, col, default=np.nan):
    """Return column if present, else a Series of *default*."""
    return df[col] if col in df.columns else pd.Series(default, index=df.index, name=col)


def info_icon(tip):
    """Inline HTML info-icon with hover tooltip."""
    return (
        f'<span class="info-wrap">'
        f'<span class="info-icon">i</span>'
        f'<span class="info-tip">{tip}</span>'
        f'</span>'
    )


def kpi_html(label, value, accent="", tooltip=""):
    """HTML for a single KPI card."""
    cls = f"kpi-card {accent}" if accent else "kpi-card"
    tip = info_icon(tooltip) if tooltip else ""
    return (
        f'<div class="{cls}">'
        f'<div class="kpi-label">{label} {tip}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'</div>'
    )


def legend_html():
    """Red/Yellow/Green legend row."""
    items = [
        ("#DC2626", "Red — High Support Needed"),
        ("#F59E0B", "Yellow — Moderate Support Needed"),
        ("#059669", "Green — Likely Competitive"),
    ]
    parts = "".join(
        f'<span class="legend-item">'
        f'<span class="legend-dot" style="background:{c}"></span>{lbl}</span>'
        for c, lbl in items
    )
    return f'<div class="legend-row">{parts}</div>'


def plotly_clean(fig, height=420):
    """Apply executive styling to a Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=40, r=20, t=30, b=40),
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#374151"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )


def assign_likelihood(prob):
    """Map predicted probability → likelihood flag."""
    if prob < 0.35:
        return "Red"
    if prob <= 0.60:
        return "Yellow"
    return "Green"


def suggest_action(flag):
    """Return suggested coach action for a likelihood flag."""
    return {
        "Red": "Immediate intervention: refine strategy, target fit, interview prep, and outreach",
        "Yellow": "Moderate coaching: strengthen positioning, sharpen application materials",
        "Green": "Maintain momentum: prepare for interviews and close opportunities",
    }.get(flag, "")


def role_alignment(row):
    """Heuristic: does functional interest align with the role title?"""
    title = str(row.get("Title", "")).lower()
    func = str(row.get("Primary Functional Interest", "")).lower()
    kw_map = {
        "consulting": ["consulting", "consultant", "strategy", "advisory"],
        "software": ["software", "engineer", "developer", "swe", "programming"],
        "marketing": ["marketing", "brand", "digital", "growth", "content"],
        "finance": ["finance", "banking", "analyst", "investment", "wealth"],
        "product": ["product", "pm", "product manager"],
        "operations": ["operations", "supply chain", "logistics"],
        "human resources": ["hr", "human resources", "talent", "people"],
        "research": ["research", "analytics", "data", "scientist"],
        "sales": ["sales", "business development", "account"],
        "engineering": ["engineering", "mechanical", "electrical"],
    }
    matched = None
    for key, terms in kw_map.items():
        if any(t in func for t in terms):
            matched = key
            break
    if matched is None:
        return "Moderate"
    if any(t in title for t in kw_map[matched]):
        return "Strong"
    if title in ("intern", "internship", "") or not title:
        return "Moderate"
    return "Low"


def readable_feature(name):
    """Convert internal feature name → human-readable label."""
    static = {
        "Undergrad_GPA": "Undergraduate GPA",
        "Pell_Grant_Count": "Pell Grant Count",
        "SAT_Score": "SAT Score",
        "SAT_Available": "SAT Score Reported",
        "Title_Word_Count": "Job Title Word Count",
        "Designated_Low_Income": "Designated Low Income",
        "First_Gen_College": "First-Generation College Student",
        "Is_Partner_Active": "Active MLT Partner Organization",
        "Is_Rising_Junior": "Rising Junior Internship",
        "Is_Fortune500": "Fortune 500 Company",
    }
    if name in static:
        return static[name]
    if name.startswith("Func_"):
        return "Interest: " + name[5:].replace("_", " ").strip()
    if name.startswith("Ind_"):
        return "Industry: " + name[4:].replace("_", " ").strip()
    if name.startswith("Track_"):
        return "Track: " + name[6:].replace("_", " ").strip()
    return name.replace("_", " ")


def compute_fairness(df, group_col, actual_col="Actual_Label",
                     pred_col="Predicted_Label", prob_col="Predicted_Probability"):
    """Compute subgroup fairness metrics. Returns DataFrame or None."""
    if group_col not in df.columns:
        return None
    subset = df.dropna(subset=[group_col, actual_col])
    if len(subset) == 0:
        return None
    rows = []
    for grp, gdf in subset.groupby(group_col):
        n = len(gdf)
        if n < 5:
            continue
        y_true = gdf[actual_col].astype(int).values
        y_pred = gdf[pred_col].astype(int).values
        y_prob = gdf[prob_col].values
        if len(np.unique(y_true)) < 2:
            rows.append({
                "Subgroup": grp, "Count": n,
                "Actual Offer Rate": round(y_true.mean(), 3),
                "Avg Predicted Prob": round(y_prob.mean(), 3),
                "Precision": None, "Recall": None,
                "FPR": None, "FNR": None,
            })
            continue
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append({
            "Subgroup": grp, "Count": n,
            "Actual Offer Rate": round(y_true.mean(), 3),
            "Avg Predicted Prob": round(y_prob.mean(), 3),
            "Precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "Recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
            "FPR": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 3),
            "FNR": round(fn / (fn + tp) if (fn + tp) > 0 else 0, 3),
        })
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("Count", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4 — Feature Engineering
# ═══════════════════════════════════════════════════════════════════════

def get_feature_config(train_df):
    """Derive encoding lists and medians from training data."""
    top_func = (
        train_df["Primary Functional Interest"].dropna()
        .value_counts().head(8).index.tolist()
    )
    top_ind = (
        train_df["Primary Industry Interest"].dropna()
        .value_counts().head(8).index.tolist()
    )
    medians = {
        "GPA": float(train_df["Undergrad GPA"].median()),
        "SAT": float(train_df["SAT Score"].median()),
    }
    return {"top_func": top_func, "top_ind": top_ind, "medians": medians}


def build_features(df, config):
    """Build the model feature matrix from raw data + config."""
    med = config["medians"]
    feat = pd.DataFrame(index=df.index)

    # Numeric
    feat["Undergrad_GPA"] = df["Undergrad GPA"].fillna(med["GPA"])
    feat["Pell_Grant_Count"] = df["Pell Grant Count"].fillna(0).astype(float)
    feat["SAT_Score"] = df["SAT Score"].fillna(med["SAT"])
    feat["SAT_Available"] = df["SAT Score"].notna().astype(int)
    feat["Title_Word_Count"] = (
        df["Title"].fillna("").apply(lambda x: len(str(x).split()))
    )

    # Binary
    feat["Designated_Low_Income"] = safe_col(df, "Designated Low Income", False).astype(int)
    feat["First_Gen_College"] = (
        safe_col(df, "First Generation College", "No") == "Yes"
    ).astype(int)
    feat["Is_Partner_Active"] = (
        safe_col(df, "Partner Org?", "") == "Partner - Active"
    ).astype(int)
    feat["Is_Rising_Junior"] = (
        safe_col(df, "Type", "") == "Internship (Rising Junior)"
    ).astype(int)
    feat["Is_Fortune500"] = (
        df["Related Organization"].fillna("")
        .apply(lambda x: 1 if str(x).strip() in FORTUNE_500 else 0)
    )

    # Program track dummies
    tracks = ["Corporate Management", "Software Engineering/Technology",
              "Finance", "Consulting"]
    for t in tracks:
        col_name = "Track_" + re.sub(r"[^a-zA-Z0-9]", "_", t)
        feat[col_name] = (
            safe_col(df, "Program Enrollment: Program Track", "") == t
        ).astype(int)

    # Top functional interests
    for f in config["top_func"]:
        col_name = "Func_" + re.sub(r"[^a-zA-Z0-9]", "_", f)[:30]
        feat[col_name] = (
            df["Primary Functional Interest"].fillna("") == f
        ).astype(int)

    # Top industry interests
    for ind in config["top_ind"]:
        col_name = "Ind_" + re.sub(r"[^a-zA-Z0-9]", "_", ind)[:30]
        feat[col_name] = (
            df["Primary Industry Interest"].fillna("") == ind
        ).astype(int)

    return feat


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5 — Pipeline (cached)
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def run_pipeline():
    """Train on CP 2018-2023, validate on CP 2024, score CP 2025."""

    # ── Load data ─────────────────────────────────────────────
    base = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base, "data", "train_set.xlsx")
    test_path = os.path.join(base, "data", "test_set.xlsx")

    if not os.path.exists(train_path):
        train_path = os.path.join(base, "train_set.xlsx")
    if not os.path.exists(test_path):
        test_path = os.path.join(base, "test_set.xlsx")

    for p, label in [(train_path, "train_set.xlsx"), (test_path, "test_set.xlsx")]:
        if not os.path.exists(p):
            return {"error": f"Cannot find {label}. Place it in the data/ folder."}

    train_full = pd.read_excel(train_path)
    test_raw = pd.read_excel(test_path)

    # Check expected columns
    expected = [
        "Application Status", "Undergrad GPA", "SAT Score",
        "Pell Grant Count", "Designated Low Income",
        "First Generation College", "Primary Functional Interest",
        "Primary Industry Interest", "Title", "Type",
        "Related Organization", "Partner Org?",
        "Program Enrollment: Program Track", "Program Enrollment: Program",
        "Program Enrollment: Enrollment ID", "Program Enrollment: Coach",
    ]
    missing_cols = [c for c in expected if c not in train_full.columns]

    # ── Filter training cohorts ───────────────────────────────
    train = train_full[
        train_full["Program Enrollment: Program"].isin(TRAIN_COHORTS)
    ].copy()
    train = train[train["Application Status"].isin(ALL_OUTCOME_STATUSES)].copy()
    train["Offered"] = train["Application Status"].isin(POSITIVE_STATUSES).astype(int)

    # ── Exclude MLT events & resume drops ─────────────────────
    event_kw = ["mlt event", "mlt panel", "mlt conference", "networking event"]
    resume_kw = ["resume drop", "resume book", "resume collection"]
    if "Title" in train.columns:
        title_lower = train["Title"].fillna("").str.lower()
        is_event = title_lower.apply(lambda t: any(k in t for k in event_kw))
        is_resume = title_lower.apply(lambda t: any(k in t for k in resume_kw))
        train = train[~(is_event | is_resume)].copy()

    # ── Validation cohort ─────────────────────────────────────
    val_full = train_full[
        train_full["Program Enrollment: Program"] == VALIDATION_COHORT
    ].copy()
    val = val_full[val_full["Application Status"].isin(ALL_OUTCOME_STATUSES)].copy()
    val["Offered"] = val["Application Status"].isin(POSITIVE_STATUSES).astype(int)

    # ── Scoring cohort ────────────────────────────────────────
    score_all = test_raw.copy()
    score_all["Offered"] = np.where(
        score_all["Application Status"].isin(POSITIVE_STATUSES), 1,
        np.where(score_all["Application Status"].isin(NEGATIVE_STATUSES), 0, np.nan),
    )

    # ── Feature engineering ───────────────────────────────────
    config = get_feature_config(train)
    X_train = build_features(train, config)
    X_val = build_features(val, config)
    X_score = build_features(score_all, config)
    y_train = train["Offered"].values

    # ── Scale ─────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_score_sc = scaler.transform(X_score)

    # ── Train LASSO ───────────────────────────────────────────
    model = LogisticRegressionCV(
        Cs=20, penalty="l1", solver="saga", class_weight="balanced",
        cv=5, scoring="roc_auc", max_iter=5000, random_state=42,
    )
    model.fit(X_train_sc, y_train)
    best_C = float(model.C_[0])
    coefs = model.coef_[0]
    intercept = float(model.intercept_[0])
    feature_names = X_train.columns.tolist()

    # ── Coefficient table ─────────────────────────────────────
    coef_df = (
        pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
        .assign(Abs=lambda d: d["Coefficient"].abs())
        .query("Coefficient != 0")
        .sort_values("Abs", ascending=False)
        .drop(columns="Abs")
        .reset_index(drop=True)
    )

    # ── Predict ───────────────────────────────────────────────
    def _predict(X_sc):
        logits = intercept + X_sc @ coefs
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= THRESHOLD).astype(int)
        return probs, preds

    val_probs, val_preds = _predict(X_val_sc)
    score_probs, score_preds = _predict(X_score_sc)
    y_val = val["Offered"].values

    # ── Validation metrics ────────────────────────────────────
    val_metrics = {
        "Precision": round(precision_score(y_val, val_preds, zero_division=0), 4),
        "Recall": round(recall_score(y_val, val_preds, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_val, val_probs), 4),
        "F1": round(f1_score(y_val, val_preds, zero_division=0), 4),
        "Accuracy": round(accuracy_score(y_val, val_preds), 4),
        "Total": int(len(y_val)),
        "Predicted_Offered": int(val_preds.sum()),
        "Predicted_Denied": int((val_preds == 0).sum()),
        "Avg_Prob": round(float(val_probs.mean()), 4),
    }

    # ── Build output DataFrames ───────────────────────────────
    def _build_output(source_df, probs, preds):
        out = source_df.copy()
        out["Predicted_Probability"] = np.round(probs, 4)
        out["Risk_Flag"] = out["Predicted_Probability"].apply(assign_likelihood)
        out["Predicted_Label"] = preds
        out["Predicted_Outcome"] = np.where(preds == 1, "Offered", "Denied")
        has_actual = out["Offered"].notna()
        out["Actual_Label"] = out["Offered"]
        out["Actual_Outcome"] = np.where(
            out["Offered"] == 1, "Offered",
            np.where(out["Offered"] == 0, "Denied", "Unknown"),
        )
        out["Correct"] = np.where(
            has_actual,
            (preds == out["Offered"].fillna(-1).astype(int)).astype(int),
            np.nan,
        )
        out["Suggested_Coach_Action"] = out["Risk_Flag"].apply(suggest_action)
        out["Coach_Notes"] = ""
        out["Likely_Role_Alignment"] = out.apply(role_alignment, axis=1)
        return out

    val_out = _build_output(val, val_probs, val_preds)
    score_out = _build_output(score_all, score_probs, score_preds)

    # ── Scoring-cohort metrics ────────────────────────────────
    score_eval = score_out[score_out["Offered"].notna()].copy()
    score_metrics = {}
    if len(score_eval) > 0 and score_eval["Offered"].nunique() > 1:
        y_st = score_eval["Actual_Label"].astype(int).values
        y_sp = score_eval["Predicted_Label"].values
        y_sr = score_eval["Predicted_Probability"].values
        score_metrics = {
            "Precision": round(precision_score(y_st, y_sp, zero_division=0), 4),
            "Recall": round(recall_score(y_st, y_sp, zero_division=0), 4),
            "ROC_AUC": round(roc_auc_score(y_st, y_sr), 4),
            "F1": round(f1_score(y_st, y_sp, zero_division=0), 4),
            "Accuracy": round(accuracy_score(y_st, y_sp), 4),
            "Total_Eval": int(len(y_st)),
        }

    return {
        "val_out": val_out,
        "score_out": score_out,
        "val_metrics": val_metrics,
        "score_metrics": score_metrics,
        "coef_df": coef_df,
        "feature_names": feature_names,
        "coefs_full": coefs,
        "intercept": intercept,
        "best_C": best_C,
        "scaler": scaler,
        "train_n": len(train),
        "val_n": len(val),
        "score_n": len(score_all),
        "config": config,
        "X_score": X_score,
        "X_val": X_val,
        "missing_cols": missing_cols,
    }


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6 — Run Pipeline & Header
# ═══════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="dash-header">'
    "<h1>MLT Career Prep Offer Prediction Dashboard</h1>"
    "<p>Executive and Coach View for Offer Likelihood, "
    "Support Flags, and Fairness Monitoring</p>"
    "</div>",
    unsafe_allow_html=True,
)

with st.spinner("Loading data and training model … this may take a moment on first run."):
    results = run_pipeline()

if isinstance(results, dict) and "error" in results:
    st.error(results["error"])
    st.stop()

# Unpack
val_out = results["val_out"]
score_out = results["score_out"]
val_metrics = results["val_metrics"]
score_metrics = results["score_metrics"]
coef_df = results["coef_df"]
best_C = results["best_C"]
feature_names = results["feature_names"]
intercept = results["intercept"]
coefs_full = results["coefs_full"]
scaler = results["scaler"]
config = results["config"]

if results["missing_cols"]:
    st.warning(
        f"Some expected columns were not found: {', '.join(results['missing_cols'])}"
    )

# ═══════════════════════════════════════════════════════════════════════
# SECTION 7 — Sidebar
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"### {MODEL_VERSION}")
    st.markdown("---")

    cohort = st.radio(
        "Active Cohort",
        ["CP 2025 – Current Fellows", "CP 2024 – Validation"],
        index=0,
        help="Select which cohort to display.",
    )
    is_val_view = cohort.startswith("CP 2024")
    active_df = val_out if is_val_view else score_out
    cohort_label = "CP 2024" if is_val_view else "CP 2025"

    st.markdown("---")
    st.markdown("### Data Summary")
    st.markdown(f"**Train rows:** {results['train_n']:,}")
    st.markdown(f"**Validation (CP 2024):** {results['val_n']:,}")
    st.markdown(f"**Scoring (CP 2025):** {results['score_n']:,}")
    st.markdown(f"**Threshold:** {THRESHOLD}")
    st.markdown(f"**Best C:** {best_C:.4f}")
    st.markdown(f"**Non-zero features:** {len(coef_df)}/{len(feature_names)}")

    st.markdown("---")
    st.markdown("### Likelihood Flags")
    st.markdown("- 🔴 **High Support** — prob < 0.35")
    st.markdown("- 🟡 **Moderate Support** — 0.35 – 0.60")
    st.markdown("- 🟢 **Likely Competitive** — prob > 0.60")

    st.markdown("---")
    st.markdown("### Downloads")

    pred_cols = [c for c in OUTPUT_COLS if c in active_df.columns]
    pred_csv = active_df[pred_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        f"📥 Predictions ({cohort_label})", pred_csv,
        "further_reduced_lasso_predictions.csv", "text/csv",
    )

    metrics_rows = [{"Cohort": "CP 2024 Validation", **val_metrics}]
    if score_metrics:
        metrics_rows.append({"Cohort": "CP 2025 Scoring", **score_metrics})
    metrics_csv = pd.DataFrame(metrics_rows).to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Model Metrics", metrics_csv,
        "further_reduced_lasso_metrics.csv", "text/csv",
    )

    coef_csv = coef_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Coefficients", coef_csv,
        "further_reduced_lasso_coefficients.csv", "text/csv",
    )

    st.markdown(
        f'<div class="sidebar-brand">MLT Career Prep Analytics<br>{MODEL_VERSION}</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# SECTION 8 — KPI Strip
# ═══════════════════════════════════════════════════════════════════════
total_scored = len(active_df)
pred_offered = int(active_df["Predicted_Label"].sum())
pred_denied = total_scored - pred_offered
avg_prob = active_df["Predicted_Probability"].mean()

k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
with k1:
    st.markdown(kpi_html(
        f"{cohort_label} Scored", f"{total_scored:,}", "",
        "Total applications in the active cohort scored by the model."
    ), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_html(
        "Pred. Offered", f"{pred_offered:,}", "accent-green",
        "Applications predicted to receive an offer (probability ≥ 0.50)."
    ), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_html(
        "Pred. Denied", f"{pred_denied:,}", "accent-red",
        "Applications predicted to NOT receive an offer (probability < 0.50)."
    ), unsafe_allow_html=True)
with k4:
    st.markdown(kpi_html(
        "Avg Pred. Prob", f"{avg_prob:.2%}", "",
        "Average predicted offer probability across all scored applications."
    ), unsafe_allow_html=True)
with k5:
    st.markdown(kpi_html(
        "Precision (CP24)", f"{val_metrics['Precision']:.2%}", "accent-amber",
        "Of predicted offers, the share that actually received offers. Higher = fewer false alarms."
    ), unsafe_allow_html=True)
with k6:
    st.markdown(kpi_html(
        "Recall (CP24)", f"{val_metrics['Recall']:.2%}", "accent-amber",
        "Of actual offers, the share the model correctly identified. Higher = fewer missed offers."
    ), unsafe_allow_html=True)
with k7:
    st.markdown(kpi_html(
        "ROC-AUC (CP24)", f"{val_metrics['ROC_AUC']:.3f}", "",
        "Model's ability to distinguish offers from denials. 1.0 = perfect, 0.5 = random."
    ), unsafe_allow_html=True)

st.markdown("")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 9 — Tabs
# ═══════════════════════════════════════════════════════════════════════
tab_home, tab_exec, tab_coach, tab_detail, tab_fair, tab_model = st.tabs([
    "🏠 Welcome",
    "📊 Executive Overview",
    "🎯 Coach Action Center",
    "🔍 Application Detail",
    "⚖️ Subgroup Fairness",
    "🔧 Model Insights",
])


# ─────────────────────────────────────────────────────────────────────
# TAB 0 — Welcome / Landing Page
# ─────────────────────────────────────────────────────────────────────
with tab_home:

    # ── Hero ──────────────────────────────────────────────────
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #1B2A4A 0%, #2C5F8A 50%, #3B82B0 100%);
            color: white; padding: 2.5rem 2.5rem 2rem; border-radius: 16px;
            margin-bottom: 1.5rem;
        ">
            <h1 style="font-size:1.8rem;font-weight:800;margin:0 0 .4rem;
                        letter-spacing:-0.5px;">
                Welcome to the MLT Career Prep Dashboard
            </h1>
            <p style="font-size:.92rem;opacity:.88;margin:0 0 1rem;max-width:780px;
                       line-height:1.6;">
                This tool uses a machine-learning model trained on six years of
                Career Prep application data to predict whether a fellow's
                application is likely to receive an offer. It is designed to help
                coaches prioritize support and give leaders a data-informed view
                of cohort outcomes.
            </p>
            <div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:.6rem;">
                <div style="background:rgba(255,255,255,.15);border-radius:10px;
                            padding:.6rem 1rem;font-size:.78rem;">
                    <strong>Model:</strong> LASSO Logistic Regression (L1)
                </div>
                <div style="background:rgba(255,255,255,.15);border-radius:10px;
                            padding:.6rem 1rem;font-size:.78rem;">
                    <strong>Trained:</strong> CP 2018 &ndash; 2023
                </div>
                <div style="background:rgba(255,255,255,.15);border-radius:10px;
                            padding:.6rem 1rem;font-size:.78rem;">
                    <strong>Validated:</strong> CP 2024
                </div>
                <div style="background:rgba(255,255,255,.15);border-radius:10px;
                            padding:.6rem 1rem;font-size:.78rem;">
                    <strong>Scoring:</strong> CP 2025 (current fellows)
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Likelihood legend (always visible on landing) ─────────
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Understanding Likelihood Flags</div>
            <div class="section-caption">
                Every scored application receives a colour-coded flag based on
                its predicted offer probability.
            </div>
            <div style="display:flex;gap:1.2rem;flex-wrap:wrap;">
                <div style="flex:1;min-width:200px;border-left:5px solid #DC2626;
                            background:#FEF2F2;border-radius:10px;padding:1rem 1.2rem;">
                    <strong style="color:#DC2626;">Red &mdash; High Support Needed</strong>
                    <div style="font-size:.82rem;color:#374151;margin-top:.3rem;">
                        Predicted probability <strong>&lt; 0.35</strong>.
                        Historical patterns suggest long odds. Immediate,
                        targeted coaching is recommended.
                    </div>
                </div>
                <div style="flex:1;min-width:200px;border-left:5px solid #F59E0B;
                            background:#FFFBEB;border-radius:10px;padding:1rem 1.2rem;">
                    <strong style="color:#D97706;">Yellow &mdash; Moderate Support</strong>
                    <div style="font-size:.82rem;color:#374151;margin-top:.3rem;">
                        Predicted probability <strong>0.35 &ndash; 0.60</strong>.
                        Mixed signals &mdash; focused refinement of applications
                        and interview prep can shift outcomes.
                    </div>
                </div>
                <div style="flex:1;min-width:200px;border-left:5px solid #059669;
                            background:#ECFDF5;border-radius:10px;padding:1rem 1.2rem;">
                    <strong style="color:#059669;">Green &mdash; Likely Competitive</strong>
                    <div style="font-size:.82rem;color:#374151;margin-top:.3rem;">
                        Predicted probability <strong>&gt; 0.60</strong>.
                        Strong historical likelihood of an offer. Focus on closing
                        strategy and negotiation.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Role-based guides ─────────────────────────────────────
    st.markdown(
        '<div class="section-title" style="font-size:1.15rem;margin-top:.6rem;">'
        "How to Use This Dashboard"
        "</div>",
        unsafe_allow_html=True,
    )

    role_exec, role_coach = st.columns(2)

    # ── EXECUTIVE GUIDE ───────────────────────────────────────
    with role_exec:
        st.markdown(
            """
            <div class="section-card" style="border-top:4px solid #2C5F8A;min-height:520px;">
                <div style="font-size:.68rem;font-weight:700;text-transform:uppercase;
                            letter-spacing:.8px;color:#2C5F8A;margin-bottom:.4rem;">
                    Role Guide
                </div>
                <div class="section-title" style="font-size:1.1rem;">
                    For Executives &amp; Program Leaders
                </div>
                <div style="font-size:.84rem;color:#374151;line-height:1.7;">
                    <p style="margin-bottom:.6rem;">
                        You need a fast, high-level read on cohort health and
                        programme effectiveness. Start here:
                    </p>
                    <ol style="padding-left:1.2rem;">
                        <li style="margin-bottom:.55rem;">
                            <strong>KPI Strip</strong> (always visible at the top)
                            &mdash; Scan total applications scored, predicted
                            offered/denied split, and model accuracy (Precision,
                            Recall, ROC-AUC). Hover any
                            <span class="info-icon" style="display:inline;font-size:10px;">i</span>
                            icon for a plain-language definition.
                        </li>
                        <li style="margin-bottom:.55rem;">
                            <strong>Executive Overview tab</strong> &mdash; Review
                            the probability distribution, support-band counts, and
                            offer likelihood by programme track and company.
                            The <em>Highest-Support Applications</em> table at the
                            bottom flags the 20 most at-risk fellows for leadership
                            awareness.
                        </li>
                        <li style="margin-bottom:.55rem;">
                            <strong>Subgroup Fairness tab</strong> &mdash; Check
                            whether any demographic group is disproportionately
                            flagged. Use the category selector to cycle through
                            Track, Gender, Race, Ethnicity, First Generation, and
                            Pell Grant dimensions. Disparity warnings appear
                            automatically when a subgroup metric differs by more
                            than 0.10 from the population.
                        </li>
                        <li style="margin-bottom:.55rem;">
                            <strong>Sidebar &rarr; Downloads</strong> &mdash;
                            Export predictions, model metrics, and coefficient
                            CSVs for board decks or offline analysis.
                        </li>
                    </ol>
                    <p style="font-size:.78rem;color:#6B7280;margin-top:.5rem;">
                        <em>Tip: Use the cohort switcher in the sidebar to toggle
                        between CP 2025 (current fellows) and CP 2024 (validation
                        backtest).</em>
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── COACH GUIDE ───────────────────────────────────────────
    with role_coach:
        st.markdown(
            """
            <div class="section-card" style="border-top:4px solid #059669;min-height:520px;">
                <div style="font-size:.68rem;font-weight:700;text-transform:uppercase;
                            letter-spacing:.8px;color:#059669;margin-bottom:.4rem;">
                    Role Guide
                </div>
                <div class="section-title" style="font-size:1.1rem;">
                    For Coaches
                </div>
                <div style="font-size:.84rem;color:#374151;line-height:1.7;">
                    <p style="margin-bottom:.6rem;">
                        Your primary workspace is the <strong>Coach Action
                        Center</strong>. Here is a recommended workflow:
                    </p>
                    <ol style="padding-left:1.2rem;">
                        <li style="margin-bottom:.55rem;">
                            <strong>Filter your caseload</strong> &mdash; Use
                            the Coach, Track, Company, Functional Interest,
                            Likelihood Flag, and Predicted Outcome dropdowns to
                            narrow the table to your fellows.
                        </li>
                        <li style="margin-bottom:.55rem;">
                            <strong>Sort by &ldquo;Lowest first&rdquo;</strong>
                            &mdash; Red-flagged fellows appear at the top so you
                            can prioritise immediate interventions.
                        </li>
                        <li style="margin-bottom:.55rem;">
                            <strong>Company Likelihood Explorer</strong> &mdash;
                            Select a fellow and compare how their predicted
                            probability changes across different companies. Use
                            this to help fellows target higher-probability matches.
                        </li>
                        <li style="margin-bottom:.55rem;">
                            <strong>Application Detail tab</strong> &mdash;
                            Select any row to see the probability gauge, top
                            contributing factors, and a plain-English
                            interpretation of what is helping or hurting the
                            prediction. Record session notes at the bottom.
                        </li>
                        <li style="margin-bottom:.55rem;">
                            <strong>Export</strong> &mdash; Download your filtered
                            table as a CSV to share with your team or attach to
                            coaching logs.
                        </li>
                    </ol>
                    <p style="font-size:.78rem;color:#6B7280;margin-top:.5rem;">
                        <em>Tip: The &ldquo;Suggested Coach Action&rdquo; column
                        provides a starting point for each flag colour &mdash;
                        adapt it to the individual fellow.</em>
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Feature-by-feature guides ─────────────────────────────
    st.markdown(
        '<div class="section-title" style="font-size:1.15rem;margin-top:.4rem;">'
        "Feature Guides"
        "</div>",
        unsafe_allow_html=True,
    )

    fg1, fg2, fg3 = st.columns(3)

    # Application Detail guide
    with fg1:
        st.markdown(
            """
            <div class="section-card" style="min-height:440px;">
                <div style="font-size:1.3rem;margin-bottom:.4rem;">🔍</div>
                <div class="section-title">Application Detail</div>
                <div style="font-size:.82rem;color:#374151;line-height:1.65;">
                    <p><strong>What it does:</strong> Deep-dive into a single
                    application&rsquo;s prediction.</p>
                    <p><strong>How to use it:</strong></p>
                    <ul style="padding-left:1.1rem;">
                        <li><strong>Select</strong> an application from the
                            dropdown (sorted by probability, lowest first).</li>
                        <li><strong>Profile card</strong> &mdash; At a glance:
                            enrollment ID, coach, track, company, title,
                            functional interest, probability, flag, and
                            role-alignment heuristic.</li>
                        <li><strong>Probability gauge</strong> &mdash; A Plotly
                            gauge coloured by band with the 0.50 decision
                            threshold marked.</li>
                        <li><strong>Top contributing factors</strong> &mdash;
                            A horizontal bar chart showing which features push
                            the probability up (green) or down (red), with a
                            plain-English paragraph explaining the key drivers.</li>
                        <li><strong>Coach Notes</strong> &mdash; A free-text area
                            for session notes. Notes persist only while the
                            browser tab is open.</li>
                    </ul>
                    <p style="font-size:.78rem;color:#6B7280;margin-top:.4rem;">
                        <em>Best for: one-on-one coaching sessions where you need
                        to explain &ldquo;why&rdquo; the model scores a fellow a
                        certain way.</em>
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Subgroup Fairness guide
    with fg2:
        st.markdown(
            """
            <div class="section-card" style="min-height:440px;">
                <div style="font-size:1.3rem;margin-bottom:.4rem;">⚖️</div>
                <div class="section-title">Subgroup Fairness</div>
                <div style="font-size:.82rem;color:#374151;line-height:1.65;">
                    <p><strong>What it does:</strong> Monitors whether the model
                    treats demographic groups equitably.</p>
                    <p><strong>How to use it:</strong></p>
                    <ul style="padding-left:1.1rem;">
                        <li><strong>Category selector</strong> &mdash; Choose
                            from Programme Track, First Generation, Low Income,
                            Gender, Race, Ethnicity, or Pell Grant.</li>
                        <li><strong>Metrics table</strong> &mdash; Shows Count,
                            Actual Offer Rate, Avg Predicted Prob, Precision,
                            Recall, FPR, and FNR for each subgroup.</li>
                        <li><strong>Recall / FNR chart</strong> &mdash; Side-by-side
                            bars make it easy to spot which groups have more
                            missed offers (high FNR).</li>
                        <li><strong>Avg Predicted Prob chart</strong> &mdash;
                            Highlights groups that consistently receive lower
                            model scores.</li>
                        <li><strong>Disparity flags</strong> &mdash; Automatic
                            warnings appear when a subgroup&rsquo;s Recall or
                            FNR differs from the overall by more than 0.10.</li>
                        <li><strong>Export</strong> &mdash; Download the full
                            fairness table as a CSV.</li>
                    </ul>
                    <p style="font-size:.78rem;color:#6B7280;margin-top:.4rem;">
                        <em>Best for: team reviews, board reporting, and
                        proactive equity monitoring.</em>
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Model Insights guide
    with fg3:
        st.markdown(
            """
            <div class="section-card" style="min-height:440px;">
                <div style="font-size:1.3rem;margin-bottom:.4rem;">🔧</div>
                <div class="section-title">Model Insights</div>
                <div style="font-size:.82rem;color:#374151;line-height:1.65;">
                    <p><strong>What it does:</strong> Exposes the full model
                    configuration and performance so you can assess its
                    strengths and limitations.</p>
                    <p><strong>How to use it:</strong></p>
                    <ul style="padding-left:1.1rem;">
                        <li><strong>Model Configuration</strong> &mdash;
                            Algorithm, class weights, cross-validation strategy,
                            best regularization (C), threshold, intercept, train
                            cohorts, and feature counts. Every number has an
                            <span class="info-icon" style="display:inline;
                            font-size:10px;">i</span> tooltip.</li>
                        <li><strong>Positive / Negative coefficient lists</strong>
                            &mdash; See at a glance which features the model
                            says help versus hurt offer odds.</li>
                        <li><strong>Coefficient bar chart</strong> &mdash; Sorted
                            by absolute magnitude &mdash; the widest bars are
                            the most influential features.</li>
                        <li><strong>Coefficient table</strong> &mdash; Searchable
                            table with readable names and raw internal names.</li>
                        <li><strong>Performance metrics</strong> &mdash; Precision,
                            Recall, ROC-AUC, F1, and Accuracy for both the
                            CP 2024 validation set and (where ground truth is
                            available) the CP 2025 scoring set.</li>
                    </ul>
                    <p style="font-size:.78rem;color:#6B7280;margin-top:.4rem;">
                        <em>Best for: data-science review, audit documentation,
                        and stakeholder transparency.</em>
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Quick-reference workflow table ─────────────────────────
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Quick-Reference Workflow</div>
            <div style="font-size:.84rem;color:#374151;line-height:1.75;">
                <table style="width:100%;border-collapse:collapse;">
                    <thead>
                        <tr style="border-bottom:2px solid #E5E7EB;">
                            <th style="text-align:center;padding:.5rem .6rem;
                                       font-size:.72rem;color:#6B7280;
                                       text-transform:uppercase;letter-spacing:.5px;">
                                Step</th>
                            <th style="text-align:left;padding:.5rem .6rem;
                                       font-size:.72rem;color:#6B7280;
                                       text-transform:uppercase;letter-spacing:.5px;">
                                Action</th>
                            <th style="text-align:left;padding:.5rem .6rem;
                                       font-size:.72rem;color:#6B7280;
                                       text-transform:uppercase;letter-spacing:.5px;">
                                Tab</th>
                            <th style="text-align:left;padding:.5rem .6rem;
                                       font-size:.72rem;color:#6B7280;
                                       text-transform:uppercase;letter-spacing:.5px;">
                                Audience</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom:1px solid #F0F0F3;">
                            <td style="text-align:center;padding:.5rem .6rem;
                                       font-weight:700;color:#2C5F8A;">1</td>
                            <td style="padding:.5rem .6rem;">
                                Scan KPI strip for cohort health</td>
                            <td style="padding:.5rem .6rem;">
                                <em>Always visible</em></td>
                            <td style="padding:.5rem .6rem;">Everyone</td>
                        </tr>
                        <tr style="border-bottom:1px solid #F0F0F3;">
                            <td style="text-align:center;padding:.5rem .6rem;
                                       font-weight:700;color:#2C5F8A;">2</td>
                            <td style="padding:.5rem .6rem;">
                                Review distributions and top companies</td>
                            <td style="padding:.5rem .6rem;">
                                Executive Overview</td>
                            <td style="padding:.5rem .6rem;">Executives</td>
                        </tr>
                        <tr style="border-bottom:1px solid #F0F0F3;">
                            <td style="text-align:center;padding:.5rem .6rem;
                                       font-weight:700;color:#2C5F8A;">3</td>
                            <td style="padding:.5rem .6rem;">
                                Filter &amp; sort your fellows; explore company
                                fit</td>
                            <td style="padding:.5rem .6rem;">
                                Coach Action Center</td>
                            <td style="padding:.5rem .6rem;">Coaches</td>
                        </tr>
                        <tr style="border-bottom:1px solid #F0F0F3;">
                            <td style="text-align:center;padding:.5rem .6rem;
                                       font-weight:700;color:#2C5F8A;">4</td>
                            <td style="padding:.5rem .6rem;">
                                Drill into one application&rsquo;s drivers
                                &amp; add notes</td>
                            <td style="padding:.5rem .6rem;">
                                Application Detail</td>
                            <td style="padding:.5rem .6rem;">Coaches</td>
                        </tr>
                        <tr style="border-bottom:1px solid #F0F0F3;">
                            <td style="text-align:center;padding:.5rem .6rem;
                                       font-weight:700;color:#2C5F8A;">5</td>
                            <td style="padding:.5rem .6rem;">
                                Check subgroup equity &amp; flag disparities</td>
                            <td style="padding:.5rem .6rem;">
                                Subgroup Fairness</td>
                            <td style="padding:.5rem .6rem;">Everyone</td>
                        </tr>
                        <tr style="border-bottom:1px solid #F0F0F3;">
                            <td style="text-align:center;padding:.5rem .6rem;
                                       font-weight:700;color:#2C5F8A;">6</td>
                            <td style="padding:.5rem .6rem;">
                                Inspect model coefficients &amp; accuracy</td>
                            <td style="padding:.5rem .6rem;">
                                Model Insights</td>
                            <td style="padding:.5rem .6rem;">
                                Data&nbsp;team / Auditors</td>
                        </tr>
                        <tr>
                            <td style="text-align:center;padding:.5rem .6rem;
                                       font-weight:700;color:#2C5F8A;">7</td>
                            <td style="padding:.5rem .6rem;">
                                Export CSVs for records or presentations</td>
                            <td style="padding:.5rem .6rem;">
                                Sidebar &rarr; Downloads</td>
                            <td style="padding:.5rem .6rem;">Everyone</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Important caveats ─────────────────────────────────────
    st.markdown(
        """
        <div class="section-card" style="border-left:4px solid #F59E0B;">
            <div class="section-title" style="color:#D97706;">
                Important: What the Model Does Not Know
            </div>
            <div style="font-size:.84rem;color:#374151;line-height:1.7;">
                <p>This model identifies <em>statistical tendencies</em> from
                prior cohorts. It <strong>cannot</strong> observe:</p>
                <div style="display:flex;flex-wrap:wrap;gap:.5rem .8rem;
                            margin:.5rem 0 .8rem;">
                    <span style="background:#F3F4F6;border-radius:8px;
                                 padding:.3rem .7rem;font-size:.78rem;">
                        Interview performance</span>
                    <span style="background:#F3F4F6;border-radius:8px;
                                 padding:.3rem .7rem;font-size:.78rem;">
                        Networking &amp; referrals</span>
                    <span style="background:#F3F4F6;border-radius:8px;
                                 padding:.3rem .7rem;font-size:.78rem;">
                        Personal narrative</span>
                    <span style="background:#F3F4F6;border-radius:8px;
                                 padding:.3rem .7rem;font-size:.78rem;">
                        Market timing</span>
                    <span style="background:#F3F4F6;border-radius:8px;
                                 padding:.3rem .7rem;font-size:.78rem;">
                        Culture fit</span>
                    <span style="background:#F3F4F6;border-radius:8px;
                                 padding:.3rem .7rem;font-size:.78rem;">
                        Recent skill development</span>
                    <span style="background:#F3F4F6;border-radius:8px;
                                 padding:.3rem .7rem;font-size:.78rem;">
                        Family or health circumstances</span>
                </div>
                <p>A <span style="color:#DC2626;font-weight:600;">Red</span>
                flag does <strong>not</strong> mean a fellow cannot succeed
                &mdash; it means the coach&rsquo;s human judgement is especially
                valuable. Never share raw scores or flag colours directly with
                fellows. Use predictions to inform <em>your</em> coaching
                strategy, not to label individuals.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────
# TAB 1 — Executive Overview
# ─────────────────────────────────────────────────────────────────────
with tab_exec:
    st.markdown(legend_html(), unsafe_allow_html=True)

    # Probability distribution
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Overall Predicted Probability Distribution '
        + info_icon("Histogram of predicted offer probabilities for all scored applications.")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">Each bar shows the count of applications '
        "within a probability range.</div>",
        unsafe_allow_html=True,
    )
    fig_hist = go.Figure()
    bins_edges = np.arange(0, 1.05, 0.05)
    counts, _ = np.histogram(active_df["Predicted_Probability"], bins=bins_edges)
    mids = (bins_edges[:-1] + bins_edges[1:]) / 2
    colors = [LIKELIHOOD_COLORS[assign_likelihood(m)] for m in mids]
    fig_hist.add_trace(go.Bar(x=mids, y=counts, marker_color=colors, width=0.045))
    fig_hist.update_layout(xaxis_title="Predicted Probability", yaxis_title="Count")
    plotly_clean(fig_hist, 350)
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    ec1, ec2 = st.columns(2)

    # Support band counts
    with ec1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Support Band Counts '
            + info_icon("Number of applications in each likelihood band.")
            + "</div>",
            unsafe_allow_html=True,
        )
        band_counts = active_df["Risk_Flag"].value_counts().reindex(
            ["Red", "Yellow", "Green"], fill_value=0
        )
        fig_band = go.Figure(go.Bar(
            x=band_counts.index,
            y=band_counts.values,
            marker_color=[LIKELIHOOD_COLORS[c] for c in band_counts.index],
            text=band_counts.values, textposition="auto",
        ))
        fig_band.update_layout(xaxis_title="Likelihood Band", yaxis_title="Count")
        plotly_clean(fig_band, 320)
        st.plotly_chart(fig_band, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Offer likelihood by track
    with ec2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Offer Likelihood by Program Track '
            + info_icon("Average predicted probability grouped by program track.")
            + "</div>",
            unsafe_allow_html=True,
        )
        track_col = "Program Enrollment: Program Track"
        if track_col in active_df.columns:
            track_avg = (
                active_df.groupby(track_col)["Predicted_Probability"]
                .mean().sort_values(ascending=True)
            )
            fig_track = go.Figure(go.Bar(
                x=track_avg.values, y=track_avg.index, orientation="h",
                marker_color=[LIKELIHOOD_COLORS[assign_likelihood(v)] for v in track_avg.values],
                text=[f"{v:.1%}" for v in track_avg.values], textposition="auto",
            ))
            fig_track.update_layout(xaxis_title="Avg Predicted Probability", yaxis_title="")
            plotly_clean(fig_track, 320)
            st.plotly_chart(fig_track, use_container_width=True)
        else:
            st.info("Program Track column not available.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Offer likelihood by company (top 15)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Offer Likelihood by Company (Top 15) '
        + info_icon("Average predicted probability for the 15 most-applied-to companies.")
        + "</div>",
        unsafe_allow_html=True,
    )
    if "Related Organization" in active_df.columns:
        co_counts = active_df["Related Organization"].dropna().value_counts()
        top_cos = co_counts.head(15).index
        co_avg = (
            active_df[active_df["Related Organization"].isin(top_cos)]
            .groupby("Related Organization")["Predicted_Probability"]
            .mean().sort_values(ascending=True)
        )
        fig_co = go.Figure(go.Bar(
            x=co_avg.values, y=co_avg.index, orientation="h",
            marker_color=[LIKELIHOOD_COLORS[assign_likelihood(v)] for v in co_avg.values],
            text=[f"{v:.1%}" for v in co_avg.values], textposition="auto",
        ))
        fig_co.update_layout(xaxis_title="Avg Predicted Probability", yaxis_title="")
        plotly_clean(fig_co, max(350, len(co_avg) * 28 + 80))
        st.plotly_chart(fig_co, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Highest-risk applications
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Highest-Support Applications '
        + info_icon("The 20 lowest-probability applications requiring immediate coach intervention.")
        + "</div>",
        unsafe_allow_html=True,
    )
    risk_cols = [
        "Program Enrollment: Enrollment ID", "Program Enrollment: Coach",
        "Related Organization", "Title", "Predicted_Probability",
        "Risk_Flag", "Suggested_Coach_Action",
    ]
    risk_cols = [c for c in risk_cols if c in active_df.columns]
    risk_table = active_df.nsmallest(20, "Predicted_Probability")[risk_cols].reset_index(drop=True)
    st.dataframe(
        risk_table, use_container_width=True, hide_index=True, height=500,
        column_config={
            "Predicted_Probability": st.column_config.ProgressColumn(
                "Pred Prob", format="%.3f", min_value=0, max_value=1,
            ),
            "Risk_Flag": st.column_config.TextColumn("Likelihood"),
        },
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# TAB 2 — Coach Action Center
# ─────────────────────────────────────────────────────────────────────
with tab_coach:
    st.markdown(legend_html(), unsafe_allow_html=True)

    # Filters
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Filters</div>', unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        coach_opts = ["All"] + sorted(
            active_df["Program Enrollment: Coach"].dropna().unique().tolist()
        )
        sel_coach = st.selectbox("Coach", coach_opts, key="coach_filter")
    with fc2:
        track_opts = ["All"] + sorted(
            active_df["Program Enrollment: Program Track"].dropna().unique().tolist()
        )
        sel_track = st.selectbox("Program Track", track_opts, key="coach_track_filter")
    with fc3:
        org_opts = ["All"] + sorted(
            active_df["Related Organization"].dropna().value_counts().head(50).index.tolist()
        )
        sel_org = st.selectbox("Company", org_opts, key="coach_org_filter")

    fc4, fc5, fc6 = st.columns(3)
    with fc4:
        func_opts = ["All"] + sorted(
            active_df["Primary Functional Interest"].dropna().unique().tolist()
        )
        sel_func = st.selectbox("Functional Interest", func_opts, key="coach_func_filter")
    with fc5:
        sel_flag = st.selectbox("Likelihood Flag", ["All", "Red", "Yellow", "Green"], key="coach_flag_filter")
    with fc6:
        sel_outcome = st.selectbox("Predicted Outcome", ["All", "Offered", "Denied"], key="coach_outcome_filter")

    st.markdown("</div>", unsafe_allow_html=True)

    # Apply filters
    filtered = active_df.copy()
    if sel_coach != "All":
        filtered = filtered[filtered["Program Enrollment: Coach"] == sel_coach]
    if sel_track != "All":
        filtered = filtered[filtered["Program Enrollment: Program Track"] == sel_track]
    if sel_org != "All":
        filtered = filtered[filtered["Related Organization"] == sel_org]
    if sel_func != "All":
        filtered = filtered[filtered["Primary Functional Interest"] == sel_func]
    if sel_flag != "All":
        filtered = filtered[filtered["Risk_Flag"] == sel_flag]
    if sel_outcome != "All":
        filtered = filtered[filtered["Predicted_Outcome"] == sel_outcome]

    # Sort
    sort_order = st.radio(
        "Sort by probability:", ["Lowest first (highest support need)", "Highest first"],
        horizontal=True, key="coach_sort",
    )
    ascending = sort_order.startswith("Lowest")
    filtered = filtered.sort_values("Predicted_Probability", ascending=ascending).reset_index(drop=True)

    # KPI strip for filtered view
    fk1, fk2, fk3, fk4, fk5 = st.columns(5)
    with fk1:
        st.markdown(kpi_html("Filtered", f"{len(filtered):,}", "",
                             "Applications matching current filters."), unsafe_allow_html=True)
    with fk2:
        n_red = (filtered["Risk_Flag"] == "Red").sum()
        st.markdown(kpi_html("Red", f"{n_red:,}", "accent-red",
                             "High-support applications in filtered set."), unsafe_allow_html=True)
    with fk3:
        n_yel = (filtered["Risk_Flag"] == "Yellow").sum()
        st.markdown(kpi_html("Yellow", f"{n_yel:,}", "accent-amber",
                             "Moderate-support applications."), unsafe_allow_html=True)
    with fk4:
        n_grn = (filtered["Risk_Flag"] == "Green").sum()
        st.markdown(kpi_html("Green", f"{n_grn:,}", "accent-green",
                             "Likely competitive applications."), unsafe_allow_html=True)
    with fk5:
        f_avg = filtered["Predicted_Probability"].mean() if len(filtered) else 0
        st.markdown(kpi_html("Avg Prob", f"{f_avg:.2%}", "",
                             "Average predicted probability for filtered applications."), unsafe_allow_html=True)

    # Main table
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Application Table</div>', unsafe_allow_html=True)
    table_cols = [
        "Program Enrollment: Enrollment ID", "Program Enrollment: Coach",
        "Program Enrollment: Program Track", "Related Organization", "Title",
        "Primary Functional Interest", "Predicted_Probability", "Risk_Flag",
        "Predicted_Outcome", "Actual_Outcome", "Correct",
        "Suggested_Coach_Action", "Coach_Notes",
    ]
    table_cols = [c for c in table_cols if c in filtered.columns]
    st.dataframe(
        filtered[table_cols], use_container_width=True, hide_index=True, height=500,
        column_config={
            "Predicted_Probability": st.column_config.ProgressColumn(
                "Pred Prob", format="%.3f", min_value=0, max_value=1,
            ),
            "Risk_Flag": st.column_config.TextColumn("Likelihood"),
            "Coach_Notes": st.column_config.TextColumn("Coach Notes", width="large"),
        },
    )

    # Export
    if len(filtered) > 0:
        export_csv = filtered[table_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Export Filtered Results (.csv)", export_csv,
            "coach_action_filtered.csv", "text/csv", key="coach_export",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Company Likelihood Explorer ───────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Company Likelihood Explorer '
        + info_icon(
            "See how a fellow's predicted probability would change "
            "across different companies. Swaps the company features "
            "while holding the fellow's profile constant."
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">Select a fellow, then choose companies '
        "to compare their predicted offer likelihood.</div>",
        unsafe_allow_html=True,
    )

    if len(filtered) > 0:
        eid_col = "Program Enrollment: Enrollment ID"
        fellow_ids = filtered[eid_col].dropna().unique().tolist()[:200]
        if fellow_ids:
            sel_fellow = st.selectbox(
                "Select Fellow (Enrollment ID)", fellow_ids, key="cle_fellow",
            )
            fellow_row = filtered[filtered[eid_col] == sel_fellow].iloc[0:1]

            all_companies = sorted(
                active_df["Related Organization"].dropna().unique().tolist()
            )
            sel_companies = st.multiselect(
                "Select Companies to Compare",
                all_companies,
                default=all_companies[:10],
                key="cle_companies",
            )

            if sel_companies and len(fellow_row) > 0:
                rows_sim = []
                for company in sel_companies:
                    sim_row = fellow_row.copy()
                    sim_row["Related Organization"] = company
                    sim_row["Partner Org?"] = "Non-Partner"
                    rows_sim.append(sim_row)
                sim_df = pd.concat(rows_sim, ignore_index=True)
                X_sim = build_features(sim_df, config)
                X_sim_sc = scaler.transform(X_sim)
                logits_sim = intercept + X_sim_sc @ coefs_full
                probs_sim = 1.0 / (1.0 + np.exp(-logits_sim))

                cle_results = pd.DataFrame({
                    "Company": sel_companies,
                    "Predicted_Probability": np.round(probs_sim, 4),
                    "Likelihood": [assign_likelihood(p) for p in probs_sim],
                }).sort_values("Predicted_Probability", ascending=False).reset_index(drop=True)
                cle_results["Rank"] = range(1, len(cle_results) + 1)

                fig_cle = go.Figure(go.Bar(
                    x=cle_results["Predicted_Probability"],
                    y=cle_results["Company"],
                    orientation="h",
                    marker_color=[
                        LIKELIHOOD_COLORS[f] for f in cle_results["Likelihood"]
                    ],
                    text=[f"{p:.1%}" for p in cle_results["Predicted_Probability"]],
                    textposition="auto",
                ))
                fig_cle.update_layout(
                    xaxis_title="Predicted Probability",
                    yaxis_title="",
                    xaxis=dict(range=[0, 1]),
                )
                plotly_clean(fig_cle, max(300, len(cle_results) * 30 + 80))
                st.plotly_chart(fig_cle, use_container_width=True)

                st.dataframe(
                    cle_results, use_container_width=True, hide_index=True,
                    column_config={
                        "Predicted_Probability": st.column_config.ProgressColumn(
                            "Pred Prob", format="%.3f", min_value=0, max_value=1,
                        ),
                    },
                )
        else:
            st.info("No fellows in the current filtered view.")
    else:
        st.info("No applications match the current filters.")
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# TAB 3 — Application Detail
# ─────────────────────────────────────────────────────────────────────
with tab_detail:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Select an Application</div>',
        unsafe_allow_html=True,
    )

    detail_df = active_df.sort_values("Predicted_Probability").reset_index(drop=True)
    label_series = (
        detail_df["Program Enrollment: Enrollment ID"].astype(str)
        + " | "
        + detail_df["Related Organization"].fillna("N/A").astype(str)
        + " | "
        + detail_df["Predicted_Probability"].apply(lambda x: f"{x:.1%}")
    )
    sel_idx = st.selectbox(
        "Application", range(len(detail_df)),
        format_func=lambda i: label_series.iloc[i], key="detail_select",
    )
    app = detail_df.iloc[sel_idx]
    st.markdown("</div>", unsafe_allow_html=True)

    # Profile card
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Application Profile</div>', unsafe_allow_html=True)

    dc1, dc2, dc3, dc4 = st.columns(4)
    with dc1:
        st.markdown(f"**Enrollment ID:** {app.get('Program Enrollment: Enrollment ID', 'N/A')}")
        st.markdown(f"**Coach:** {app.get('Program Enrollment: Coach', 'N/A')}")
    with dc2:
        st.markdown(f"**Track:** {app.get('Program Enrollment: Program Track', 'N/A')}")
        st.markdown(f"**Company:** {app.get('Related Organization', 'N/A')}")
    with dc3:
        st.markdown(f"**Title:** {app.get('Title', 'N/A')}")
        st.markdown(f"**Functional Interest:** {app.get('Primary Functional Interest', 'N/A')}")
    with dc4:
        flag = app.get("Risk_Flag", "N/A")
        color = LIKELIHOOD_COLORS.get(flag, "#6B7280")
        st.markdown(f"**Predicted Probability:** {app['Predicted_Probability']:.1%}")
        st.markdown(
            f'**Likelihood:** <span class="risk-badge {flag.lower()}">{flag} — '
            f'{LIKELIHOOD_LABELS.get(flag, "")}</span>',
            unsafe_allow_html=True,
        )
    st.markdown(f"**Predicted Outcome:** {app.get('Predicted_Outcome', 'N/A')}")
    st.markdown(f"**Actual Outcome:** {app.get('Actual_Outcome', 'N/A')}")
    st.markdown(f"**Role Alignment:** {app.get('Likely_Role_Alignment', 'N/A')}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Probability gauge
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Predicted Probability Gauge '
        + info_icon("Visual representation of the model's predicted offer probability.")
        + "</div>",
        unsafe_allow_html=True,
    )
    prob_val = app["Predicted_Probability"]
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_val,
        number={"suffix": "", "valueformat": ".1%"},
        gauge=dict(
            axis=dict(range=[0, 1], tickformat=".0%"),
            bar=dict(color=color),
            steps=[
                dict(range=[0, 0.35], color="#FEE2E2"),
                dict(range=[0.35, 0.60], color="#FEF3C7"),
                dict(range=[0.60, 1.0], color="#D1FAE5"),
            ],
            threshold=dict(line=dict(color="#1B2A4A", width=3), thickness=0.8, value=THRESHOLD),
        ),
    ))
    plotly_clean(fig_gauge, 280)
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Top contributing factors
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Top Contributing Factors '
        + info_icon(
            "Approximate feature contributions based on scaled feature values "
            "multiplied by model coefficients. Positive = increases offer likelihood."
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    app_row_df = detail_df.iloc[sel_idx:sel_idx + 1]
    X_app = build_features(app_row_df, config)
    X_app_sc = scaler.transform(X_app.values)
    contributions = X_app_sc[0] * coefs_full
    contrib_df = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": contributions,
    })
    contrib_df["Abs"] = contrib_df["Contribution"].abs()
    contrib_df = contrib_df[contrib_df["Abs"] > 0.001].sort_values("Abs", ascending=False).head(12)
    contrib_df["Readable"] = contrib_df["Feature"].apply(readable_feature)
    contrib_df = contrib_df.sort_values("Contribution", ascending=True)

    if len(contrib_df) > 0:
        fig_contrib = go.Figure(go.Bar(
            x=contrib_df["Contribution"],
            y=contrib_df["Readable"],
            orientation="h",
            marker_color=[
                LIKELIHOOD_COLORS["Green"] if c > 0 else LIKELIHOOD_COLORS["Red"]
                for c in contrib_df["Contribution"]
            ],
        ))
        fig_contrib.update_layout(xaxis_title="Contribution to Prediction", yaxis_title="")
        plotly_clean(fig_contrib, max(300, len(contrib_df) * 28 + 60))
        st.plotly_chart(fig_contrib, use_container_width=True)

        # Plain-English interpretation
        pos_feats = contrib_df[contrib_df["Contribution"] > 0]["Readable"].tolist()
        neg_feats = contrib_df[contrib_df["Contribution"] < 0]["Readable"].tolist()

        interp_parts = []
        if pos_feats:
            interp_parts.append(
                f"**Factors increasing likelihood:** {', '.join(pos_feats[:4])}."
            )
        if neg_feats:
            interp_parts.append(
                f"**Factors decreasing likelihood:** {', '.join(neg_feats[:4])}."
            )
        if flag == "Red":
            interp_parts.append(
                "This application has a **low predicted offer probability**. "
                "Coaches should prioritize immediate intervention — review target fit, "
                "interview readiness, and application materials."
            )
        elif flag == "Yellow":
            interp_parts.append(
                "This application shows **mixed signals**. Targeted coaching on "
                "the key negative factors could meaningfully improve outcomes."
            )
        else:
            interp_parts.append(
                "This application appears **well-positioned** for an offer. "
                "Focus coaching on closing strategy and negotiation preparation."
            )
        st.markdown(" ".join(interp_parts))
    else:
        st.info("All feature contributions are near zero for this application.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Coach notes (session-only)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Coach Notes</div>', unsafe_allow_html=True)
    note_key = f"note_{app.get('Program Enrollment: Enrollment ID', sel_idx)}"
    st.text_area(
        "Enter coaching notes for this application",
        value=st.session_state.get(note_key, ""),
        height=150, key=note_key, label_visibility="collapsed",
    )
    st.caption("Notes persist for this browser session only.")
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# TAB 4 — Subgroup Fairness
# ─────────────────────────────────────────────────────────────────────
with tab_fair:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Subgroup Fairness Monitoring '
        + info_icon(
            "Fairness metrics broken down by demographic subgroups. "
            "These diagnostics help identify potential disparities in model performance."
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">These fairness checks are diagnostic and should '
        "be interpreted cautiously, especially for small subgroup sample sizes.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Only compute fairness on data with known outcomes
    fair_df = active_df[active_df["Offered"].notna()].copy()

    if len(fair_df) < 20:
        st.warning("Too few applications with known outcomes for fairness analysis.")
    else:
        fairness_groups = {
            "Program Track": "Program Enrollment: Program Track",
            "First Generation": "First Generation College",
            "Low Income": "Designated Low Income",
            "Gender": "Gender",
            "Race": "Race",
            "Ethnicity": "Ethnicity",
        }

        # Pell Grant binary
        if "Pell Grant Count" in fair_df.columns:
            fair_df["Pell Grant Recipient"] = np.where(
                fair_df["Pell Grant Count"].fillna(0) > 0,
                "Pell Grant Recipient", "No Pell Grant",
            )
            fairness_groups["Pell Grant"] = "Pell Grant Recipient"

        # Category selector
        avail_groups = {k: v for k, v in fairness_groups.items() if v in fair_df.columns}
        sel_group = st.selectbox(
            "Select Subgroup Category",
            list(avail_groups.keys()),
            key="fairness_group",
        )
        group_col = avail_groups[sel_group]

        fm = compute_fairness(fair_df, group_col)
        if fm is not None:
            # KPI strip
            fk1, fk2, fk3, fk4 = st.columns(4)
            with fk1:
                st.markdown(kpi_html("Subgroups", f"{len(fm)}", "",
                                     "Number of subgroups with ≥ 5 applications."),
                            unsafe_allow_html=True)
            with fk2:
                total_n = fm["Count"].sum()
                st.markdown(kpi_html("Total Apps", f"{total_n:,}", "",
                                     "Total applications across all subgroups."),
                            unsafe_allow_html=True)
            with fk3:
                recall_range = fm["Recall"].dropna()
                if len(recall_range) > 0:
                    r_spread = recall_range.max() - recall_range.min()
                    st.markdown(kpi_html("Recall Spread", f"{r_spread:.3f}",
                                         "accent-amber" if r_spread > 0.10 else "",
                                         "Difference between highest and lowest subgroup recall."),
                                unsafe_allow_html=True)
            with fk4:
                fnr_range = fm["FNR"].dropna()
                if len(fnr_range) > 0:
                    f_spread = fnr_range.max() - fnr_range.min()
                    st.markdown(kpi_html("FNR Spread", f"{f_spread:.3f}",
                                         "accent-red" if f_spread > 0.10 else "",
                                         "Difference between highest and lowest false negative rate."),
                                unsafe_allow_html=True)

            # Table
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-title">Fairness Metrics Table</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(fm, use_container_width=True, hide_index=True, height=400)
            st.markdown("</div>", unsafe_allow_html=True)

            # Charts
            chart_c1, chart_c2 = st.columns(2)
            with chart_c1:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-title">Recall & FNR by Subgroup '
                    + info_icon("Recall = correctly identified offers. FNR = missed offers.")
                    + "</div>",
                    unsafe_allow_html=True,
                )
                fm_chart = fm.dropna(subset=["Recall", "FNR"])
                if len(fm_chart) > 0:
                    fig_fair1 = go.Figure()
                    fig_fair1.add_trace(go.Bar(
                        x=fm_chart["Subgroup"], y=fm_chart["Recall"],
                        name="Recall", marker_color="#059669",
                    ))
                    fig_fair1.add_trace(go.Bar(
                        x=fm_chart["Subgroup"], y=fm_chart["FNR"],
                        name="FNR", marker_color="#DC2626",
                    ))
                    fig_fair1.update_layout(barmode="group")
                    plotly_clean(fig_fair1, 350)
                    st.plotly_chart(fig_fair1, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with chart_c2:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-title">Avg Predicted Probability by Subgroup '
                    + info_icon("Average model-predicted probability for each subgroup.")
                    + "</div>",
                    unsafe_allow_html=True,
                )
                fig_fair2 = go.Figure(go.Bar(
                    x=fm["Subgroup"], y=fm["Avg Predicted Prob"],
                    marker_color=[
                        LIKELIHOOD_COLORS[assign_likelihood(v)]
                        for v in fm["Avg Predicted Prob"]
                    ],
                    text=[f"{v:.1%}" for v in fm["Avg Predicted Prob"]],
                    textposition="auto",
                ))
                fig_fair2.update_layout(yaxis_title="Avg Predicted Prob")
                plotly_clean(fig_fair2, 350)
                st.plotly_chart(fig_fair2, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Disparity warnings
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-title">Disparity Flags '
                + info_icon(
                    "Subgroups where recall or FNR differs by more than 0.10 "
                    "from the overall population are flagged."
                )
                + "</div>",
                unsafe_allow_html=True,
            )
            y_act = fair_df["Actual_Label"].astype(int).values
            y_prd = fair_df["Predicted_Label"].astype(int).values
            overall_recall = recall_score(y_act, y_prd, zero_division=0)
            tn_o, fp_o, fn_o, tp_o = confusion_matrix(y_act, y_prd, labels=[0, 1]).ravel()
            overall_fnr = fn_o / (fn_o + tp_o) if (fn_o + tp_o) > 0 else 0

            flags_found = False
            for _, row in fm.dropna(subset=["Recall", "FNR"]).iterrows():
                if abs(row["Recall"] - overall_recall) > 0.10:
                    st.warning(
                        f"⚠️ **{row['Subgroup']}** — Recall ({row['Recall']:.3f}) "
                        f"differs from overall ({overall_recall:.3f}) by more than 0.10."
                    )
                    flags_found = True
                if abs(row["FNR"] - overall_fnr) > 0.10:
                    st.warning(
                        f"⚠️ **{row['Subgroup']}** — FNR ({row['FNR']:.3f}) "
                        f"differs from overall ({overall_fnr:.3f}) by more than 0.10."
                    )
                    flags_found = True
            if not flags_found:
                st.success("No subgroup disparities exceed the 0.10 threshold.")
            st.markdown("</div>", unsafe_allow_html=True)

            # Export
            fair_csv = fm.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Export Fairness Summary (.csv)", fair_csv,
                "subgroup_fairness_summary.csv", "text/csv", key="fair_export",
            )
        else:
            st.info(f"Not enough data to compute fairness for '{sel_group}'.")


# ─────────────────────────────────────────────────────────────────────
# TAB 5 — Model Insights
# ─────────────────────────────────────────────────────────────────────
with tab_model:
    # Model configuration
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Model Configuration '
        + info_icon("Technical details of the trained LASSO logistic regression model.")
        + "</div>",
        unsafe_allow_html=True,
    )

    mi1, mi2, mi3 = st.columns(3)
    with mi1:
        st.markdown(
            f'**Algorithm:** L1 (LASSO) Logistic Regression '
            f'{info_icon("LASSO uses L1 regularization to shrink weak coefficients to zero.")}',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'**Class Weights:** Balanced '
            f'{info_icon("Up-weights minority class to prevent majority-class bias.")}',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'**Cross-Validation:** 5-fold, scoring = ROC-AUC '
            f'{info_icon("Best C chosen by highest average ROC-AUC across 5 folds.")}',
            unsafe_allow_html=True,
        )
    with mi2:
        st.markdown(
            f'**Best Regularization (C):** {best_C:.5f} '
            f'{info_icon("Smaller C = stronger regularization = fewer features.")}',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'**Decision Threshold:** {THRESHOLD} '
            f'{info_icon("Probability ≥ this value → predicted Offered.")}',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'**Intercept:** {intercept:.4f} '
            f'{info_icon("Baseline log-odds before feature contributions.")}',
            unsafe_allow_html=True,
        )
    with mi3:
        st.markdown(f"**Train Cohorts:** {', '.join(TRAIN_COHORTS)}")
        st.markdown(f"**Validation Cohort:** {VALIDATION_COHORT}")
        st.markdown(
            f'**Total Features:** {len(feature_names)} '
            f'{info_icon("Engineered features fed into the model.")}',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'**Non-Zero (Selected):** {len(coef_df)} '
            f'{info_icon("Features the model actually uses (non-zero coefficients).")}',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Coefficient lists
    cl1, cl2 = st.columns(2)
    with cl1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Features Increasing Offer Likelihood</div>',
            unsafe_allow_html=True,
        )
        pos_c = coef_df[coef_df["Coefficient"] > 0].copy()
        pos_c["Readable"] = pos_c["Feature"].apply(readable_feature)
        if len(pos_c) > 0:
            for _, r in pos_c.iterrows():
                st.markdown(f"- **{r['Readable']}**: +{r['Coefficient']:.4f}")
        else:
            st.markdown("_No positive coefficients._")
        st.markdown("</div>", unsafe_allow_html=True)

    with cl2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Features Decreasing Offer Likelihood</div>',
            unsafe_allow_html=True,
        )
        neg_c = coef_df[coef_df["Coefficient"] < 0].copy()
        neg_c["Readable"] = neg_c["Feature"].apply(readable_feature)
        if len(neg_c) > 0:
            for _, r in neg_c.iterrows():
                st.markdown(f"- **{r['Readable']}**: {r['Coefficient']:.4f}")
        else:
            st.markdown("_No negative coefficients._")
        st.markdown("</div>", unsafe_allow_html=True)

    # Coefficient bar chart
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Non-Zero LASSO Coefficients '
        + info_icon("Sorted by absolute magnitude. Green = increases offer odds, Red = decreases.")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">Sorted by absolute magnitude.</div>',
        unsafe_allow_html=True,
    )

    if len(coef_df) > 0:
        chart_df = coef_df.copy()
        chart_df["Readable"] = chart_df["Feature"].apply(readable_feature)
        chart_df = chart_df.sort_values("Coefficient", ascending=True)
        fig_coef = go.Figure(go.Bar(
            x=chart_df["Coefficient"], y=chart_df["Readable"], orientation="h",
            marker_color=[
                LIKELIHOOD_COLORS["Green"] if c > 0 else LIKELIHOOD_COLORS["Red"]
                for c in chart_df["Coefficient"]
            ],
        ))
        fig_coef.update_layout(xaxis_title="Coefficient Value", yaxis_title="")
        plotly_clean(fig_coef, max(350, len(chart_df) * 28 + 80))
        st.plotly_chart(fig_coef, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Coefficient table
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Coefficient Table</div>',
        unsafe_allow_html=True,
    )
    coef_display = coef_df.copy()
    coef_display["Readable"] = coef_display["Feature"].apply(readable_feature)
    coef_display = coef_display[["Readable", "Feature", "Coefficient"]].rename(
        columns={"Readable": "Feature (Readable)", "Feature": "Internal Name"},
    )
    st.dataframe(
        coef_display, use_container_width=True, hide_index=True,
        height=min(400, 40 * len(coef_display) + 40),
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Performance metrics
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Validation Performance (CP 2024) '
        + info_icon("Model performance on the held-out CP 2024 cohort.")
        + "</div>",
        unsafe_allow_html=True,
    )
    vm1, vm2, vm3, vm4, vm5 = st.columns(5)
    with vm1:
        st.metric("Precision", f"{val_metrics['Precision']:.3f}")
    with vm2:
        st.metric("Recall", f"{val_metrics['Recall']:.3f}")
    with vm3:
        st.metric("ROC-AUC", f"{val_metrics['ROC_AUC']:.3f}")
    with vm4:
        st.metric("F1 Score", f"{val_metrics['F1']:.3f}")
    with vm5:
        st.metric("Accuracy", f"{val_metrics['Accuracy']:.3f}")

    if score_metrics:
        st.markdown("---")
        st.markdown(
            '<div class="section-title">Scoring Performance '
            "(CP 2025, known outcomes only)</div>",
            unsafe_allow_html=True,
        )
        sm1, sm2, sm3, sm4, sm5 = st.columns(5)
        with sm1:
            st.metric("Precision", f"{score_metrics.get('Precision', 0):.3f}")
        with sm2:
            st.metric("Recall", f"{score_metrics.get('Recall', 0):.3f}")
        with sm3:
            st.metric("ROC-AUC", f"{score_metrics.get('ROC_AUC', 0):.3f}")
        with sm4:
            st.metric("F1 Score", f"{score_metrics.get('F1', 0):.3f}")
        with sm5:
            st.metric("Accuracy", f"{score_metrics.get('Accuracy', 0):.3f}")
    st.markdown("</div>", unsafe_allow_html=True)
