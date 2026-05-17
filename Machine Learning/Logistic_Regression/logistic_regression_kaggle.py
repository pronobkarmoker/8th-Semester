# =============================================================================
# LOGISTIC REGRESSION — PRODUCTION-QUALITY IMPLEMENTATION
# Dataset: Titanic (Kaggle) — Binary Classification
# Author: ML Pipeline Template
# =============================================================================
# This script demonstrates a complete, modular, production-grade Logistic
# Regression workflow on a real-world Kaggle dataset (Titanic). Every step
# is documented and can be adapted to any binary-classification CSV dataset.
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_validate
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Plotting style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#aaa",
    "ytick.color":      "#aaa",
    "text.color":       "#e0e0e0",
    "grid.color":       "#333",
    "grid.linestyle":   "--",
    "font.family":      "monospace",
})
ACCENT  = "#00d4ff"
ACCENT2 = "#ff6b6b"
CMAP    = "Blues"

# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load a CSV dataset.  Adjust `filepath` to your local Kaggle download.

    For the Titanic dataset:
        kaggle competitions download -c titanic
        unzip titanic.zip          # gives train.csv, test.csv
        Use 'train.csv' here (it has the target column 'Survived').
    """
    print(f"\n{'='*60}")
    print("  STEP 1 — DATA LOADING")
    print(f"{'='*60}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Download it with:\n"
            "  kaggle competitions download -c titanic\n"
            "  unzip titanic.zip\n"
            "Then set filepath='train.csv'."
        )

    # dtype_backend='numpy_nullable' reduces memory vs object dtype
    df = pd.read_csv(filepath)
    print(f"  Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\n  Head:\n{df.head(3).to_string()}")
    print(f"\n  dtypes:\n{df.dtypes.to_string()}")
    return df


# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS (quick)
# =============================================================================

def quick_eda(df: pd.DataFrame, target: str) -> None:
    """Print basic EDA summary."""
    print(f"\n{'='*60}")
    print("  STEP 2 — QUICK EDA")
    print(f"{'='*60}")

    print(f"\n  Missing values (%):\n"
          f"{(df.isnull().mean()*100).round(2).to_string()}")

    print(f"\n  Target distribution:\n"
          f"{df[target].value_counts(normalize=True).mul(100).round(2).to_string()} %")

    imbalance_ratio = df[target].value_counts().iloc[0] / df[target].value_counts().iloc[1]
    if imbalance_ratio > 1.5:
        print(f"\n  ⚠  Class imbalance detected (ratio ≈ {imbalance_ratio:.2f}x)."
              " Will apply class_weight='balanced'.")
    else:
        print(f"\n  ✔  Classes reasonably balanced (ratio ≈ {imbalance_ratio:.2f}x).")


# =============================================================================
# 3. PREPROCESSING
# =============================================================================

def preprocess_titanic(df: pd.DataFrame, target: str):
    """
    Titanic-specific feature engineering + generic preprocessing.

    Returns
    -------
    X : pd.DataFrame  — feature matrix
    y : pd.Series     — target vector
    feature_names : list[str]
    """
    print(f"\n{'='*60}")
    print("  STEP 3 — PREPROCESSING")
    print(f"{'='*60}")

    df = df.copy()

    # ── 3a. Drop leaky / irrelevant columns ──────────────────────────────────
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # ── 3b. Feature engineering ───────────────────────────────────────────────
    if "SibSp" in df.columns and "Parch" in df.columns:
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

    # ── 3c. Impute missing values ─────────────────────────────────────────────
    # Numeric: median imputation (robust to outliers)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != target]
    for col in num_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Imputed {col} (numeric) with median = {median_val:.2f}")

    # Categorical: mode imputation
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  Imputed {col} (categorical) with mode = '{mode_val}'")

    # ── 3d. Encode categorical features ──────────────────────────────────────
    for col in cat_cols:
        unique_vals = df[col].nunique()
        if unique_vals == 2:
            # Binary → Label encode
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            print(f"  Label-encoded '{col}' ({unique_vals} classes)")
        else:
            # Nominal → One-hot encode (drop first to avoid multicollinearity)
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            print(f"  One-hot-encoded '{col}' → {dummies.shape[1]} dummy cols")

    # ── 3e. Separate features / target ───────────────────────────────────────
    y = df[target].astype(int)
    X = df.drop(columns=[target])

    # Cast bool columns (from get_dummies) to int8 for memory efficiency
    bool_cols = X.select_dtypes(include=["bool"]).columns
    X[bool_cols] = X[bool_cols].astype(np.int8)

    print(f"\n  Feature matrix shape : {X.shape}")
    print(f"  Target distribution  : {dict(y.value_counts().sort_index())}")
    return X, y, X.columns.tolist()


# =============================================================================
# 4. TRAIN-TEST SPLIT + FEATURE SCALING
# =============================================================================

def split_and_scale(X: pd.DataFrame, y: pd.Series, test_size: float = 0.20):
    """
    Stratified split (preserves class ratio) + StandardScaler.
    Scaler is fit ONLY on training data to prevent data leakage.
    """
    print(f"\n{'='*60}")
    print("  STEP 4 — SPLIT & SCALE")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"  Train size : {X_train_sc.shape[0]} samples")
    print(f"  Test  size : {X_test_sc.shape[0]}  samples")
    return X_train_sc, X_test_sc, y_train, y_test, scaler


# =============================================================================
# 5. CLASS IMBALANCE HANDLING
# =============================================================================

def get_class_weights(y_train: pd.Series) -> dict:
    """
    Compute balanced class weights.
    These are passed to LogisticRegression(class_weight=...) so the model
    penalises misclassification of the minority class more heavily.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = dict(zip(classes, weights))
    print(f"\n  Class weights (balanced): {cw}")
    return cw


# =============================================================================
# 6. HYPERPARAMETER TUNING — GridSearchCV
# =============================================================================

def tune_hyperparameters(X_train, y_train, class_weights: dict):
    """
    Grid-search over C (regularisation strength) and solver.
    Uses StratifiedKFold to maintain class ratios in each fold.

    Regularisation recap
    --------------------
    penalty='l2' (Ridge)  → shrinks all coefficients; handles multicollinearity
    penalty='l1' (Lasso)  → drives some coefficients to 0; automatic feature selection
    C = 1/λ  → smaller C = stronger regularisation (helps overfitting)
    """
    print(f"\n{'='*60}")
    print("  STEP 5 — HYPERPARAMETER TUNING (GridSearchCV)")
    print(f"{'='*60}")

    param_grid = [
        {
            "penalty": ["l2"],
            "C":       [0.001, 0.01, 0.1, 1, 10, 100],
            "solver":  ["lbfgs", "saga"],
        },
        {
            "penalty": ["l1"],
            "C":       [0.001, 0.01, 0.1, 1, 10, 100],
            "solver":  ["liblinear", "saga"],
        },
    ]

    base_model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
        class_weight=class_weights,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",     # AUC is robust to class imbalance
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    print(f"\n  Best parameters : {grid_search.best_params_}")
    print(f"  Best CV AUC     : {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_, grid_search


# =============================================================================
# 7. CROSS-VALIDATION
# =============================================================================

def cross_validate_model(model, X_train, y_train) -> pd.DataFrame:
    """
    5-fold stratified cross-validation on the training set.
    Provides an unbiased estimate of generalisation performance
    BEFORE looking at the test set.
    """
    print(f"\n{'='*60}")
    print("  STEP 6 — CROSS-VALIDATION (5-Fold Stratified)")
    print(f"{'='*60}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    cv_results = cross_validate(
        model, X_train, y_train, cv=cv, scoring=scoring,
        return_train_score=True, n_jobs=-1
    )

    summary = {}
    for metric in scoring:
        train_scores = cv_results[f"train_{metric}"]
        val_scores   = cv_results[f"test_{metric}"]
        summary[metric] = {
            "Train mean": train_scores.mean(),
            "Train std" : train_scores.std(),
            "Val mean"  : val_scores.mean(),
            "Val std"   : val_scores.std(),
        }
        gap = train_scores.mean() - val_scores.mean()
        note = "Potential overfitting" if gap > 0.05 else "OK"
        print(f"  {metric:12s}  train={train_scores.mean():.4f}±{train_scores.std():.4f}"
              f"  val={val_scores.mean():.4f}±{val_scores.std():.4f}  [{note}]")

    return pd.DataFrame(summary).T


# =============================================================================
# 8. MODEL EVALUATION
# =============================================================================

def evaluate_model(model, X_test, y_test, feature_names: list):
    """
    Full evaluation suite on the held-out test set.
    """
    print(f"\n{'='*60}")
    print("  STEP 7 — MODEL EVALUATION (Test Set)")
    print(f"{'='*60}")

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_pred_prob)

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")

    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")

    return {
        "y_pred": y_pred, "y_pred_prob": y_pred_prob,
        "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "auc": auc,
    }


# =============================================================================
# 9. VISUALISATIONS
# =============================================================================

def plot_all(model, X_test, y_test, y_pred, y_pred_prob,
             feature_names: list, grid_search) -> None:
    """
    Render a 2×3 dashboard:
      [0,0] Confusion Matrix   [0,1] ROC Curve      [0,2] Precision-Recall Curve
      [1,0] Coeff Importance   [1,1] CV AUC dist.   [1,2] Probability Distribution
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor("#0f0f0f")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 9.1 Confusion Matrix ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    cm  = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=1, linecolor="#0f0f0f",
                ax=ax1, cbar=False,
                annot_kws={"size": 16, "weight": "bold"})
    ax1.set_title("Confusion Matrix", color=ACCENT, fontsize=13, pad=10)
    ax1.set_xlabel("Predicted", fontsize=10)
    ax1.set_ylabel("Actual",    fontsize=10)
    ax1.set_xticklabels(["No (0)", "Yes (1)"])
    ax1.set_yticklabels(["No (0)", "Yes (1)"], rotation=0)

    # ── 9.2 ROC Curve ────────────────────────────────────────────────────────
    ax2  = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_score   = roc_auc_score(y_test, y_pred_prob)
    ax2.plot(fpr, tpr, color=ACCENT,  lw=2.5, label=f"AUC = {auc_score:.4f}")
    ax2.plot([0,1],[0,1], color="#555", lw=1.5, linestyle="--", label="Random")
    ax2.fill_between(fpr, tpr, alpha=0.12, color=ACCENT)
    ax2.set_title("ROC Curve", color=ACCENT, fontsize=13, pad=10)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ── 9.3 Precision-Recall Curve ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_pred_prob)
    ap = average_precision_score(y_test, y_pred_prob)
    ax3.plot(rec_curve, prec_curve, color=ACCENT2, lw=2.5, label=f"AP = {ap:.4f}")
    ax3.fill_between(rec_curve, prec_curve, alpha=0.12, color=ACCENT2)
    ax3.set_title("Precision-Recall Curve", color=ACCENT, fontsize=13, pad=10)
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # ── 9.4 Coefficient / Feature Importance ─────────────────────────────────
    ax4   = fig.add_subplot(gs[1, 0])
    coefs = model.coef_[0]
    idx   = np.argsort(np.abs(coefs))[::-1][:15]   # top-15
    feat_labels  = [feature_names[i] for i in idx]
    feat_vals    = coefs[idx]
    colours      = [ACCENT if v > 0 else ACCENT2 for v in feat_vals]

    bars = ax4.barh(feat_labels[::-1], feat_vals[::-1],
                    color=colours[::-1], edgecolor="#333", height=0.65)
    ax4.axvline(0, color="#888", lw=1.2)
    ax4.set_title("Top-15 Feature Coefficients", color=ACCENT, fontsize=13, pad=10)
    ax4.set_xlabel("Coefficient value (scaled)")
    ax4.grid(True, axis="x", alpha=0.3)

    # ── 9.5 CV AUC Distribution ──────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    cv_scores = grid_search.cv_results_["mean_test_score"]
    ax5.hist(cv_scores, bins=20, color=ACCENT, edgecolor="#0f0f0f", alpha=0.85)
    ax5.axvline(cv_scores.max(), color=ACCENT2, lw=2.5, linestyle="--",
                label=f"Best = {cv_scores.max():.4f}")
    ax5.set_title("CV AUC Score Distribution", color=ACCENT, fontsize=13, pad=10)
    ax5.set_xlabel("Mean CV AUC")
    ax5.set_ylabel("Count")
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # ── 9.6 Predicted Probability Distribution ───────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    prob_0 = y_pred_prob[y_test == 0]
    prob_1 = y_pred_prob[y_test == 1]
    ax6.hist(prob_0, bins=25, color=ACCENT2, alpha=0.7, label="Class 0", edgecolor="#0f0f0f")
    ax6.hist(prob_1, bins=25, color=ACCENT,  alpha=0.7, label="Class 1", edgecolor="#0f0f0f")
    ax6.axvline(0.5, color="#fff", lw=1.5, linestyle="--", label="Threshold=0.5")
    ax6.set_title("Predicted Probability Distribution", color=ACCENT, fontsize=13, pad=10)
    ax6.set_xlabel("P(class = 1)")
    ax6.set_ylabel("Frequency")
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    fig.suptitle("Logistic Regression — Model Dashboard",
                 fontsize=18, color="#ffffff", y=1.01, fontweight="bold")

    out_path = "logistic_regression_dashboard.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print(f"\n  Dashboard saved → {out_path}")


# =============================================================================
# 10. PROBABILITY PREDICTION (inference helper)
# =============================================================================

def predict_new_samples(model, scaler, feature_names: list) -> None:
    """
    Demonstrate how to predict on new, unseen samples.
    Replace the `new_data` dict with real values for your dataset.
    """
    print(f"\n{'='*60}")
    print("  STEP 9 — PROBABILITY PREDICTION ON NEW SAMPLES")
    print(f"{'='*60}")

    # Example: two hypothetical Titanic passengers
    sample_1 = {
        "Pclass": 1, "Sex": 0,   # 0=female after label-encoding
        "Age": 29, "SibSp": 0, "Parch": 0, "Fare": 211.3375,
        "Embarked_Q": 0, "Embarked_S": 0,
        "FamilySize": 1, "IsAlone": 1,
    }
    sample_2 = {
        "Pclass": 3, "Sex": 1,   # 1=male
        "Age": 22, "SibSp": 1, "Parch": 0, "Fare": 7.25,
        "Embarked_Q": 0, "Embarked_S": 1,
        "FamilySize": 2, "IsAlone": 0,
    }

    for i, sample in enumerate([sample_1, sample_2], 1):
        # Align columns exactly with training feature order
        row = pd.DataFrame([sample]).reindex(columns=feature_names, fill_value=0)
        row_sc   = scaler.transform(row)
        prob     = model.predict_proba(row_sc)[0]
        pred_cls = model.predict(row_sc)[0]
        print(f"\n  Sample {i}: P(survive=0)={prob[0]:.4f}  "
              f"P(survive=1)={prob[1]:.4f}  → Predicted class: {pred_cls}")


# =============================================================================
# 11. FINAL SUMMARY REPORT
# =============================================================================

def print_final_summary(metrics: dict, best_model, cv_df: pd.DataFrame) -> None:
    """
    Print a structured performance summary and interpretation notes.
    """
    print(f"\n{'='*60}")
    print("  STEP 10 — FINAL SUMMARY REPORT")
    print(f"{'='*60}")

    print(f"""
  ┌─────────────────────────────────────────────┐
  │         FINAL MODEL PERFORMANCE             │
  ├─────────────────────────────────────────────┤
  │  Accuracy   : {metrics['accuracy']:.4f}                      │
  │  Precision  : {metrics['precision']:.4f}                      │
  │  Recall     : {metrics['recall']:.4f}                      │
  │  F1-Score   : {metrics['f1']:.4f}                      │
  │  ROC-AUC    : {metrics['auc']:.4f}                      │
  ├─────────────────────────────────────────────┤
  │  Best Params: {best_model.get_params()}
  └─────────────────────────────────────────────┘
""")

    print("  CROSS-VALIDATION SUMMARY (Train vs Validation):")
    print(cv_df.round(4).to_string())

    print("""
  ─────────────────────────────────────────────────
  IMPORTANT OBSERVATIONS
  ─────────────────────────────────────────────────
  1. Stratified splitting and balanced class weights
     prevent the model from ignoring the minority class.

  2. Regularisation (L1/L2) controls overfitting.
     A small C → strong regularisation. Monitor the
     train-val gap in cross-validation to tune it.

  3. One-hot encoding avoids ordinal assumptions for
     nominal features like 'Embarked'.

  4. StandardScaler is critical: without it, features
     with large numeric ranges dominate the gradient.

  5. ROC-AUC is the primary metric here because it is
     threshold-agnostic and robust to class imbalance.

  LOGISTIC REGRESSION COEFFICIENT INTERPRETATION
  ─────────────────────────────────────────────────
  • Each coefficient represents the change in log-odds
    of the positive class per unit increase in that
    feature (all others held constant).
  • Positive coeff → increases P(class=1)
  • Negative coeff → decreases P(class=1)
  • After exponentiation (exp(coef)) you get the
    Odds Ratio, e.g. exp(coef)=2 means 2× higher odds.

  ADVANTAGES OF LOGISTIC REGRESSION FOR THIS DATASET
  ─────────────────────────────────────────────────
  ✔ Interpretable: coefficients have clear meaning
  ✔ Fast to train even on large datasets
  ✔ Outputs calibrated probabilities
  ✔ Works well when the decision boundary is roughly linear
  ✔ L1 penalty provides built-in feature selection

  LIMITATIONS
  ─────────────────────────────────────────────────
  ✗ Assumes linear decision boundary → may underfit
    complex, non-linear relationships
  ✗ Sensitive to correlated features (multicollinearity)
  ✗ Requires manual feature engineering for interactions
  ✗ Outliers in numeric features can bias coefficients
    (mitigated here by StandardScaler + median imputation)
""")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    # ── Configuration ─────────────────────────────────────────────────────────
    DATASET_PATH = "Titanic-Dataset.csv"   # ← Change to your Kaggle CSV path
    TARGET_COL   = "Survived"    # ← Change to your target column name

    # ── Run pipeline ─────────────────────────────────────────────────────────
    df = load_data(DATASET_PATH)
    quick_eda(df, TARGET_COL)

    X, y, feature_names = preprocess_titanic(df, TARGET_COL)

    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    class_weights = get_class_weights(y_train)

    best_model, grid_search = tune_hyperparameters(X_train, y_train, class_weights)

    cv_df = cross_validate_model(best_model, X_train, y_train)

    metrics = evaluate_model(best_model, X_test, y_test, feature_names)

    plot_all(
        best_model, X_test, y_test,
        metrics["y_pred"], metrics["y_pred_prob"],
        feature_names, grid_search
    )

    predict_new_samples(best_model, scaler, feature_names)

    print_final_summary(metrics, best_model, cv_df)


if __name__ == "__main__":
    main()
