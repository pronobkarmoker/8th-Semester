"""
================================================================================
  Decision Tree & Random Forest — Production-Quality ML Pipeline
  Dataset : Titanic (CSV) — swap CSV_PATH to use any Kaggle dataset
  Author  : Claude (Anthropic)
================================================================================
"""

# ─── Standard Library ────────────────────────────────────────────────────────
import warnings
import time
warnings.filterwarnings("ignore")

# ─── Third-Party ─────────────────────────────────────────────────────────────
import numpy  as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import (train_test_split, GridSearchCV,
                                       RandomizedSearchCV, cross_val_score,
                                       StratifiedKFold)
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.impute            import SimpleImputer
from sklearn.tree              import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble          import RandomForestClassifier
from sklearn.metrics           import (accuracy_score, precision_score,
                                       recall_score, f1_score,
                                       confusion_matrix, classification_report,
                                       roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.pipeline          import Pipeline
from sklearn.inspection        import permutation_importance

matplotlib.rcParams.update({
    "figure.dpi"       : 120,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "font.family"      : "DejaVu Sans",
})

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION  — edit this block to adapt to any Kaggle CSV
# ══════════════════════════════════════════════════════════════════════════════
CSV_PATH   = "Titanic-Dataset.csv"          # ← path to your downloaded Kaggle CSV
TARGET_COL = "Survived"             # ← name of the label column
DROP_COLS  = ["PassengerId",        # ← columns that leak or are irrelevant
               "Name", "Ticket", "Cabin"]

# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_data(path: str) -> pd.DataFrame:
    """Load CSV with basic sanity checks and memory optimisation."""
    print(f"\n{'─'*60}")
    print(f"  Loading dataset: {path}")
    print(f"{'─'*60}")

    df = pd.read_csv(path, low_memory=False)

    # Downcast numeric columns to save RAM (important for large Kaggle files)
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    print(f"  Shape            : {df.shape}")
    print(f"  Memory usage     : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print(f"  Target classes   : {df[TARGET_COL].value_counts().to_dict()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3.  EXPLORATORY DATA ANALYSIS  (brief summary printed to console)
# ══════════════════════════════════════════════════════════════════════════════
def eda(df: pd.DataFrame) -> None:
    print(f"\n{'─'*60}")
    print("  EDA Summary")
    print(f"{'─'*60}")
    print(df.info())
    print("\n  Missing values (%):")
    miss = (df.isnull().mean() * 100).round(2)
    print(miss[miss > 0].to_string())
    print("\n  Descriptive statistics:")
    print(df.describe().T.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame) -> tuple:
    """
    Returns
    -------
    X_train, X_test, y_train, y_test : split arrays
    feature_names                     : list[str]
    """
    print(f"\n{'─'*60}")
    print("  Preprocessing")
    print(f"{'─'*60}")

    # ── 4a. Drop irrelevant columns ──────────────────────────────────────────
    existing_drops = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=existing_drops)
    print(f"  Dropped columns  : {existing_drops}")

    # ── 4b. Separate features / target ───────────────────────────────────────
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].copy()

    # ── 4c. Identify column types ────────────────────────────────────────────
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"  Numeric features : {num_cols}")
    print(f"  Categorical feat : {cat_cols}")

    # ── 4d. Impute missing values ─────────────────────────────────────────────
    #   Numeric  → median  (robust to outliers)
    #   Categor. → most-frequent
    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        X[num_cols] = num_imputer.fit_transform(X[num_cols])

    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

    # ── 4e. Encode categoricals ───────────────────────────────────────────────
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    # ── 4f. Encode target if necessary ───────────────────────────────────────
    if y.dtype == object:
        y = le.fit_transform(y.astype(str))

    feature_names = X.columns.tolist()

    # ── 4g. Train / test split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train size       : {X_train.shape}  |  Test size: {X_test.shape}")

    # ── 4h. Feature scaling (kept for completeness; trees don't require it) ──
    #   We scale for consistency but pass raw arrays to the tree models.
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    print("  Scaling done (StandardScaler). Trees will use raw features.")

    return (X_train, X_test, y_train, y_test,
            X_train_sc, X_test_sc, feature_names)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  HYPERPARAMETER TUNING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def tune_decision_tree(X_train, y_train) -> DecisionTreeClassifier:
    """GridSearchCV tuning for Decision Tree."""
    print(f"\n{'─'*60}")
    print("  Tuning Decision Tree (GridSearchCV) …")
    print(f"{'─'*60}")

    param_grid = {
        "max_depth"        : [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf" : [1, 2, 4],
        "criterion"        : ["gini", "entropy"],
    }
    base_dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    gs = GridSearchCV(
        base_dt, param_grid, scoring="f1_weighted",
        cv=cv, n_jobs=-1, verbose=0
    )
    gs.fit(X_train, y_train)
    print(f"  Best params      : {gs.best_params_}")
    print(f"  Best CV F1       : {gs.best_score_:.4f}")
    return gs.best_estimator_


def tune_random_forest(X_train, y_train) -> RandomForestClassifier:
    """RandomizedSearchCV tuning for Random Forest (faster for large param space)."""
    print(f"\n{'─'*60}")
    print("  Tuning Random Forest (RandomizedSearchCV) …")
    print(f"{'─'*60}")

    param_dist = {
        "n_estimators"     : [100, 200, 300, 500],
        "max_depth"        : [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf" : [1, 2, 4],
        "max_features"     : ["sqrt", "log2", None],
        "bootstrap"        : [True, False],
    }
    base_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rs = RandomizedSearchCV(
        base_rf, param_dist, n_iter=30, scoring="f1_weighted",
        cv=cv, n_jobs=-1, verbose=0, random_state=RANDOM_STATE
    )
    rs.fit(X_train, y_train)
    print(f"  Best params      : {rs.best_params_}")
    print(f"  Best CV F1       : {rs.best_score_:.4f}")
    return rs.best_estimator_


# ══════════════════════════════════════════════════════════════════════════════
# 6.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_model(model, X_test, y_test,
                   X_train, y_train,
                   model_name: str) -> dict:
    """Full evaluation suite; returns a metrics dict."""
    print(f"\n{'═'*60}")
    print(f"  Evaluation — {model_name}")
    print(f"{'═'*60}")

    y_pred  = model.predict(X_test)
    classes = np.unique(y_test)
    is_bin  = len(classes) == 2

    metrics = {
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall"   : recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1"       : f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    # ROC-AUC (binary only for simplicity; extend with OvR for multiclass)
    if is_bin and hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
    else:
        metrics["roc_auc"] = None

    # Cross-validation (on training set)
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_f1   = cross_val_score(model, X_train, y_train,
                               cv=cv, scoring="f1_weighted", n_jobs=-1)
    metrics["cv_f1_mean"] = cv_f1.mean()
    metrics["cv_f1_std"]  = cv_f1.std()

    # Print summary
    for k, v in metrics.items():
        val = f"{v:.4f}" if v is not None else "N/A"
        print(f"  {k:<20}: {val}")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Overfitting check
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = metrics["accuracy"]
    gap       = train_acc - test_acc
    print(f"  Train accuracy   : {train_acc:.4f}")
    print(f"  Test  accuracy   : {test_acc:.4f}")
    print(f"  Generalisation Δ : {gap:.4f}  "
          + ("⚠ Possible overfit" if gap > 0.10 else "✓ Good"))

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 7.  VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(model, X_test, y_test, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, ax=ax, colorbar=False,
        cmap="Blues"
    )
    ax.set_title(title, fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"confusion_{title.replace(' ', '_').lower()}.png")
    plt.show()


def plot_roc_curves(dt_model, rf_model, X_test, y_test) -> None:
    """Overlay ROC curves for both models (binary classification)."""
    if len(np.unique(y_test)) != 2:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    for model, label, color in [
        (dt_model, "Decision Tree", "#E05C5C"),
        (rf_model, "Random Forest", "#3C7DC4"),
    ]:
        prob  = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc   = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{label}  (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve Comparison", xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curves.png")
    plt.show()


def plot_feature_importance(model, feature_names: list, title: str,
                             top_n: int = 15) -> None:
    """Bar chart of top-N feature importances."""
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    names       = [feature_names[i] for i in indices]
    vals        = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = sns.color_palette("viridis", top_n)
    ax.barh(names[::-1], vals[::-1], color=colors[::-1])
    ax.set(title=f"Feature Importance — {title}",
           xlabel="Importance score")
    ax.set_title(f"Feature Importance — {title}", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"feat_importance_{title.replace(' ', '_').lower()}.png")
    plt.show()


def plot_decision_tree(model, feature_names: list,
                       max_depth_display: int = 3) -> None:
    """Render the decision tree structure (first few levels)."""
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(
        model, feature_names=feature_names,
        filled=True, rounded=True,
        max_depth=max_depth_display,
        fontsize=9, ax=ax
    )
    ax.set_title("Decision Tree Structure (first 3 levels)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("decision_tree_structure.png")
    plt.show()

    # Text representation (useful for debugging)
    print("\n  Decision Tree rules (max_depth=3):")
    print(export_text(model, feature_names=feature_names, max_depth=3))


def plot_comparison(dt_metrics: dict, rf_metrics: dict) -> None:
    """Side-by-side bar chart for key metrics."""
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    labels      = [k.capitalize() for k in metric_keys]
    dt_vals     = [dt_metrics[k] for k in metric_keys]
    rf_vals     = [rf_metrics[k] for k in metric_keys]

    x   = np.arange(len(labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1  = ax.bar(x - w/2, dt_vals, w, label="Decision Tree",
                 color="#E05C5C", alpha=0.88)
    b2  = ax.bar(x + w/2, rf_vals, w, label="Random Forest",
                 color="#3C7DC4", alpha=0.88)
    ax.bar_label(b1, fmt="%.3f", fontsize=8)
    ax.bar_label(b2, fmt="%.3f", fontsize=8)
    ax.set(xticks=x, xticklabels=labels,
           ylim=[0, 1.10], ylabel="Score",
           title="Model Performance Comparison")
    ax.set_title("Model Performance Comparison", fontweight="bold", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()


def plot_cv_distributions(dt_model, rf_model,
                           X_train, y_train) -> None:
    """Violin plot of 5-fold CV F1 scores for both models."""
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    dt_cv = cross_val_score(dt_model, X_train, y_train,
                             cv=cv, scoring="f1_weighted", n_jobs=-1)
    rf_cv = cross_val_score(rf_model, X_train, y_train,
                             cv=cv, scoring="f1_weighted", n_jobs=-1)

    fig, ax = plt.subplots(figsize=(6, 4))
    data    = pd.DataFrame({"Decision Tree": dt_cv, "Random Forest": rf_cv})
    sns.violinplot(data=data, palette=["#E05C5C", "#3C7DC4"],
                   inner="point", ax=ax)
    ax.set(ylabel="F1-Score (weighted)", title="5-Fold CV Score Distribution")
    ax.set_title("5-Fold CV Score Distribution", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig("cv_distributions.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 8.  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def print_final_summary(dt_metrics: dict, rf_metrics: dict) -> None:
    banner = "═" * 60
    print(f"\n{banner}")
    print("  FINAL SUMMARY")
    print(banner)

    header = f"  {'Metric':<22} {'Decision Tree':>14} {'Random Forest':>14}"
    print(header)
    print(f"  {'─'*50}")

    all_keys = ["accuracy", "precision", "recall", "f1", "roc_auc",
                "cv_f1_mean", "cv_f1_std"]
    for k in all_keys:
        dt_v = dt_metrics.get(k)
        rf_v = rf_metrics.get(k)
        dt_s = f"{dt_v:.4f}" if dt_v is not None else "  N/A"
        rf_s = f"{rf_v:.4f}" if rf_v is not None else "  N/A"
        print(f"  {k:<22} {dt_s:>14} {rf_s:>14}")

    # Determine winner
    winner = ("Random Forest" if rf_metrics["f1"] >= dt_metrics["f1"]
              else "Decision Tree")
    delta  = abs(rf_metrics["f1"] - dt_metrics["f1"])

    print(f"\n  Winner (by weighted F1): {winner}  (+{delta:.4f})")
    print(f"""
  Key Observations
  ────────────────
  1. Random Forest almost always generalises better than a single Decision
     Tree because it averages the predictions of many decorrelated trees,
     reducing variance while keeping bias low (bias-variance trade-off).

  2. Decision Trees are highly interpretable — you can print the rules and
     follow the path for any individual prediction. Random Forests sacrifice
     that transparency for accuracy.

  3. The cross-validation standard deviation for Random Forest is typically
     smaller, confirming more stable generalisation across unseen splits.

  4. If the generalisation gap (train acc − test acc) is large for the
     Decision Tree, it is over-fitting. Reducing max_depth or increasing
     min_samples_leaf during GridSearchCV addresses this.

  5. Feature importances from Random Forest are generally more reliable than
     those from a single Decision Tree because they are aggregated over many
     trees built on different bootstrap samples.

  Recommendations
  ───────────────
  • Use Random Forest when accuracy / robustness is the priority.
  • Use Decision Tree when model explainability / audit-ability is required.
  • For very large datasets consider LightGBM / XGBoost as next steps.
""")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()

    # ── Load ──────────────────────────────────────────────────────────────────
    df = load_data(CSV_PATH)
    eda(df)

    # ── Preprocess ────────────────────────────────────────────────────────────
    (X_train, X_test, y_train, y_test,
     X_train_sc, X_test_sc,
     feature_names) = preprocess(df)

    # ── Tune & Train ─────────────────────────────────────────────────────────
    dt_model = tune_decision_tree(X_train, y_train)
    rf_model = tune_random_forest(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    dt_metrics = evaluate_model(dt_model, X_test, y_test,
                                X_train, y_train, "Decision Tree")
    rf_metrics = evaluate_model(rf_model, X_test, y_test,
                                X_train, y_train, "Random Forest")

    # ── Visualise ─────────────────────────────────────────────────────────────
    plot_confusion_matrix(dt_model, X_test, y_test, "Decision Tree")
    plot_confusion_matrix(rf_model, X_test, y_test, "Random Forest")
    plot_roc_curves(dt_model, rf_model, X_test, y_test)
    plot_feature_importance(dt_model, feature_names, "Decision Tree")
    plot_feature_importance(rf_model, feature_names, "Random Forest")
    plot_decision_tree(dt_model, feature_names, max_depth_display=3)
    plot_comparison(dt_metrics, rf_metrics)
    plot_cv_distributions(dt_model, rf_model, X_train, y_train)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_final_summary(dt_metrics, rf_metrics)
    print(f"\n  Total wall-clock time : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
