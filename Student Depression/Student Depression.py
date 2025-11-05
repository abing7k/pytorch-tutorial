"""
End-to-end workflow for predicting student depression using multiple
machine-learning models. The pipeline follows these stages:

1. Load and inspect the dataset.
2. Preprocess features (handle missing values, encode categoricals, scale).
3. Train and evaluate baseline and advanced models.
4. Tune hyperparameters for tree-based models.
5. Compare metrics and visualize feature importances.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load raw dataset and print basic diagnostics."""
    df = pd.read_csv(data_path)

    print(f"Dataset Shape: {df.shape}\n")
    print("First 5 rows:")
    print(df.head(), "\n")

    print("Dataset Info:")
    df.info()
    print()

    print("Missing Values:")
    print(df.isnull().sum(), "\n")

    print("Target Variable Distribution:")
    print(df["Depression"].value_counts(), "\n")

    print("Target Variable Percentage:")
    print(df["Depression"].value_counts(normalize=True) * 100)

    return df


def preprocess_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Preprocess features: fill NA, drop identifiers, encode categoricals."""
    df_processed = df.copy()
    df_processed["Financial Stress"].fillna(
        df_processed["Financial Stress"].median(), inplace=True
    )
    df_processed = df_processed.drop(columns=["id"])

    categorical_columns = [
        "Gender",
        "City",
        "Profession",
        "Sleep Duration",
        "Dietary Habits",
        "Degree",
        "Have you ever had suicidal thoughts ?",
        "Family History of Mental Illness",
    ]

    label_encoders: dict[str, LabelEncoder] = {}
    for col in categorical_columns:
        encoder = LabelEncoder()
        df_processed[col] = encoder.fit_transform(df_processed[col])
        label_encoders[col] = encoder

    print("\nPreprocessing completed!")
    print(f"\nProcessed Dataset Shape: {df_processed.shape}\n")
    print("Processed Dataset Preview:")
    print(df_processed.head())

    return df_processed, label_encoders


def split_and_scale(
    df_processed: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler]:
    """Split the dataset and standardize numeric features."""
    X = df_processed.drop(columns=["Depression"])
    y = df_processed["Depression"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTraining set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}\n")
    print("Training set target distribution:")
    print(y_train.value_counts(), "\n")
    print("Testing set target distribution:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    model_name: str = "Model",
):
    """Fit a model, compute performance metrics, and plot diagnostics."""
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_pred_proba_train = model.predict_proba(X_train)[:, 1]
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba_train = model.decision_function(X_train)
        y_pred_proba_test = model.decision_function(X_test)

    results = {
        "Model": model_name,
        "Train Accuracy": accuracy_score(y_train, y_pred_train),
        "Test Accuracy": accuracy_score(y_test, y_pred_test),
        "Train Precision": precision_score(y_train, y_pred_train),
        "Test Precision": precision_score(y_test, y_pred_test),
        "Train Recall": recall_score(y_train, y_pred_train),
        "Test Recall": recall_score(y_test, y_pred_test),
        "Train F1-Score": f1_score(y_train, y_pred_train),
        "Test F1-Score": f1_score(y_test, y_pred_test),
        "Train ROC-AUC": roc_auc_score(y_train, y_pred_proba_train),
        "Test ROC-AUC": roc_auc_score(y_test, y_pred_proba_test),
    }

    print("\n" + "=" * 60)
    print(f"{model_name} Performance")
    print("=" * 60)
    print(f"Training Accuracy:   {results['Train Accuracy']:.4f}")
    print(f"Testing Accuracy:    {results['Test Accuracy']:.4f}")
    print(f"Training Precision:  {results['Train Precision']:.4f}")
    print(f"Testing Precision:   {results['Test Precision']:.4f}")
    print(f"Training Recall:     {results['Train Recall']:.4f}")
    print(f"Testing Recall:      {results['Test Recall']:.4f}")
    print(f"Training F1-Score:   {results['Train F1-Score']:.4f}")
    print(f"Testing F1-Score:    {results['Test F1-Score']:.4f}")
    print(f"Training ROC-AUC:    {results['Train ROC-AUC']:.4f}")
    print(f"Testing ROC-AUC:     {results['Test ROC-AUC']:.4f}")

    cm = confusion_matrix(y_test, y_pred_test)
    print("\nConfusion Matrix (Test Set):")
    print(cm)

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_test))

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Depression", "Depression"],
        yticklabels=["No Depression", "Depression"],
    )
    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {results['Test ROC-AUC']:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return results, model


def plot_feature_importance(features: list[str], importances: np.ndarray, title: str):
    """Plot top feature importances for a fitted tree-based model."""
    feature_importance = pd.DataFrame(
        {"Feature": features, "Importance": importances}
    ).sort_values("Importance", ascending=False)

    print(f"Top 10 Most Important Features ({title}):")
    print(feature_importance.head(10))

    plt.figure(figsize=(12, 8))
    plt.barh(
        feature_importance["Feature"].head(15),
        feature_importance["Importance"].head(15),
    )
    plt.xlabel("Importance")
    plt.title(f"Top 15 Feature Importances - {title}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def compare_model_performance(results: list[dict]):
    """Visualize train/test metrics for all evaluated models."""
    results_df = pd.DataFrame(results)
    models = results_df["Model"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Train vs Test Performance Comparison", fontsize=16, fontweight="bold")

    metrics_to_compare = [
        ("Train Accuracy", "Test Accuracy", "Accuracy"),
        ("Train F1-Score", "Test F1-Score", "F1-Score"),
        ("Train ROC-AUC", "Test ROC-AUC", "ROC-AUC"),
    ]

    x = np.arange(len(models))
    width = 0.35

    for idx, (train_metric, test_metric, title) in enumerate(metrics_to_compare):
        ax = axes[idx]
        train_values = results_df[train_metric].values
        test_values = results_df[test_metric].values

        ax.bar(x - width / 2, train_values, width, label="Train", alpha=0.8)
        ax.bar(x + width / 2, test_values, width, label="Test", alpha=0.8)

        ax.set_xlabel("Models", fontweight="bold")
        ax.set_ylabel(title, fontweight="bold")
        ax.set_title(f"{title} Comparison", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    RANDOM_STATE = 42
    data_path = Path(__file__).parent / "Student Depression Dataset.csv"

    pd.set_option("display.max_columns", None)
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    df = load_dataset(data_path)
    df_processed, _label_encoders = preprocess_dataset(df)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        _scaler,
    ) = split_and_scale(df_processed)

    all_results: list[dict] = []

    print("\nEvaluation function defined successfully!")

    baseline_model = LogisticRegression(
        random_state=RANDOM_STATE, max_iter=1000, class_weight="balanced"
    )
    baseline_results, trained_baseline = evaluate_model(
        baseline_model,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        model_name="Baseline - Logistic Regression",
    )
    all_results.append(baseline_results)

    rf_model_default = RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
    )
    rf_default_results, trained_rf_default = evaluate_model(
        rf_model_default,
        X_train,
        X_test,
        y_train,
        y_test,
        model_name="Random Forest (Default)",
    )
    all_results.append(rf_default_results)
    plot_feature_importance(
        features=X_train.columns.tolist(),
        importances=trained_rf_default.feature_importances_,
        title="Random Forest",
    )

    param_grid_rf = {
        "n_estimators": [200],
        "max_depth": [None, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
    }
    rf_model = RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
    )

    print("Performing Grid Search... This may take a while.")
    grid_search_rf = GridSearchCV(
        rf_model,
        param_grid_rf,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    grid_search_rf.fit(X_train, y_train)
    print(f"\nBest Parameters: {grid_search_rf.best_params_}")
    print(f"Best Cross-Validation F1-Score: {grid_search_rf.best_score_}")

    best_rf_model = grid_search_rf.best_estimator_
    rf_tuned_results, trained_rf_tuned = evaluate_model(
        best_rf_model,
        X_train,
        X_test,
        y_train,
        y_test,
        model_name="Random Forest (Tuned)",
    )
    all_results.append(rf_tuned_results)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\nScale pos weight: {scale_pos_weight:.2f}")

    xgb_model_default = XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
    )
    xgb_default_results, trained_xgb_default = evaluate_model(
        xgb_model_default,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        model_name="XGBoost (Default)",
    )
    all_results.append(xgb_default_results)
    plot_feature_importance(
        features=X_train.columns.tolist(),
        importances=trained_xgb_default.feature_importances_,
        title="XGBoost",
    )

    param_grid_xgb = {
        "n_estimators": [200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 3],
    }
    xgb_model = XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
    )

    print("Performing Grid Search for XGBoost... This may take a while.")
    grid_search_xgb = GridSearchCV(
        xgb_model,
        param_grid_xgb,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    grid_search_xgb.fit(X_train_scaled, y_train)
    print(f"\nBest Parameters: {grid_search_xgb.best_params_}")
    print(f"Best Cross-Validation F1-Score: {grid_search_xgb.best_score_}")

    best_xgb_model = grid_search_xgb.best_estimator_
    xgb_tuned_results, trained_xgb_tuned = evaluate_model(
        best_xgb_model,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        model_name="XGBoost (Tuned)",
    )
    all_results.append(xgb_tuned_results)

    lgbm_model_default = LGBMClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, is_unbalance=True, verbose=-1
    )
    lgbm_default_results, trained_lgbm_default = evaluate_model(
        lgbm_model_default,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        model_name="LightGBM (Default)",
    )
    all_results.append(lgbm_default_results)
    plot_feature_importance(
        features=X_train.columns.tolist(),
        importances=trained_lgbm_default.feature_importances_,
        title="LightGBM",
    )

    param_grid_lgbm = {
        "n_estimators": [200],
        "max_depth": [5],
        "learning_rate": [0.1],
        "num_leaves": [31],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "min_child_samples": [20],
    }
    lgbm_model = LGBMClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, is_unbalance=True, verbose=-1
    )

    print("Performing Grid Search for LightGBM... This may take a while.")
    grid_search_lgbm = GridSearchCV(
        lgbm_model,
        param_grid_lgbm,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    grid_search_lgbm.fit(X_train_scaled, y_train)
    print(f"\nBest Parameters: {grid_search_lgbm.best_params_}")
    print(f"Best Cross-Validation F1-Score: {grid_search_lgbm.best_score_}")

    best_lgbm_model = grid_search_lgbm.best_estimator_
    lgbm_tuned_results, trained_lgbm_tuned = evaluate_model(
        best_lgbm_model,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        model_name="LightGBM (Tuned)",
    )
    all_results.append(lgbm_tuned_results)

    compare_model_performance(all_results)


if __name__ == "__main__":
    main()
