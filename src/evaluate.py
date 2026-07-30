"""Model evaluation and feature importance analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

from src.preprocessing import load_data, create_features_target, split_data, preprocess
from src.train import train_models, evaluate_models, load_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "notebooks"


def plot_feature_importance(model, feature_names: list[str], top_n: int = 10) -> None:
    if not hasattr(model, "feature_importances_"):
        print("Model does not support feature importance.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importances[indices][::-1], align="center")
    plt.yticks(range(top_n), [feature_names[i] for i in indices[::-1]])
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Most Important Features")
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "models" / "feature_importance.png", dpi=150)
    plt.close()

    print(f"\n=== Top {top_n} Feature Importances ===")
    for i, idx in enumerate(indices):
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")


def plot_residuals(y_test: pd.Series, y_pred: np.ndarray) -> None:
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")

    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=20, edgecolor="black")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")

    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "models" / "residual_plot.png", dpi=150)
    plt.close()


def plot_prediction_error(y_test: pd.Series, y_pred: np.ndarray) -> None:
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.6)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("Prediction Error Plot")
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "models" / "prediction_error.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    df = load_data()
    X, y = create_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess(
        X_train, X_test, y_train, y_test
    )

    models = train_models(X_train_scaled, y_train)
    results = evaluate_models(models, X_test_scaled, y_test)
    print("\n=== Model Evaluation ===")
    print(results.to_string(index=False))

    best_model = load_model()
    feature_names = X_train_scaled.columns.tolist()
    plot_feature_importance(best_model, feature_names)

    y_pred = best_model.predict(X_test_scaled)
    plot_residuals(y_test, y_pred)
    plot_prediction_error(y_test, y_pred)

    print("\nEvaluation plots saved to models/")
