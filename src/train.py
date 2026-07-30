"""Model training module.

Trains multiple regression models and saves the best one.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path

from src.preprocessing import load_data, create_features_target, split_data, preprocess, save_scaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=200),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Extra Trees": ExtraTreesRegressor(random_state=42, n_estimators=200),
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def evaluate_models(
    models: dict, X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        results.append(
            {"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
        )
    return pd.DataFrame(results).sort_values("R2", ascending=False)


def save_model(model, path: str | Path = PROJECT_ROOT / "models" / "model.pkl") -> None:
    joblib.dump(model, path)


def load_model(path: str | Path = PROJECT_ROOT / "models" / "model.pkl"):
    return joblib.load(path)


if __name__ == "__main__":
    df = load_data()
    X, y = create_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess(
        X_train, X_test, y_train, y_test
    )
    save_scaler(scaler)

    models = train_models(X_train_scaled, y_train)
    results = evaluate_models(models, X_test_scaled, y_test)

    best_model_name = results.iloc[0]["Model"]
    best_model = models[best_model_name]
    save_model(best_model)

    print("\n=== Model Comparison ===")
    print(results.to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    print(f"R2 Score: {results.iloc[0]['R2']:.4f}")
    print(f"RMSE: {results.iloc[0]['RMSE']:.4f}")
