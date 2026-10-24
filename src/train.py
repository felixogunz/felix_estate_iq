import json

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    TARGET,
)
from src.data_preparation import load_and_clean_data


def train_model():
    df = load_and_clean_data()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numbers", "passthrough", NUMERIC_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=250,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("\nTraining EstateIQ model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))

    metrics = {
        "model": "RandomForestRegressor",
        "dataset": "Nigerian property listings",
        "target": "asking price in Nigerian naira",
        "currency": "NGN",
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURES,
        "mae_naira": round(mae, 2),
        "rmse_naira": round(rmse, 2),
        "r2": round(r2, 4),
        "random_state": 42,
    }

    artifact = {
        "model": model,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "currency": "NGN",
        "target": TARGET,
        "dataset": "Nigerian property listings",
    }

    joblib.dump(artifact, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\n--- ESTATEIQ MODEL RESULTS ---")
    print(f"Training listings: {len(X_train):,}")
    print(f"Test listings:     {len(X_test):,}")
    print(f"MAE:               ₦{mae:,.0f}")
    print(f"RMSE:              ₦{rmse:,.0f}")
    print(f"R² score:          {r2:.4f}")
    print(f"\nModel saved to:    {MODEL_PATH}")
    print(f"Metrics saved to:  {METRICS_PATH}")

    print(
        "\nImportant: these scores measure performance on a held-out "
        "sample of advertised prices, not verified market values."
    )

    return metrics


if __name__ == "__main__":
    train_model()
