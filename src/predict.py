import joblib
import pandas as pd

from src.config import FEATURES, MODEL_PATH


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "EstateIQ has no trained model yet. "
            "Run: python -m src.train"
        )

    return joblib.load(MODEL_PATH)


def predict_price(
    bedrooms: int,
    bathrooms: int,
    toilets: int,
    parking_space: int,
    title: str,
    town: str,
    state: str,
) -> float:
    artifact = load_model()
    model = artifact["model"]

    property_data = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "toilets": toilets,
        "parking_space": parking_space,
        "title": title,
        "town": town,
        "state": state,
    }

    input_data = pd.DataFrame(
        [property_data],
        columns=FEATURES,
    )

    prediction = float(model.predict(input_data)[0])

    # Asking prices should not be negative.
    return max(0.0, prediction)
