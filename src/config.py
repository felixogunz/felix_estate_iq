from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "nigeria_houses_data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DIR / "nigeria_houses_clean.csv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "estateiq_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


TARGET = "price"
NUMERIC_FEATURES = [
    "bedrooms",
    "bathrooms",
    "toilets",
    "parking_space",
]

CATEGORICAL_FEATURES = [
    "title",
    "town",
    "state",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)