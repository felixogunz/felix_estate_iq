import re

import pandas as pd

from src.config import (
    FEATURES,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    TARGET,
)


def normalize_column_name(name: str) -> str:
    """Convert column names into a consistent format."""
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def clean_price(value):
    """Convert a price such as '₦250,000,000' into a number."""
    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()

    # Keep digits, decimal points and minus signs.
    text = re.sub(r"[^0-9.\-]", "", text)

    try:
        return float(text) if text else None
    except ValueError:
        return None


def load_and_clean_data() -> pd.DataFrame:
    """Load, validate and save the cleaned Nigerian listings."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}\n"
            "Place nigeria_houses_data.csv in data/raw/."
        )

    df = pd.read_csv(RAW_DATA_PATH)

    # Standardise column names, e.g. Parking_space -> parking_space.
    df.columns = [normalize_column_name(c) for c in df.columns]

    # Support common alternative column names.
    aliases = {
        "property_type": "title",
        "property_title": "title",
        "price_ngn": "price",
        "price_naira": "price",
        "parking_spaces": "parking_space",
        "parking": "parking_space",
        "bedroom": "bedrooms",
        "bathroom": "bathrooms",
        "toilet": "toilets",
    }

    for old_name, new_name in aliases.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})

    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}\n"
            f"Columns found: {df.columns.tolist()}"
        )

    df = df[required].copy()

    # Convert prices and room counts to numeric values.
    df[TARGET] = df[TARGET].map(clean_price)

    for column in [
        "bedrooms",
        "bathrooms",
        "toilets",
        "parking_space",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove rows that cannot teach the model a meaningful price.
    df = df.dropna(subset=required)
    df = df[df[TARGET] > 0]

    # Room and parking counts should not be negative.
    for column in [
        "bedrooms",
        "bathrooms",
        "toilets",
        "parking_space",
    ]:
        df = df[df[column] >= 0]

    # Remove exact duplicate listings.
    df = df.drop_duplicates()

    # Clean categorical values and handle empty strings.
    for column in ["title", "town", "state"]:
        df[column] = df[column].astype(str).str.strip()
        df[column] = df[column].replace("", "Unknown")
        df[column] = df[column].fillna("Unknown")

    # Counts are whole numbers.
    for column in [
        "bedrooms",
        "bathrooms",
        "toilets",
        "parking_space",
    ]:
        df[column] = df[column].astype(int)

    if len(df) < 30:
        raise ValueError(
            f"Only {len(df)} usable rows remain. "
            "Inspect the dataset before training."
        )

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("\nEstateIQ data preparation complete")
    print(f"Original dataset: {len(pd.read_csv(RAW_DATA_PATH)):,} rows")
    print(f"Cleaned dataset:  {len(df):,} rows")
    print(f"Removed rows:     {len(pd.read_csv(RAW_DATA_PATH)) - len(df):,}")
    print(f"Saved to:         {PROCESSED_DATA_PATH}")
    print(f"States:           {df['state'].nunique()}")
    print(f"Towns:            {df['town'].nunique()}")
    print(f"Property types:   {df['title'].nunique()}")
    print(f"Price range:      ₦{df[TARGET].min():,.0f} – ₦{df[TARGET].max():,.0f}")

    return df


if __name__ == "__main__":
    load_and_clean_data()
