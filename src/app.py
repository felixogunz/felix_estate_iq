import json
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.config import METRICS_PATH, PROCESSED_DATA_PATH
from src.predict import predict_price

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)

def read_metrics():
    if not METRICS_PATH.exists():
        return None

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def read_listings():
    if not PROCESSED_DATA_PATH.exists():
        return None

    return pd.read_csv(PROCESSED_DATA_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    from src.config import MODEL_PATH

    return jsonify({
        "status": "ok",
        "model_ready": MODEL_PATH.exists(),
        "dataset_ready": PROCESSED_DATA_PATH.exists(),
    })


@app.route("/api/metrics")
def metrics():
    result = read_metrics()

    if result is None:
        return jsonify({
            "ready": False,
            "message": "Train the model to see its evaluation metrics.",
        })

    return jsonify({"ready": True, **result})

@app.route("/api/options")
def options():
    df = read_listings()

    if df is None:
        return jsonify({
            "error": "Cleaned dataset not found. Run python -m src.train first."
        }), 503

    return jsonify({
        "states": sorted(df["state"].dropna().astype(str).unique().tolist()),
        "towns_by_state": {
            str(state): sorted(
                group["town"].dropna().astype(str).unique().tolist()
            )
            for state, group in df.groupby("state")
        },
        "property_types": sorted(
            df["title"].dropna().astype(str).unique().tolist()
        ),
        "listing_count": int(len(df)),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "Send property details as JSON."}), 400

    required = [
        "bedrooms",
        "bathrooms",
        "toilets",
        "parking_space",
        "title",
        "town",
        "state",
    ]

    missing = [
        field for field in required
        if field not in payload or payload[field] in (None, "")
    ]

    if missing:
        return jsonify({
            "error": f"Missing required fields: {', '.join(missing)}"
        }), 400

    numeric_fields = [
        "bedrooms",
        "bathrooms",
        "toilets",
        "parking_space",
    ]

    values = {}

    try:
        for field in numeric_fields:
            value = float(payload[field])

            if not value.is_integer() or value < 0 or value > 30:
                raise ValueError

            values[field] = int(value)

    except (TypeError, ValueError):

        return jsonify({
            "error": "Room and parking counts must be whole numbers from 0 to 30."
        }), 400

    for field in ["title", "town", "state"]:
        if not isinstance(payload[field], str) or not payload[field].strip():
            return jsonify({"error": f"{field} must be a non-empty string."}), 400

    df = read_listings()

    if df is None:
        return jsonify({
            "error": "Dataset is not ready. Prepare and train the model first."
        }), 503

    # Only allow categories represented in our dataset.
    state = payload["state"].strip()
    town = payload["town"].strip()
    title = payload["title"].strip()

    if state not in set(df["state"].astype(str)):
        return jsonify({"error": "Choose a state present in the dataset."}), 400

    matching_towns = set(
        df.loc[df["state"].astype(str) == state, "town"].astype(str)
    )

    if town not in matching_towns:
        return jsonify({
            "error": "Choose a town listed for the selected state."
        }), 400

    if title not in set(df["title"].astype(str)):
        return jsonify({
            "error": "Choose a property type present in the dataset."
        }), 400

    try:
        estimated_price = predict_price(
            bedrooms=values["bedrooms"],
            bathrooms=values["bathrooms"],
            toilets=values["toilets"],
            parking_space=values["parking_space"],
            title=title,
            town=town,
            state=state,
        )

    except FileNotFoundError:
        return jsonify({
            "error": "The model is not trained yet. Run python -m src.train."
        }), 503
    
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "EstateIQ could not produce a prediction for this input."
        }), 500

    return jsonify({
        "estimated_price_ngn": round(estimated_price, 2),
        "currency": "NGN",
        "disclaimer": (
            "This is a model estimate based on historical advertised listings, "
            "not a professional valuation or a guarantee of sale price."
        ),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
