# Felix Estate IQ — EstateIQ

EstateIQ is a Nigerian property price estimation application built with
Python, Flask, pandas, and scikit-learn.

## Features

- Estimates property asking prices in Nigerian naira (NGN).
- Uses bedrooms, bathrooms, toilets, parking, property type, town, and state.
- Includes a responsive dashboard with animated UI elements.
- Displays model evaluation metrics from a held-out test set.
- Provides JSON endpoints for health, options, metrics, and predictions.
- Includes automated tests for key API behaviour.

## Project structure

- `src/` — data preparation, model training, prediction, and Flask API.
- `data/raw/` — original dataset.
- `data/processed/` — cleaned dataset.
- `models/` — saved model and evaluation metrics.
- `templates/` — HTML dashboard.
- `static/` — CSS and JavaScript.
- `tests/` — automated tests.

## Setup

Create and activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate


Install dependencies:

python -m pip install -r requirements.txt


Place the Nigerian listings CSV at:

`data/raw/nigeria_houses_data.csv`

Prepare the dataset and train the model:

python -m src.train

Run the application:

python -m src.app

Open http://127.0.0.1:5000 in your browser.

Run tests:

pytest -q

## API endpoints

- `GET /api/health`
- `GET /api/metrics`
- `GET /api/options`
- `POST /api/predict`

Prediction requests should be JSON objects containing the fields
`bedrooms`, `bathrooms`, `toilets`, `parking_space`, `title`, `town`,
and `state`.

## Data and limitations

The initial dataset consists of Nigerian property listings. Prices may be
advertised asking prices rather than completed sale prices, and the dataset
may contain duplicated or stale listings.

A random train/test split measures performance on a held-out sample from
the same source; it does not establish how accurately the model predicts
current prices or properties in unfamiliar locations.

EstateIQ estimates are informational and are not professional valuations,
mortgage appraisals, or guarantees of a sale price.

Before production use, audit duplicates, outliers, data recency, geographic
coverage, and performance across locations and property-price ranges.
