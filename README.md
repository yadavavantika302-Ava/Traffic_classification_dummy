# Traffic Classification

A simple machine learning project for classifying traffic conditions from road traffic features. The project labels raw traffic data, trains a Random Forest classifier, evaluates the model, and runs example predictions.

## Project Structure

```text
Traffic_classification/
|-- data/
|   |-- member2_traffic_features_dataset.csv
|   `-- traffic_labeled.csv
|-- models/
|   |-- label_encoder.pkl
|   `-- traffic_classifier.pkl
|-- 01_labeling.py
|-- 02_train_model.py
|-- 03_evaluate.py
|-- 04_prediction.py
|-- 05_live_prediction.py
|-- requirements.txt
`-- README.md
```

## Features Used

The model uses these input columns:

- `vehicle_count`
- `avg_speed`
- `density`
- `occupancy`
- `flow`

The predicted target is `traffic_class`, with classes such as `Low`, `Moderate`, `Heavy`, and `Severe`.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Run the scripts in order when starting from the original dataset.

### 1. Label the Dataset

```bash
python 01_labeling.py
```

This reads `data/member2_traffic_features_dataset.csv`, creates the `traffic_class` column, and saves:

```text
data/traffic_labeled.csv
```

### 2. Train the Model

```bash
python 02_train_model.py
```

This trains a Random Forest classifier and saves:

```text
models/traffic_classifier.pkl
models/label_encoder.pkl
```

### 3. Evaluate the Model

```bash
python 03_evaluate.py
```

This prints a confusion matrix and classification report, then displays a confusion matrix plot.

### 4. Run a Single Prediction

```bash
python 04_prediction.py
```

This uses one example traffic record and prints the predicted traffic class with class probabilities.

### 5. Run Multiple Example Predictions

```bash
python 05_live_prediction.py
```

This calls `predict_traffic()` with multiple sample inputs and prints traffic analysis results.

## Labeling Rules

The current labeling logic in `01_labeling.py` uses `density` and `avg_speed`:

- `Low`: density below 25 and average speed at least 35
- `Moderate`: density below 50 and average speed at least 20
- `Heavy`: density below 75 and average speed at least 10
- `Severe`: all other traffic conditions

## Notes

- Run scripts from the project root directory so relative paths like `data/...` and `models/...` work correctly.
- Retrain the model after changing the labeling logic or replacing the dataset.
- The saved model files are required before running prediction scripts.
