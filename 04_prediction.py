import pandas as pd
import pickle

# Load model
with open("models/traffic_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# Load label encoder
with open("models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)


# Example current traffic data
data = {
    "vehicle_count": [50],
    "avg_speed": [10],
    "density": [50],
    "occupancy": [69],
    "flow": [50]
}

df = pd.DataFrame(data)


# Predict class
prediction = model.predict(df)[0]

# Predict probabilities
probabilities = model.predict_proba(df)[0]

# Convert prediction number to class name
predicted_class = encoder.inverse_transform([prediction])[0]


print("\n===== TRAFFIC PREDICTION =====")

print("Current Traffic:", predicted_class)

print("\nTraffic Probabilities:")

for class_name, probability in zip(encoder.classes_, probabilities):
    print(f"{class_name}: {probability * 100:.2f}%")