import pandas as pd
import pickle

# -----------------------------
# 1. Load trained model
# -----------------------------

with open("models/traffic_classifier.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)


# -----------------------------
# 2. Function for prediction
# -----------------------------

def predict_traffic(vehicle_count, avg_speed, density, occupancy, flow):

    data = pd.DataFrame({
        "vehicle_count": [vehicle_count],
        "avg_speed": [avg_speed],
        "density": [density],
        "occupancy": [occupancy],
        "flow": [flow]
    })

    # Predict traffic class
    prediction = model.predict(data)[0]

    # Predict probabilities
    probabilities = model.predict_proba(data)[0]

    # Convert number to class name
    predicted_class = encoder.inverse_transform([prediction])[0]

    print("\n==============================")
    print("     TRAFFIC ANALYSIS")
    print("==============================")

    print(f"Vehicle Count : {vehicle_count}")
    print(f"Average Speed : {avg_speed}")
    print(f"Density       : {density}")
    print(f"Occupancy     : {occupancy}")
    print(f"Flow          : {flow}")

    print("\nCurrent Traffic:", predicted_class)

    print("\nPrediction Probability:")

    for class_name, probability in zip(
        encoder.classes_,
        probabilities
    ):
        print(f"{class_name:10} : {probability * 100:.2f}%")

    print("==============================\n")


# -----------------------------
# 3. Test with incoming data
# -----------------------------

# Test 1 - Low traffic
predict_traffic(
    vehicle_count=15,
    avg_speed=45,
    density=15,
    occupancy=20,
    flow=15
)

# Test 2 - Moderate traffic
predict_traffic(
    vehicle_count=15,
    avg_speed=30,
    density=35,
    occupancy=45,
    flow=30
)

# Test 3 - Heavy traffic
predict_traffic(
    vehicle_count=60,
    avg_speed=15,
    density=60,
    occupancy=70,
    flow=50
)