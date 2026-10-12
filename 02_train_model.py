import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import pickle

# 1. Load labeled dataset
df = pd.read_csv("data/traffic_labeled.csv")

# 2. Features used by the ML model
features = [
    "vehicle_count",
    "avg_speed",
    "density",
    "occupancy",
    "flow"
]

X = df[features]
y = df["traffic_class"]

# 3. Convert class names into numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("Classes:")
print(encoder.classes_)

# 4. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 5. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 6. Train
model.fit(X_train, y_train)

# 7. Predict
y_pred = model.predict(X_test)

# 8. Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy * 100, "%")

# 9. Detailed evaluation
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_,
        zero_division=0
    )
)

# 10. Save model
os.makedirs("models", exist_ok=True)

with open("models/traffic_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

# 11. Save encoder
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("\nModel saved successfully!")