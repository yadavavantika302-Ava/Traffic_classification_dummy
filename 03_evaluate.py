import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/traffic_labeled.csv")

# Features
features = [
    "vehicle_count",
    "avg_speed",
    "density",
    "occupancy",
    "flow"
]

X = df[features]
y = df["traffic_class"]

# Load encoder
with open("models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

y_encoded = encoder.transform(y)

# Same split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# Load model
with open("models/traffic_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# Predict
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_,
        zero_division=0
    )
)

# Plot
plt.figure(figsize=(7, 5))

plt.imshow(cm)

plt.title("Traffic Classification Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    range(len(encoder.classes_)),
    encoder.classes_
)

plt.yticks(
    range(len(encoder.classes_)),
    encoder.classes_
)

# Display numbers inside cells
for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()