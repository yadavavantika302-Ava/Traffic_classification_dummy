import pandas as pd

# 1. Load dataset
df = pd.read_csv("data/member2_traffic_features_dataset.csv")

# 2. Check missing values
print("Missing values:")
print(df.isnull().sum())


# 3. Function to classify traffic
def classify_traffic(row):

    density = row["density"]
    speed = row["avg_speed"]

    if density < 25 and speed >= 35:
        return "Low"

    elif density < 50 and speed >= 20:
        return "Moderate"

    elif density < 75 and speed >= 10:
        return "Heavy"

    else:
        return "Severe"


# 4. Create traffic class
df["traffic_class"] = df.apply(classify_traffic, axis=1)


# 5. Display some results
print("\nTraffic classification:")
print(
    df[
        ["vehicle_count", "avg_speed", "density", "traffic_class"]
    ].head(20)
)


# 6. Count each class
print("\nTraffic class distribution:")
print(df["traffic_class"].value_counts())


# 7. Save the labeled dataset
df.to_csv(
    "data/traffic_labeled.csv",
    index=False
)

print("\nLabeled dataset saved successfully!")