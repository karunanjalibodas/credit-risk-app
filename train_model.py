import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("german_credit_data.csv")

# Preprocessing
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df.fillna("Unknown", inplace=True)

# Create target
df["Risk"] = (df["Credit amount"] > 5000).astype(int)

# Encode categorical
le = LabelEncoder()
for col in df.select_dtypes(include="object"):
    df[col] = le.fit_transform(df[col])

# Split
X = df.drop("Risk", axis=1)
y = df["Risk"]

# Train
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save
joblib.dump(model, "model.pkl")

print("✅ Model trained & saved")