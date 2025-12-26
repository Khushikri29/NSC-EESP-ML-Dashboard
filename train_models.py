import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# -----------------------------
# Load data from SQLite
# -----------------------------
engine = create_engine("sqlite:///nsc_eesp.db")
df = pd.read_sql("SELECT * FROM nsc_requests", engine)

print("Data loaded:", df.shape)

# -----------------------------
# Simple preprocessing
# (example: load prediction using SANC_LOAD)
# -----------------------------
df = df.dropna(subset=["SANC_LOAD", "APPPHASE"])

X = df[["APPPHASE"]]
y = df["SANC_LOAD"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train model
# -----------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model trained successfully ✅")

# -----------------------------
# Save model
# -----------------------------
joblib.dump(model, "load_model.pkl")
print("Model saved as load_model.pkl ✅")
