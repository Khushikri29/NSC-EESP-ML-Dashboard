import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv(
    "nsc_data.csv",
    encoding="latin-1",
    low_memory=False
)

# -------------------------
# Select required columns
# -------------------------
df = df[[
    "SUB_DIV_ID",
    "CONN_TYPE",
    "APPPHASE",
    "SANC_LOAD"
]].dropna()

# -------------------------
# Encode categorical features
# -------------------------
encoders = {}

for col in ["SUB_DIV_ID", "CONN_TYPE"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# -------------------------
# Features & Target
# -------------------------
X = df[["SUB_DIV_ID", "CONN_TYPE", "APPPHASE"]]
y = df["SANC_LOAD"]

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Train ML model
# -------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------
# Evaluation
# -------------------------
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# -------------------------
# Save model & encoders
# -------------------------
joblib.dump(model, "models/load_prediction_model.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print("✅ ML Model trained & saved successfully!")
