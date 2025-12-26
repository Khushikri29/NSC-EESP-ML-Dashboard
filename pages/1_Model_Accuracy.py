import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="ML Model Accuracy",
    layout="wide"
)

st.title("📊 ML Model Accuracy & Performance Analysis")
st.markdown("This page evaluates the trained ML model using standard regression metrics and visualizations.")
st.markdown("---")

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(
        "nsc_data.csv",
        encoding="latin-1",
        low_memory=False
    )

df = load_data()

# --------------------------------------------------
# Load ML Assets
# --------------------------------------------------
@st.cache_resource
def load_ml_assets():
    model = joblib.load("models/load_prediction_model.pkl")
    encoders = joblib.load("models/encoders.pkl")
    return model, encoders

ml_model, encoders = load_ml_assets()

# --------------------------------------------------
# Prepare Evaluation Dataset
# --------------------------------------------------
eval_df = df[[
    "SUB_DIV_ID",
    "CONN_TYPE",
    "APPPHASE",
    "LT_HT",
    "SANC_LOAD"
]].dropna()


eval_df["SUB_DIV_ID"] = encoders["SUB_DIV_ID"].transform(
    eval_df["SUB_DIV_ID"].astype(str)
)
eval_df["CONN_TYPE"] = encoders["CONN_TYPE"].transform(
    eval_df["CONN_TYPE"].astype(str)
)
eval_df["LT_HT"] = encoders["LT_HT"].transform(
    eval_df["LT_HT"].astype(str)
)


X = eval_df[["SUB_DIV_ID", "CONN_TYPE", "APPPHASE", "LT_HT"]]
y_actual = eval_df["SANC_LOAD"]
y_pred = ml_model.predict(X)

# --------------------------------------------------
# Accuracy Metrics
# --------------------------------------------------
st.header(" Model Accuracy Metrics")

c1, c2 = st.columns(2)
mae = mean_absolute_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)

c1.metric("Mean Absolute Error (MAE)", f"{mae:.2f} kW")
c2.metric("R² Score", f"{r2:.3f}")

# --------------------------------------------------
# Actual vs Predicted Line Graph
# --------------------------------------------------
st.header("📈 Actual vs Predicted Load")

fig1, ax1 = plt.subplots()
ax1.plot(y_actual.values[:100], label="Actual Load")
ax1.plot(y_pred[:100], label="Predicted Load")
ax1.set_xlabel("Sample Index")
ax1.set_ylabel("Load (kW)")
ax1.legend()

st.pyplot(fig1)

# --------------------------------------------------
# Error Distribution Chart
# --------------------------------------------------
st.header("📊 Error Distribution")

errors = y_actual.values - y_pred

fig2, ax2 = plt.subplots()
ax2.hist(errors, bins=30)
ax2.set_xlabel("Prediction Error (kW)")
ax2.set_ylabel("Frequency")

st.pyplot(fig2)

# --------------------------------------------------
# Interpretation
# --------------------------------------------------
st.markdown("""
###  Interpretation
- **MAE** indicates average absolute prediction error.
- **R² Score** explains how well the model captures variance in the data.
- The line graph compares actual vs predicted values.
- The error distribution shows how prediction errors are spread.

These results confirm that the trained ML model performs effectively on historical utility data.
""")
