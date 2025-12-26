import streamlit as st
import pandas as pd

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="NSC – EESP Analysis & Prediction",
    layout="wide"
)

st.title("⚡ NSC – EESP Analysis & Prediction Dashboard")
st.caption("Real utility data • Analytics • ML-based future planning")

# --------------------------------------------------
# Load Data (Cloud-safe)
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(
        "nsc_data.csv",
        encoding="latin-1",
        low_memory=False
    )

df = load_data()
st.success("Data loaded successfully ✅")

# --------------------------------------------------
# Dataset Overview
# --------------------------------------------------
st.header("📊 Dataset Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Total Records", df.shape[0])
c2.metric("Total Columns", df.shape[1])
c3.metric("Missing Values", int(df.isna().sum().sum()))

with st.expander("🔍 View Sample Data"):
    st.dataframe(df.head(30))

# --------------------------------------------------
# Navigation Info
# --------------------------------------------------
st.info(
    "➡️ Go to **Model Accuracy** page from the left sidebar "
    "to view detailed ML performance (MAE, R², graphs)."
)

# --------------------------------------------------
# Load Estimation – Demo / Safe Mode
# --------------------------------------------------
st.header(" Load Estimation ")

col1, col2, col3 = st.columns(3)

with col1:
    sub_div = st.selectbox(
        "Select Sub-Division",
        sorted(df["SUB_DIV_ID"].dropna().unique())
    )

with col2:
    consumer = st.selectbox(
        "Select Consumer Type",
        sorted(df["CONN_TYPE"].dropna().unique())
    )

with col3:
    phase = st.selectbox(
        "Select Phase",
        sorted(df["APPPHASE"].dropna().unique())
    )

user_load = st.number_input(
    "Expected Load (kW)",
    min_value=0.0,
    step=1.0
)

# --------------------------------------------------
# Prediction Logic (Temporary – Demo)
# --------------------------------------------------
if st.button(" Predict Load"):

    base_load = df["SANC_LOAD"].fillna(0).mean()
    predicted_load = base_load + (user_load * 0.2)

    # Interpretation layer (important for project)
    if predicted_load < 50:
        hotspot = "Low Demand Zone"
        required_phase = "Single Phase"
    elif predicted_load < 100:
        hotspot = "Medium Demand Zone"
        required_phase = "Three Phase (Recommended)"
    else:
        hotspot = "High Demand Zone"
        required_phase = "Three Phase (Mandatory)"

    st.success("✅ Load Estimation Completed")

    st.markdown(f"""
    ###  Prediction Results
     **Demand Hotspot:** {hotspot}  
     **Estimated Load:** {predicted_load:.2f} kW  
     **Required Phase:** {required_phase}  
     **Sub-Division:** {sub_div}  
     **Consumer Type:** {consumer}
    """)

