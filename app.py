import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="NSC – EESP Analysis & Prediction",
    layout="wide"
)

st.title("⚡ NSC – EESP Analysis & Prediction Dashboard")

# --------------------------------------------------
# Load data from SQLite
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(
        "nsc_data.csv",
        encoding="latin-1",
        low_memory=False
    )



df = load_data()

st.success("Database connected & data loaded successfully ✅")

# --------------------------------------------------
# Prepare dropdown values
# --------------------------------------------------
sub_divs = sorted(df["SUB_DIV_ID"].dropna().unique().tolist())
conn_types = sorted(df["CONN_TYPE"].dropna().unique().tolist())
phases = sorted(df["APPPHASE"].dropna().unique().tolist())

# --------------------------------------------------
# Load ML model
# --------------------------------------------------
@st.cache_resource
def load_ml_model():
    return joblib.load("models/load_model.pkl")

model = load_ml_model()

# --------------------------------------------------
# Dataset Overview
# --------------------------------------------------
st.header("📊 Dataset Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Total Records", df.shape[0])
c2.metric("Total Columns", df.shape[1])
c3.metric("Missing Values", int(df.isna().sum().sum()))

with st.expander("🔍 View Sample Data"):
    st.dataframe(df.head(50))

# --------------------------------------------------
# STEP-8.4 : Future Demand Prediction Section
# --------------------------------------------------
st.header("🔮 Future Demand Prediction (STEP-8.4)")

col1, col2, col3 = st.columns(3)

with col1:
    ml_sub = st.selectbox("Select Sub-Division", sub_divs)

with col2:
    ml_cat = st.selectbox("Select Consumer Category", conn_types)

with col3:
    ml_phase = st.selectbox("Select Phase", phases)

col4, col5 = st.columns(2)

with col4:
    month = st.selectbox(
        "Select Month",
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )

with col5:
    input_load = st.number_input(
        "Enter Expected Load (kW)",
        min_value=0.0,
        step=1.0
    )

# --------------------------------------------------
# Prediction Button & Logic (STEP-8.4.2)
# --------------------------------------------------
if st.button("🚀 Predict Future Demand"):

    # ML Load Prediction (baseline)
    ml_predicted_load = model.predict([[ml_phase]])[0]

    # Rule-based Future Demand Logic
    if input_load < 50:
        hotspot = "Low Demand Zone"
        required_phase = "1 Phase"
        capacity = "Existing capacity sufficient"
    elif input_load < 100:
        hotspot = "Medium Demand Zone"
        required_phase = "3 Phase (Recommended)"
        capacity = "Moderate capacity planning required"
    else:
        hotspot = "High Demand Zone"
        required_phase = "3 Phase (Mandatory)"
        capacity = "High capacity upgrade required"

    st.success("✅ Future Demand Prediction Completed")

    st.markdown(f"""
    ### 📌 Prediction Results
    🔥 **Future Request Hotspot:** {hotspot}  
    ⚡ **ML Predicted Load:** {ml_predicted_load:.2f} kW  
    🔢 **User Expected Load:** {input_load} kW  
    🔌 **Required Phase:** {required_phase}  
    🏗️ **Recommended Capacity:** {capacity}  
    📅 **Month Considered:** {month}  
    """)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Final Year Project | NSC – EESP | Streamlit • ML • SQLite")


