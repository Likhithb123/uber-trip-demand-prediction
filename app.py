import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Uber Trip Demand Prediction",
    page_icon="🚕",
    layout="centered"
)

# -----------------------------
# Load model
# -----------------------------
model = joblib.load("uber_trip_model.pkl")

# -----------------------------
# Header Section
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🚕 Uber Trip Demand Prediction</h1>
    <p style='text-align: center; font-size:16px; color: #6c757d;'>
    Predict expected Uber trip demand using operational and time-based insights
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# About Section
# -----------------------------
st.markdown(
    """
    ### 📊 About the App
    This application estimates **Uber trip demand** by analyzing  
    **active vehicle availability**, **dispatching base behavior**, and  
    **temporal patterns** such as day and month.

    It demonstrates an **end-to-end machine learning workflow** —
    from data analysis to a **live deployed prediction system**.
    """
)

# -----------------------------
# Input Section
# -----------------------------
st.divider()
st.markdown("### 🧾 Enter Trip Details")

col1, col2 = st.columns(2)

with col1:
    dispatch_base = st.number_input(
        "🚏 Dispatching Base Number (encoded)",
        min_value=0,
        help="Encoded identifier for Uber dispatch base"
    )

    active_vehicles = st.number_input(
        "🚗 Active Vehicles",
        min_value=0,
        help="Number of vehicles available at the given time"
    )

with col2:
    day_of_week = st.selectbox(
        "📅 Day of Week",
        options=[
            (0, "Monday"),
            (1, "Tuesday"),
            (2, "Wednesday"),
            (3, "Thursday"),
            (4, "Friday"),
            (5, "Saturday"),
            (6, "Sunday")
        ],
        format_func=lambda x: x[1]
    )[0]

    month = st.selectbox(
        "🗓️ Month",
        list(range(1, 13))
    )

# -----------------------------
# Prediction Section
# -----------------------------
st.divider()

predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:
    if st.button("🔮 Predict Trip Demand", use_container_width=True):
        input_data = np.array([[dispatch_base, active_vehicles, 2024, month, 1, day_of_week]])
        prediction = model.predict(input_data)

        st.success(f"### 🚕 Estimated Trips: **{int(prediction[0])}**")

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption(
    "Built with  using Machine Learning, Streamlit & Uber operational data"
)
