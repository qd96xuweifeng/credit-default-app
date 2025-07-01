import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("xgb_model.pkl")

# App title
st.title("Credit Default Risk Predictor")

st.markdown("Enter the features below to estimate the probability of default.")

# User inputs
tenure = st.number_input("Tenure (months)", min_value=0, max_value=240, value=24)
utilization = st.slider("Utilization Ratio", 0.0, 1.0, 0.3)
delinq_12m = st.number_input("Delinquency Count (last 12 months)", min_value=0, max_value=12, value=0)
open_to_buy = st.number_input("Open to Buy Amount ($)", min_value=0, value=5000)

# Predict button
if st.button("Predict Default Risk"):
    input_data = np.array([[tenure, utilization, delinq_12m, open_to_buy]])
    prob_default = model.predict_proba(input_data)[0, 1]
    
    st.markdown(f"### Estimated Default Probability: **{prob_default:.2%}**")

    probs = model.predict_proba(input_data)
    st.write("Predicted Probabilities:", probs)

    # Display risk category
    if prob_default >= 0.0001:
        st.error("⚠️ High Risk of Default")
    elif prob_default >= 0.00001:
        st.warning("🟠 Moderate Risk")
    else:
        st.success("🟢 Low Risk")

    # Plot
    fig, ax = plt.subplots()
    ax.bar(["No Default", "Default"], model.predict_proba(input_data)[0,1])
    ax.set_ylabel("Probability")
    ax.set_ylim(0,1)
    st.pyplot(fig)