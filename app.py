import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("reg_model.pkl")

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
    prob_default_orig = model.predict_proba(input_data)[0, 1]
    prob_default = min(prob_default_orig * 100,1)
    prob_Non_default = max(0,1 - prob_default)
    
    st.markdown(f"### Estimated Default Probability: **{prob_default:.2%}**")

    #probs = model.predict_proba(input_data)
    #st.write("Predicted Probabilities:", probs)

    # Display risk category
    if prob_default >= 0.1:
        st.error("⚠️ High Risk of Default")
    elif prob_default >= 0.01:
        st.warning("🟠 Moderate Risk")
    else:
        st.success("🟢 Low Risk")

    # Get class probabilities and class labels
    probs = model.predict_proba(input_data)[0]
    classes = model.classes_

    # Create a dict of label: prob, explicitly
    prob_dict = {str(cls): prob for cls, prob in zip(classes, probs)}

    # Map labels to human-readable names
    labels = ['Non-default', 'Default']
    values = [prob_Non_default, prob_default]  # class 0 = non-default, class 1 = default

    # Build the chart
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=["green", "red"])

    # Optional: Add percentage annotations
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2%}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 10),
                textcoords="offset points",
                ha='center', va='bottom')

    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title('Estimated Risk Breakdown', pad=20)
    st.pyplot(fig)