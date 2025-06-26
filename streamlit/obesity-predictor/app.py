import streamlit as st
import requests
import matplotlib.pyplot as plt

st.set_page_config(page_title="Obesity Prediction", layout="centered")

st.title("Obesity Risk Prediction System")
st.markdown("""
This system helps predict a person's level of obesity based on lifestyle and daily habits.
Please complete the form below to receive a risk classification.
""")

with st.form("prediction_form"):
    st.subheader("Patient Lifestyle Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age (years)", min_value=5.0, max_value=120.0, step=0.5)
        height = st.number_input("Height (in meters)", min_value=1.0, max_value=2.5, step=0.1)
        weight = st.number_input("Weight (in kg)", min_value=10.0, max_value=200.0, step=0.5)
        history = st.radio("Family history of overweight?", ["yes", "no"])
        favc = st.radio("Do you frequently eat high-calorie food?", ["yes", "no"])
        fcvc = st.slider("Frequency of vegetable consumption (1 = rarely, 3 = often)", 1.0, 3.0, step=0.1)

    with col2:
        ncp = st.slider("Number of main meals per day", 1.0, 4.0, step=0.5)
        caec = st.selectbox("Eating between meals?", ["no", "Sometimes", "Frequently", "Always"])
        smoke = st.radio("Do you smoke?", ["yes", "no"])
        ch2o = st.slider("Water intake (1 = low, 3 = high)", 1.0, 3.0, step=0.1)
        scc = st.radio("Do you monitor calorie intake?", ["yes", "no"])
        faf = st.slider("Physical activity per week (0 = low, 3 = high)", 0.0, 3.0, step=0.1)
        tue = st.slider("Technology usage (hours/day)", 0.0, 3.0, step=0.1)
        calc = st.selectbox("Alcohol consumption", ["no", "Sometimes", "Frequently", "Always"])
        mtrans_display = st.selectbox("Main mode of transportation", [
            "Public Transportation", "Walking", "Automobile", "Motorbike", "Bike"
        ])

    mtrans_backend = mtrans_display.replace(" ", "_")

    submitted = st.form_submit_button("Predict")

if submitted:
    with st.spinner("Sending data to prediction model..."):
        input_data = {
            "Gender": gender,
            "Age": age,
            "Height": height,
            "Weight": weight,
            "family_history_with_overweight": history,
            "FAVC": favc,
            "FCVC": fcvc,
            "NCP": ncp,
            "CAEC": caec,
            "SMOKE": smoke,
            "CH2O": ch2o,
            "SCC": scc,
            "FAF": faf,
            "TUE": tue,
            "CALC": calc,
            "MTRANS": mtrans_backend
        }

        try:
            response = requests.post("http://localhost:8000/predict", json=input_data)
            if response.status_code == 200:
                prediction = response.json()['prediction'].replace("_", " ")

                st.success("Prediction successful.")
                st.markdown(f"### Prediction Result: **{prediction}**")

                bmi = round(weight / (height ** 2), 1)
                st.markdown(f"**Body Mass Index (BMI): {bmi} kg/m²**")

                fig, ax = plt.subplots(figsize=(8, 1.5))
                ax.axvline(bmi, color='red', linewidth=4, label="Your BMI")
                ax.set_xlim(10, 50)
                ax.set_yticks([])
                ax.set_xticks([18.5, 24.9, 29.9, 34.9, 39.9])
                ax.set_xticklabels(["Normal", "Overweight", "Obesity I", "Obesity II", "Obesity III"])
                ax.set_title("BMI Category (WHO)")
                st.pyplot(fig)
            else:
                st.error(f"Server error: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the prediction API. Ensure FastAPI is running at http://localhost:8000.")
st.markdown("---")
st.caption("This system is for educational and preventive purposes only. Please consult medical professionals for a full diagnosis.")
