import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
h1 {
    color: #2c3e50;
    text-align: center;
}
h2, h3 {
    color: #34495e;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.markdown("<h1>🩺 VitalSense AI</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>AI-Powered Diabetes Risk Prediction</h3>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("---")
st.write("### 🧠 AI-powered Diabetes Risk Assessment")

# Mode selection
mode = st.radio("Select Mode", ["Clinical Mode", "Quick Check (User)"])
st.sidebar.markdown("## 🧪 Clinical Inputs")
st.sidebar.markdown("Adjust values based on patient data")
st.sidebar.markdown("---")
# ------------------ CLINICAL MODE ------------------
if mode == "Clinical Mode":
    st.info("""
🔹 Clinical Mode: Use lab test values (for doctors / reports)  
🔹 Quick Check: Simple estimate using basic details  
""")

    st.sidebar.header("Enter Clinical Data")

    preg = st.sidebar.number_input("Pregnancies", 0, 20, 1)
    glucose = st.sidebar.number_input("Glucose (mg/dL)", 0, 200, 100)
    bp = st.sidebar.number_input("Blood Pressure", 0, 150, 70)
    skin = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.number_input("Insulin", 0, 900, 80)
    bmi = st.sidebar.number_input("BMI", 0.0, 60.0, 25.0)
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.sidebar.number_input("Age", 1, 120, 30)

    if st.button("Predict"):
        data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        prediction = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]

        if prediction == 1:
            st.markdown(f"""
            <div style="background-color:#ffe6e6;padding:20px;border-radius:10px">
            <h3 style="color:red;">⚠️ High Risk Detected</h3>
            <p><b>Probability:</b> {prob*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color:#e6ffe6;padding:20px;border-radius:10px">
            <h3 style="color:green;">✅ Low Risk</h3>
            <p><b>Probability:</b> {prob*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

        # Feature importance
        feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                         "Insulin", "BMI", "DPF", "Age"]

        importance = model.feature_importances_

        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        st.write("### 🔍 What influenced this prediction?")
        st.bar_chart(imp_df.set_index("Feature"))

# ------------------ USER MODE ------------------
else:
    st.info("Quick self-assessment (approximate prediction).")

    st.sidebar.header("Enter Basic Details")

    age = st.sidebar.number_input("Age", 1, 120, 30)
    weight = st.sidebar.number_input("Weight (kg)", 30, 200, 70)
    height = st.sidebar.number_input("Height (cm)", 100, 220, 165)
    family_history = st.sidebar.selectbox("Family History of Diabetes?", ["No", "Yes"])

    # Convert to model-compatible values
    bmi = weight / ((height / 100) ** 2)

    preg = 0
    glucose = 120 if family_history == "Yes" else 100
    bp = 70
    skin = 20
    insulin = 80
    dpf = 0.5 if family_history == "Yes" else 0.2

    if st.button("Predict"):
        data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        prediction = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]

        st.write(f"Calculated BMI: {bmi:.2f}")

        if prediction == 1:
            st.error(f"⚠️ Possible Risk ({prob*100:.2f}%)")
        else:
            st.success(f"✅ Low Risk ({prob*100:.2f}%)")

        # Feature importance
        feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                         "Insulin", "BMI", "DPF", "Age"]

        importance = model.feature_importances_

        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        st.write("### 🔍 What influenced this prediction?")
        st.bar_chart(imp_df.set_index("Feature"))
        top_features=imp_df.head(3)["Feature"].values
        st.write("### 🧾 Key Factors:")
        for f in top_features:
            st.write(f"• {f} influenced the prediction")

# Footer

st.caption("⚠️ This tool is for educational purposes only. Consult a doctor for medical advice.")