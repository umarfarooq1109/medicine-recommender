# app.py
import streamlit as st
import pandas as pd
import pickle
from utils import load_data, get_recommendations

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
symptoms_df = pd.read_csv("datasets/Training.csv")
symptom_list = symptoms_df.columns[:-1]

# Streamlit page config
st.set_page_config(page_title="bhaai's Health AI", layout="centered")

# ----------------------------------------
# ✅ Custom Header
# ----------------------------------------
st.markdown("""
<style>
    .header {
        text-align: center;
        padding: 20px;
        background-color: #e8f5e9;
        border-radius: 16px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .header h1 {
        color: #2e7d32;
        font-size: 36px;
        margin: 10px 0 5px 0;
    }
    .header p {
        color: #555;
        font-size: 16px;
        margin: 0;
    }
</style>
<div class="header">
    <img src="https://cdn-icons-png.flaticon.com/512/4320/4320337.png" width="80">
    <h1>Farooq's Health Advisor AI 🧠</h1>
    <p>Your AI-powered Medical & Fitness Assistant</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------
# ✅ Symptom Selection UI
# ----------------------------------------
st.markdown("<h3>🩺 Select Your Symptoms Below:</h3>", unsafe_allow_html=True)
selected_symptoms = st.multiselect("Choose symptoms", symptom_list)

# ----------------------------------------
# ✅ Prediction & Output
# ----------------------------------------
if st.button("🔍 Predict Disease"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
        prediction = model.predict([input_vector])[0]
        disease = le.inverse_transform([prediction])[0]

        st.success(f"✅ **Predicted Disease:** {disease}")

        # Load recommendation data
        data = load_data()
        results = get_recommendations(disease, data)

        # ----------------------------------------
        # Description
        # ----------------------------------------
        st.markdown(f"""
        <div style="background-color:#f0fdf4; padding:15px; border-radius:10px; margin-top:20px;">
            <h4 style="color:#388e3c;">📄 Description</h4>
            <p style="color:red;">{results['description']}</p>
        </div>
        """, unsafe_allow_html=True)

        # ----------------------------------------
        # Render Each Section as Cards
        # ----------------------------------------
        def render_section(title, icon, items):
            st.markdown(f"""
            <div style="margin-top: 20px;">
                <h4 style="color:#2e7d32;">{icon} {title}</h4>
            """, unsafe_allow_html=True)

            if items:
                for item in items:
                    st.markdown(f"""
                    <div style="background-color:#f1f8e9; border-radius:10px; padding:10px; margin:5px 0;
                                box-shadow:0 2px 6px rgba(0,0,0,0.08); color:red;">
                        {item}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"No {title.lower()} available.")

        render_section("Medications", "💊", results["medications"])
        render_section("Diet Recommendations", "🍽️", results["diet"])
        render_section("Precautions", "⚠️", results["precautions"])
        render_section("Workout Suggestions", "🏋️", results["workout"])
