import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(page_title="🌧️ Rainfall Predictor", layout="wide")

# Load trained model
try:
    with open("rainfall_prediction_model.pkl", "rb") as file:
        model = joblib.load(file)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Title Section
st.markdown("""
    <h1 style='text-align: center; color: #1e88e5;'>🌧️ Rainfall Prediction Web App</h1>
    <p style='text-align: center; font-size: 18px;'>Enter the weather conditions in the sidebar to predict if it will rain.</p>
    <hr style='border: 1px solid #ccc'>
""", unsafe_allow_html=True)

# Sidebar Input Form
st.sidebar.header("📊 Weather Parameters")

# Input Fields
pressure = st.sidebar.number_input("📈 PRESSURE (hPa)", min_value=900.0, max_value=3000.0, value=1012.0, format="%.1f")
temperature = st.sidebar.number_input("🌡️ TEMPERATURE (°C)", min_value=-10.0, max_value=60.0, value=22.0, format="%.1f")
dewpoint = st.sidebar.number_input("🌫️ DEW POINT (°C)", value=15.0, format="%.1f")
humidity = st.sidebar.slider("💧 HUMIDITY (%)", 0, 100, 60)
cloud = st.sidebar.slider("☁️ CLOUD COVER (%)", 0, 100, 50)
sunshine = st.sidebar.slider("☀️ SUNSHINE (hours)", 0, 24, 6)
windspeed = st.sidebar.number_input("🍃 WIND SPEED (km/h)", value=12.0, format="%.1f")
winddirection = st.sidebar.number_input("🌬️ WIND DIRECTION (degrees 0-360)", min_value=0, max_value=360, value=90, step=1)

# Prediction Section
st.subheader("🔍 Prediction Result")

if st.button("🔮 Predict Rainfall"):
    input_data = pd.DataFrame({
        'pressure': [pressure],
        'dewpoint': [dewpoint],
        'humidity': [humidity],
        'cloud': [cloud],
        'sunshine': [sunshine],
        'winddirection': [winddirection],
        'windspeed': [windspeed]
    })

    with st.expander("🧐 See input data sent to model"):
        st.write(input_data)

    try:
        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)
        rain_prob = proba[0][1] * 100

        st.metric("Probability of Rain", f"{rain_prob:.1f}%")

        if prediction[0] == 1:
            st.success(f"☔ **Rain is likely to occur ({rain_prob:.1f}% chance). Don't forget your umbrella!**")
            st.balloons()
        else:
            st.info(f"🌤️ **No rain expected ({rain_prob:.1f}% chance). Enjoy your day!**")
    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
        st.error("Please check if the model expects exactly these features in this order.")

# Input Explanation
with st.expander("ℹ️ What do the inputs mean?"):
    st.markdown("""
    - **📈 Pressure:** Atmospheric pressure in hPa (lower = more likely to rain)
    - **🌡️ Temperature:** Current temperature in Celsius
    - **🌫️ Dew Point:** Temperature at which dew forms (closer to air temp = more humid)
    - **💧 Humidity:** Relative humidity percentage
    - **☁️ Cloud Cover:** Percentage of sky covered by clouds
    - **☀️ Sunshine:** Hours of sunshine expected
    - **🌬️ Wind Direction:** Wind direction in degrees (0-360)
    - **🍃 Wind Speed:** Wind speed in km/h
    """)

# Footer
st.markdown("<hr><center>🚀 Built with ❤️ by Nidhi Chobey | Powered by Machine Learning</center>", unsafe_allow_html=True)
