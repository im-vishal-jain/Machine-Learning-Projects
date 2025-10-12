import streamlit as st
import numpy as np
import pandas as pd
import joblib
from streamlit_lottie import st_lottie
import requests
import json

# Page configuration
st.set_page_config(
    page_title="🌧️ Rainfall Predictor", 
    layout="wide",
    page_icon="🌧️"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #1e88e5, #0d47a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .sub-header {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .input-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #1e88e5;
    }
    .prediction-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-top: 5px solid #1e88e5;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1e88e5, #0d47a1);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Function to load Lottie animations
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load animations
rain_animation = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_kbfz3bbd.json")
sun_animation = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_6e1Lvg.json")

# Load trained model
try:
    with open("rainfall_prediction_model.pkl", "rb") as file:
        model = joblib.load(file)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Header Section
st.markdown('<h1 class="main-header">🌧️ Smart Rainfall Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enter weather parameters below to predict rainfall probability with AI</p>', unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📊 Weather Parameters")
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    
    # Create a grid layout for inputs
    col1a, col1b = st.columns(2)
    
    with col1a:
        pressure = st.number_input("📈 PRESSURE (hPa)", min_value=900.0, max_value=3000.0, value=1012.0, format="%.1f")
        temperature = st.number_input("🌡️ TEMPERATURE (°C)", min_value=-10.0, max_value=60.0, value=22.0, format="%.1f")
        dewpoint = st.number_input("🌫️ DEW POINT (°C)", value=15.0, format="%.1f")
        humidity = st.slider("💧 HUMIDITY (%)", 0, 100, 60)
    
    with col1b:
        cloud = st.slider("☁️ CLOUD COVER (%)", 0, 100, 50)
        sunshine = st.slider("☀️ SUNSHINE (hours)", 0, 24, 6)
        windspeed = st.number_input("🍃 WIND SPEED (km/h)", value=12.0, format="%.1f")
        winddirection = st.slider("🌬️ WIND DIRECTION (degrees)", min_value=0, max_value=360, value=90, step=1)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction button
    if st.button("🔮 PREDICT RAINFALL", use_container_width=True):
        input_data = pd.DataFrame({
            'pressure': [pressure],
            'dewpoint': [dewpoint],
            'humidity': [humidity],
            'cloud': [cloud],
            'sunshine': [sunshine],
            'winddirection': [winddirection],
            'windspeed': [windspeed]
        })
        
        try:
            prediction = model.predict(input_data)
            proba = model.predict_proba(input_data)
            rain_prob = proba[0][1] * 100
            
            # Store results in session state
            st.session_state.prediction = prediction[0]
            st.session_state.rain_prob = rain_prob
            st.session_state.show_result = True
            
        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")

with col2:
    # Show prediction results
    if hasattr(st.session_state, 'show_result') and st.session_state.show_result:
        st.markdown("### 🔍 Prediction Result")
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        
        # Display appropriate animation
        if st.session_state.prediction == 1:
            if rain_animation:
                st_lottie(rain_animation, height=150, key="rain")
            st.metric("🌧️ RAIN PROBABILITY", f"{st.session_state.rain_prob:.1f}%", delta="High Chance")
            st.success(f"## ☔ Rain Expected!\n**{st.session_state.rain_prob:.1f}% chance of rainfall**\n\nDon't forget your umbrella!")
            st.balloons()
        else:
            if sun_animation:
                st_lottie(sun_animation, height=150, key="sun")
            st.metric("☀️ RAIN PROBABILITY", f"{st.session_state.rain_prob:.1f}%", delta="Low Chance", delta_color="inverse")
            st.info(f"## 🌤️ Clear Skies!\n**{st.session_state.rain_prob:.1f}% chance of rainfall**\n\nEnjoy your day!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show input data
        with st.expander("🧐 View Input Data Sent to Model"):
            input_df = pd.DataFrame({
                'Parameter': ['Pressure', 'Dew Point', 'Humidity', 'Cloud Cover', 'Sunshine', 'Wind Direction', 'Wind Speed'],
                'Value': [f"{pressure} hPa", f"{dewpoint}°C", f"{humidity}%", f"{cloud}%", f"{sunshine} hrs", f"{winddirection}°", f"{windspeed} km/h"]
            })
            st.dataframe(input_df, use_container_width=True)
    else:
        # Show placeholder when no prediction has been made
        st.markdown("### 🔍 Prediction Result")
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.info("👆 Enter weather parameters and click **PREDICT RAINFALL** to see the results here!")
        st.markdown('</div>', unsafe_allow_html=True)

# Information Section
st.markdown("---")
st.markdown("### ℹ️ Weather Parameter Guide")

# Create feature cards in a grid
col3, col4, col5, col6 = st.columns(4)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**📈 Pressure**")
    st.markdown("Atmospheric pressure in hPa. Lower pressure often indicates stormy weather.")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**🌡️ Temperature**")
    st.markdown("Current air temperature in Celsius. Affects evaporation and condensation rates.")
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**🌫️ Dew Point**")
    st.markdown("Temperature at which air becomes saturated. Closer to air temperature means higher humidity.")
    st.markdown('</div>', unsafe_allow_html=True)

with col6:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**💧 Humidity**")
    st.markdown("Relative humidity percentage. Higher values increase rain probability.")
    st.markdown('</div>', unsafe_allow_html=True)

# Second row of feature cards
col7, col8, col9, col10 = st.columns(4)

with col7:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**☁️ Cloud Cover**")
    st.markdown("Percentage of sky covered by clouds. More clouds = higher rain chance.")
    st.markdown('</div>', unsafe_allow_html=True)

with col8:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**☀️ Sunshine**")
    st.markdown("Hours of sunshine expected. Less sunshine can indicate cloudier conditions.")
    st.markdown('</div>', unsafe_allow_html=True)

with col9:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**🌬️ Wind Direction**")
    st.markdown("Wind direction in degrees (0-360). Certain directions bring moisture.")
    st.markdown('</div>', unsafe_allow_html=True)

with col10:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**🍃 Wind Speed**")
    st.markdown("Wind speed in km/h. Affects weather system movement and intensity.")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<center>🚀 Built with ❤️ by Nidhi Chobey | Powered by Machine Learning</center>", unsafe_allow_html=True)
