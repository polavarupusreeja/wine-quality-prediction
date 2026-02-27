import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Wine Quality Prediction", 
    page_icon="🍷", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# ----------------------------
# Load Saved Model & Scaler
# ----------------------------
@st.cache_resource
def get_models():
    # Safe loading with fallback to .pkl if .sav isn't present
    model_path = "finalized_RFmodel.sav" if os.path.exists("finalized_RFmodel.sav") else "model.pkl"
    scaler_path = "scaler_model.sav" if os.path.exists("scaler_model.sav") else "scaler.pkl"
    
    with open(model_path, "rb") as f:
        m = pickle.load(f)
    with open(scaler_path, "rb") as f:
        s = pickle.load(f)
    return m, s

try:
    model, scaler = get_models()
    models_ready = True
except Exception as e:
    models_ready = False

# ----------------------------
# CSS Styling (Premium + Background Image + Grapes)
# ----------------------------
# High-quality wine background
bg_url = "https://miro.medium.com/v2/resize:fit:1200/1*8RLR69O6wQIVgeZAZyy11Q.png"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}
    
    /* Background Image setup */
    .stApp {{
        background: url('{bg_url}') no-repeat center center fixed;
        background-size: cover;
    }}
    /* Dark overlay so text is readable */
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background: linear-gradient(135deg, rgba(15, 5, 10, 0.88), rgba(45, 10, 15, 0.92));
        z-index: -1;
    }}
    
    /* Typography Overrides */
    h1, h2, h3, p, label {{
        color: #f8f0e3 !important;
    }}

    .header-title {{
        font-family: 'Playfair Display', serif;
        font-size: 4rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(to right, #F5D7A1, #c89b53);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 50px;
        margin-bottom: 0px;
    }}
    .header-sub {{
        text-align: center;
        font-family: 'Playfair Display', serif;
        font-style: italic;
        color: #c0b9a6 !important;
        font-size: 1.3rem;
        margin-bottom: 50px;
    }}

    /* Glassmorphism Form Card */
    div[data-testid="stForm"] {{
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(245, 215, 161, 0.15) !important;
        border-radius: 24px !important;
        padding: 50px !important;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5) !important;
    }}
    
    /* Sliders styling */
    .stSlider > div > div > div > div {{
        background-color: #c89b53 !important;
    }}
    
    /* Button */
    [data-testid="stFormSubmitButton"] button {{
        background: linear-gradient(135deg, #7A0025, #4A0015) !important;
        color: #FDF1D6 !important;
        border: 1px solid #c89b53 !important;
        border-radius: 50px !important;
        padding: 15px 0 !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        width: 100% !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        box-shadow: 0 10px 20px rgba(0,0,0,0.4) !important;
        margin-top: 30px;
        height: auto !important;
    }}
    [data-testid="stFormSubmitButton"] button:hover {{
        background: linear-gradient(135deg, #A80033, #6B001F) !important;
        box-shadow: 0 15px 30px rgba(200, 155, 83, 0.3) !important;
        transform: translateY(-5px) !important;
        color: #ffffff !important;
    }}

    /* Result Box & Animations */
    .result-box {{
        margin-top: 40px;
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        padding: 50px 30px;
        text-align: center;
        box-shadow: 0 20px 60px rgba(0,0,0,0.6);
        border-top: 10px solid;
        animation: slideUp 0.8s cubic-bezier(0.23, 1, 0.32, 1);
    }}
    .result-score {{
        font-size: 6rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 10px;
    }}
    .result-text {{
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #333;
    }}
    
    @keyframes slideUp {{
        from {{ opacity: 0; transform: translateY(40px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    /* Grapes Animation 🍇 */
    .falling-grape {{
        position: fixed;
        top: -10%;
        font-size: 2.5rem;
        z-index: 99999;
        pointer-events: none;
        animation: fall linear forwards;
    }}
    
    @keyframes fall {{
        0% {{ transform: translateY(0vh) rotate(0deg); opacity: 1; }}
        100% {{ transform: translateY(115vh) rotate(720deg); opacity: 0; }}
    }}

    /* Custom Scrollbar */
    ::-webkit-scrollbar {{ width: 10px; }}
    ::-webkit-scrollbar-track {{ background: #0f050a; }}
    ::-webkit-scrollbar-thumb {{ background: #7A0025; border-radius: 5px; }}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header Section
# ----------------------------
st.markdown("<div class='header-title'>Wine Quality Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='header-sub'>Evaluating wine characteristics with advanced Machine Learning.</div>", unsafe_allow_html=True)

if not models_ready:
    st.error("⚠️ System Offline: Model configuration files missing.")
    st.stop()

# ----------------------------
# Main Interaction UI
# ----------------------------
# Create spacing for center alignment
_, main_col, _ = st.columns([1, 6, 1])

with main_col:
    with st.form("vintage_form"):

        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            fixed_acidity = st.slider("Fixed Acidity", 0.0, 20.0, 7.5)
            volatile_acidity = st.slider("Volatile Acidity", 0.0, 2.0, 0.70)
            citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.0)
            pH = st.slider("pH Level", 0.0, 14.0, 3.0)
            
        with c2:
            free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 0.0, 100.0, 15.0)
            total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 0.0, 300.0, 98.0)
            chlorides = st.slider("Chlorides", 0.0, 2.0, 0.9)
            sulphates = st.slider("Sulphates", 0.0, 10.0, 0.6)
            
        with c3:
            residual_sugar = st.slider("Residual Sugar", 0.0, 15.0, 0.6)
            density = st.slider("Density", 0.0, 5.0, 1.0)
            alcohol = st.slider("Alcohol (%)", 0.0, 20.0, 11.5)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Prediction 🍷")

    # ----------------------------
    # Results Processing
    # ----------------------------
    if submitted:
        # 1. Structure the input
        input_data = pd.DataFrame([[
            fixed_acidity, volatile_acidity, citric_acid,
            residual_sugar, chlorides, free_sulfur_dioxide,
            total_sulfur_dioxide, density, pH,
            sulphates, alcohol
        ]], columns=[
            'fixed acidity', 'volatile acidity', 'citric acid',
            'residual sugar', 'chlorides', 'free sulfur dioxide',
            'total sulfur dioxide', 'density', 'pH',
            'sulphates', 'alcohol'
        ])

        # 2. Log conversion (Match training logic exactly)
        input_data["residual sugar"] = np.log(input_data["residual sugar"] + 1)
        input_data["chlorides"] = np.log(input_data["chlorides"] + 1)
        input_data["free sulfur dioxide"] = np.log(input_data["free sulfur dioxide"] + 1)
        input_data["total sulfur dioxide"] = np.log(input_data["total sulfur dioxide"] + 1)
        input_data["sulphates"] = np.log(input_data["sulphates"] + 1)

        # 3. Predict
        with st.spinner("Decoding molecular patterns..."):
            time.sleep(1.5)
            try:
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)
                final_score = int(round(prediction[0]))
                
                # 4. Stylize results
                if final_score >= 7:
                    color = "#1b5e20" # Deep Emerald
                    label = "Exceptional Vintage 🍾"
                    text = "Characteristics denote a superior quality wine with complex balance."
                elif final_score >= 5:
                    color = "#e65100" # Deep Amber
                    label = "Standard Selection 🍷"
                    text = "A consistent and approachable molecular profile."
                else:
                    color = "#b71c1c" # Deep Ruby
                    label = "Substandard Profile ❌"
                    text = "Measurements indicate significant imbalance in chemical structure."

                st.markdown(f"""
                <div class="result-box" style="border-top-color: {color};">
                    <div style="color:#888; font-weight:700; text-transform:uppercase; letter-spacing:3px; margin-bottom:15px;">Rating Decoded</div>
                    <div class="result-score" style="background: linear-gradient(to bottom, {color}, #444); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{final_score} <span style="font-size:2rem; color:#ccc;">/ 10</span></div>
                    <div class="result-text" style="color:{color};">{label}</div>
                    <p style="color:#666; margin-top:20px; font-size:1.2rem; italic">"{text}"</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 🍇 Trigger Celebration Animation 🍇
                if final_score >= 5:
                    grapes_html = ""
                    for i in range(30):
                        l = np.random.randint(0, 100)
                        d = np.random.uniform(3, 5)
                        del_ = np.random.uniform(0, 2)
                        grapes_html += f"<div class='falling-grape' style='left:{l}vw; animation-duration:{d}s; animation-delay:{del_}s;'>🍇</div>"
                    st.markdown(grapes_html, unsafe_allow_html=True)

            except Exception as ex:
                st.error(f"Prediction Error: {ex}")

# ----------------------------
# Footer
# ----------------------------
st.markdown(f"""
<div style="text-align:center; padding-top:100px; padding-bottom:40px; color:#c0b9a6; font-size:0.9rem; letter-spacing:1px; opacity:0.6;">
    MARCHE EN AVANT | VINTAGE ANALYTICS SYSTEM © 2026
</div>
""", unsafe_allow_html=True)