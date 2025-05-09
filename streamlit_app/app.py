# streamlit_app/app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load artifacts
# BASE_DIR   = os.path.dirname(__file__)
# model      = joblib.load(os.path.join(BASE_DIR, "final_model.joblib"))
# cat_rates  = joblib.load(os.path.join(BASE_DIR, "category_rates.joblib"))

model     = joblib.load("streamlit_app/final_model.joblib")
cat_rates = joblib.load("streamlit_app/category_rates.joblib")

st.title("💳 Fraud Detection Demo")
st.write("Enter transaction details to see fraud probability and classification.")

# 2. User inputs
amt      = st.number_input("Transaction Amount", min_value=0.0, value=100.0, step=1.0)
category = st.selectbox("Category", options=list(cat_rates.keys()))
hour     = st.slider("Hour (0–23)", 0, 23, 12)
city_pop = st.number_input("City Population", min_value=0, value=100_000, step=1_000)

# 3. Feature engineering
amt_log       = np.log1p(amt)
default_rate = cat_rates.mean()
category_te  = cat_rates.get(category, default_rate)
hour_sin      = np.sin(2 * np.pi * hour / 24)
hour_cos      = np.cos(2 * np.pi * hour / 24)
is_night      = int(hour >= 22 or hour < 6)
city_pop_log  = np.log1p(city_pop)
pop_size_code = pd.cut(
    [city_pop],
    bins=[0, 10_000, 100_000, 1_000_000, np.inf],
    labels=[0, 1, 2, 3]
)[0]

amt_x_catTE    = amt_log * category_te
amt_x_hour_sin = amt_log * hour_sin

df = pd.DataFrame([{
    "amt_log": amt_log,
    "category_te": category_te,
    "hour_sin": hour_sin,
    "hour_cos": hour_cos,
    "is_night": is_night,
    "city_pop_log": city_pop_log,
    "pop_size_code": pop_size_code,
    "amt_x_catTE": amt_x_catTE,
    "amt_x_hour_sin": amt_x_hour_sin
}])

# 4. Prediction & display
prob      = model.predict_proba(df)[0,1]
threshold = 0.70
label     = "🚨 Fraud" if prob >= threshold else "✅ Legitimate"

st.metric("Fraud Probability", f"{prob:.2%}")
st.write(f"**Classification (threshold = {threshold}):** {label}")
