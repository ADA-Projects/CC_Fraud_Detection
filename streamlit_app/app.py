# streamlit_app/app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load artifacts
BASE_DIR = os.path.dirname(__file__)
model_path     = os.path.join(BASE_DIR, "final_model.joblib")
cat_rates_path = os.path.join(BASE_DIR, "category_rates.joblib")
uf_names_path  = os.path.join(BASE_DIR, "uf_names.joblib")

model     = joblib.load(model_path)
cat_rates = joblib.load(cat_rates_path)
uf_names  = joblib.load(uf_names_path)
st.write("uf_names keys:", uf_names.keys())


st.title("💳 Fraud Detection Demo")
st.write("Enter transaction details to see fraud probability and classification.")

# 2. User inputs
amt = st.number_input(
    "Transaction Amount", min_value=0.0, value=100.0, step=1.0
)
# Show friendly labels in dropdown, but keep code as underlying value
category = st.selectbox(
    "Category",
    options=list(cat_rates.index),
    format_func=lambda c: uf_names.get(c, c)
)
# Show interpreted friendly name
st.write(f"Selected category: **{uf_names.get(category, category)}**")

hour     = st.slider("Hour (0–23)", 0, 23, 12)
city_pop = st.number_input(
    "City Population", min_value=0, value=100_000, step=1_000
)

# 3. Feature engineering
amt_log       = np.log1p(amt)
default_rate  = cat_rates.mean()
category_te   = cat_rates.get(category, default_rate)
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

df = pd.DataFrame([{  # assemble features into DataFrame
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

# 4. Ensure all model-expected features are present
expected_feats = model.get_booster().feature_names
for feat in expected_feats:
    if feat not in df.columns:
        df[feat] = 0
# Reorder columns to match training
df = df[expected_feats]

# 5. Predict
prob      = model.predict_proba(df)[0, 1]
threshold = 0.70
label     = "🚨 Fraud" if prob >= threshold else "✅ Legitimate"

st.metric("Fraud Probability", f"{prob:.2%}")
st.write(f"**Classification (threshold = {threshold}):** {label}")


