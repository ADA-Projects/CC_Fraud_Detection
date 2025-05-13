# streamlit_app/app.py
import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# Load artifacts
BASE_DIR     = os.path.dirname(__file__)
ARTIFACT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../artifacts"))

# Load slim booster
booster = xgb.Booster()
booster.load_model(os.path.join(ARTIFACT_DIR, "fraud_slim.json"))

# Load expected features & mappings
features    = joblib.load(os.path.join(ARTIFACT_DIR, "slim_features.joblib"))
cat_rates   = joblib.load(os.path.join(ARTIFACT_DIR, "category_rates.joblib"))
uf_names    = joblib.load(os.path.join(ARTIFACT_DIR, "uf_names.joblib"))
le_category = joblib.load(os.path.join(ARTIFACT_DIR, "le_category.joblib"))

# build code→friendly map
raw_labels = list(le_category.classes_)
codes      = list(le_category.transform(raw_labels))
friendly   = {code: uf_names.get(lbl,lbl) for code,lbl in zip(codes,raw_labels)}

st.title("💳 Fraud Detection Demo (Slim Model)")
st.write("Enter a few fields; we’ll engineer 12 features and predict fraud.")

# 2. User inputs
amt           = st.number_input("Transaction Amount", min_value=0.0, value=100.0, step=1.0)
# dropdown of integer codes, but show friendly_map[code]
category_code = st.selectbox("Category", options=codes, format_func=lambda c: friendly[c])
st.write(f"Selected category: **{friendly[category_code]}**")
hour          = st.slider("Hour (0–23)", 0, 23, 12)
city_pop      = st.number_input("City Population", min_value=1, value=100_000, step=1_000)

st.write(f"Selected category: **{friendly[category_code]}**")

# 3. Manual feature engineering – names must exactly match `features`
amt_log         = np.log10(max(amt,1))            # your notebook used log10
amt_raw_k         = amt/1_000
amt_pop_ratio_log = np.log1p(amt/city_pop)
is_business_hour  = int(8 <= hour <= 20)
cat_te            = cat_rates.get(raw_labels[category_code], cat_rates.mean())
hour_sin          = np.sin(2*np.pi*hour/24)
hour_cos          = np.cos(2*np.pi*hour/24)
is_night          = int(hour>=22 or hour<6)
city_pop_log      = np.log1p(city_pop)
pop_size_code     = pd.cut(
    [city_pop],
    bins=[0, 10_000, 100_000, 1_000_000, np.inf],
    labels=[0,1,2,3]
)[0]
amt_x_catTE       = amt_log * cat_te
amt_x_hour_sin    = amt_log * hour_sin

data = {
    "amt_log":        amt_log,
    "amt_raw_k":        amt_raw_k,
    "category_te":      cat_te,
    "hour_sin":         hour_sin,
    "hour_cos":         hour_cos,
    "is_night":         is_night,
    "city_pop_log":     city_pop_log,
    "pop_size_code":    pop_size_code,
    "amt_x_catTE":      amt_x_catTE,
    "amt_x_hour_sin":   amt_x_hour_sin,
    "amt_pop_ratio_log": amt_pop_ratio_log,
    "is_business_hour": is_business_hour
}

# 4. Build DataFrame in exactly the right order
df = pd.DataFrame([data])[features]

# 5. Predict
dmat = xgb.DMatrix(df)
prob = booster.predict(dmat)[0]
threshold = st.slider("Decision threshold", 0.0, 1.0, 0.447, step=0.01)
label     = "🚨 Fraud" if prob >= threshold else "✅ Legitimate"

# 6. Display
st.write(f"**Raw model output:** {prob:.4f}")
st.metric("Fraud Probability", f"{prob:.1%}")
st.write(f"**Classification (threshold = {threshold}):** {label}")

