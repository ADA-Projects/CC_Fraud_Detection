# streamlit_app/app.py
import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# 1. Load artifacts
BASE_DIR = os.path.dirname(__file__)
# Load slim booster and expected features
booster = xgb.Booster()
booster.load_model(os.path.join(BASE_DIR, "fraud_slim.json"))
features = joblib.load(os.path.join(BASE_DIR, "slim_features.joblib"))
# Load category rates and user-friendly names
cat_rates = joblib.load(os.path.join(BASE_DIR, "category_rates.joblib"))
uf_names = joblib.load(os.path.join(BASE_DIR, "uf_names.joblib"))

le_category = joblib.load(os.path.join(BASE_DIR, "le_category.joblib"))

# Build mapping: integer code → friendly label
raw_labels   = list(le_category.classes_)
codes        = list(le_category.transform(raw_labels))
friendly_map = {code: uf_names.get(raw, raw) for code, raw in zip(codes, raw_labels)}

st.title("💳 Fraud Detection Demo (Slim Model)")
st.write("This demo uses a simplified model based on 11 engineered features.\n"
         "Try extreme values to see how the model responds.")

# 2. User inputs
amt       = st.number_input("Transaction Amount", min_value=0.0, value=100.0, step=1.0)
# dropdown of integer codes, but show friendly_map[code]
category_code = st.selectbox(
    "Category",
    options=codes,
    format_func=lambda c: friendly_map[c]
)
st.write(f"Selected category: **{friendly_map[category_code]}**")
hour      = st.slider("Hour (0–23)", 0, 23, 12)
city_pop  = st.number_input("City Population", min_value=1, value=100_000, step=1_000)

# 3. Feature engineering
# Amount transforms
amt_log10      = np.log10(max(amt, 1))
amt_raw_k      = amt / 1_000
# Derived flags
amt_pop_ratio   = amt / city_pop
is_business_hour = int(8 <= hour <= 20)
# Category target encoding
#cat_te         = cat_rates.get(category, cat_rates.mean())
cat_te = cat_rates.get(raw_labels[category_code], cat_rates.mean())
# Hour encodings
hour_sin       = np.sin(2 * np.pi * hour / 24)
hour_cos       = np.cos(2 * np.pi * hour / 24)
is_night       = int(hour >= 22 or hour < 6)
# Population log and code
city_pop_log   = np.log1p(city_pop)
pop_size_code  = pd.cut(
    [city_pop],
    bins=[0, 10_000, 100_000, 1_000_000, np.inf],
    labels=[0, 1, 2, 3]
)[0]
# Interaction features
amt_x_catTE    = amt_log10 * cat_te
amt_x_hour_sin = amt_log10 * hour_sin

# Assemble into a DataFrame with correct column order
data = {
    "amt_log10": amt_log10,
    "amt_raw_k": amt_raw_k,
    "category_te": cat_te,
    "hour_sin": hour_sin,
    "hour_cos": hour_cos,
    "is_night": is_night,
    "city_pop_log": city_pop_log,
    "pop_size_code": pop_size_code,
    "amt_x_catTE": amt_x_catTE,
    "amt_x_hour_sin": amt_x_hour_sin,
    "amt_pop_ratio": amt_pop_ratio,
    "is_business_hour": is_business_hour
}
df = pd.DataFrame([data])[features]

# 4. Predict using slim model
dmat = xgb.DMatrix(df)
prob = booster.predict(dmat)[0]
threshold = 0.70
label = "🚨 Fraud" if prob >= threshold else "✅ Legitimate"

# 5. Display results
# Show both raw probability and percent-formatted for clarity
st.write(f"**Raw model output:** {prob:.4f}")
st.metric("Fraud Probability", f"{prob:.1%}")
st.write(f"**Classification (threshold = {threshold}):** {label}")




# # streamlit_app/app.py
# import os
# import joblib
# import streamlit as st
# import pandas as pd
# import numpy as np
# import xgboost as xgb

# # 1. Load artifacts
# BASE_DIR = os.path.dirname(__file__)
# booster = xgb.Booster()
# booster.load_model(os.path.join(BASE_DIR, "fraud_slim.json"))
# features = joblib.load(os.path.join(BASE_DIR, "slim_features.joblib"))
# cat_rates = joblib.load(os.path.join(BASE_DIR, "category_rates.joblib"))
# uf_names = joblib.load(os.path.join(BASE_DIR, "uf_names.joblib"))
# le_category = joblib.load(os.path.join(BASE_DIR, "le_category.joblib"))

# # Build mapping: integer code → friendly label
# raw_labels   = list(le_category.classes_)
# codes        = list(le_category.transform(raw_labels))
# friendly_map = {code: uf_names.get(raw, raw) for code, raw in zip(codes, raw_labels)}

# st.title("💳 Fraud Detection Demo (Slim Model)")
# st.write("This demo uses a simplified model based on 9 engineered features.\n"
#          "**Note:** Hour is encoded with sine/cosine to capture its circular nature.")

# # Display info about sine/cosine encoding
# st.info("**Hour Encoding:** We transform the hour into two features: `hour_sin = sin(2π·hour/24)` and `hour_cos = cos(2π·hour/24)`."
#         " This captures the fact that hours wrap around (midnight is close to 23:00).")



# # 2. User inputs
# amt = st.number_input("Transaction Amount", min_value=0.0, value=100.0, step=1.0)
# # category = st.selectbox(
# #     "Category",
# #     options=list(cat_rates.index),
# #     format_func=lambda c: uf_names.get(c, c)
# # )

# # dropdown of integer codes, but show friendly_map[code]
# category_code = st.selectbox(
#     "Category",
#     options=codes,
#     format_func=lambda c: friendly_map[c]
# )
# st.write(f"Selected category: **{friendly_map[category_code]}**")
# hour = st.slider("Hour (0–23)", 0, 23, 12)
# city_pop = st.number_input("City Population", min_value=0, value=100_000, step=1_000)

# # 3. Feature engineering
# amt_log = np.log1p(amt)
# default_rate = cat_rates.mean()
# #category_te = cat_rates.get(category, default_rate)
# category_te = cat_rates.get(raw_labels[category_code], cat_rates.mean())
# hour_sin = np.sin(2 * np.pi * hour / 24)
# hour_cos = np.cos(2 * np.pi * hour / 24)
# is_night = int(hour >= 22 or hour < 6)
# city_pop_log = np.log1p(city_pop)
# pop_size_code = pd.cut(
#     [city_pop],
#     bins=[0, 10_000, 100_000, 1_000_000, np.inf],
#     labels=[0, 1, 2, 3]
# )[0]
# amt_x_catTE = amt_log * category_te
# amt_x_hour_sin = amt_log * hour_sin

# data = {
#     "amt_log": amt_log,
#     "category_te": category_te,
#     "hour_sin": hour_sin,
#     "hour_cos": hour_cos,
#     "is_night": is_night,
#     "city_pop_log": city_pop_log,
#     "pop_size_code": pop_size_code,
#     "amt_x_catTE": amt_x_catTE,
#     "amt_x_hour_sin": amt_x_hour_sin
# }
# df = pd.DataFrame([data])[features]

# # 4. Predict using slim model
# dmat = xgb.DMatrix(df)
# prob = booster.predict(dmat)[0]
# threshold = 0.70
# label = "🚨 Fraud" if prob >= threshold else "✅ Legitimate"

# # 5. Display results
# # Show both raw probability and percent-formatted for clarity
# st.write(f"**Raw model output:** {prob:.4f}")
# st.metric("Fraud Probability", f"{prob:.1%}")
# st.write(f"**Classification (threshold = {threshold}):** {label}")




