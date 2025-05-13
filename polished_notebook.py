# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 📒 Credit Card Fraud Detection
# A polished end-to-end notebook: data load, preprocessing, training, interpretation, and export.

# ## 1. Setup & Data Load

import os
import shutil
import pandas as pd

# Move kaggle credential into place (if using Kaggle API)
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
shutil.copy("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

# Read train/test CSVs
base_path = "/home/codespace/.cache/kagglehub/datasets/kartik2112/fraud-detection/versions/1"
df_train = pd.read_csv(f"{base_path}/fraudTrain.csv")
df_test  = pd.read_csv(f"{base_path}/fraudTest.csv")

print("Train shape:", df_train.shape)
print("Test shape:",  df_test.shape)

df_train.head()

df_test.head()

# ## 2. Preprocessing & Feature Engineering

from sklearn.preprocessing import LabelEncoder
import numpy as np

# 2.1: Define preprocess() with all engineered features
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def preprocess(df, cat_rates):
    df = df.copy()
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df = df.drop(columns=[
        'trans_date_trans_time','cc_num','first','last',
        'street','zip','dob','trans_num','unix_time'
    ], errors='ignore')
    # Label-encode raw categoricals
    for col in df.select_dtypes(include='object'):
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    # Numeric & interaction features
    df['amt_log']       = np.log1p(df['amt'])
    df['amt_bin_code']  = df['amt_bin'].cat.codes if 'amt_bin' in df else 0
    df['pop_size_code'] = pd.cut(
        df['city_pop'], bins=[0,1e4,1e5,1e6,np.inf], labels=[0,1,2,3]
    ).cat.codes
    df['category_te']   = df['category'].map(cat_rates)
    df['hour_sin']      = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos']      = np.cos(2*np.pi*df['hour']/24)
    df['is_night']      = ((df['hour']>=22)|(df['hour']<6)).astype(int)
    df['city_pop_log']  = np.log1p(df['city_pop'])
    df['amt_x_catTE']   = df['amt_log'] * df['category_te']
    df['amt_x_hour_sin']= df['amt_log'] * df['hour_sin']
    return df

# 2.2 Compute category→fraud rate mapping
cat_rates = df_train.groupby('category')['is_fraud'].mean()

# 2.3 Apply to train/test
train_prep = preprocess(df_train, cat_rates)
test_prep  = preprocess(df_test,  cat_rates)

# 2.4 Split into X/y
X_train = train_prep.drop(columns=['is_fraud'])
y_train = train_prep['is_fraud']
X_test  = test_prep.drop(columns=['is_fraud'])
y_test  = test_prep['is_fraud']

print("Example engineered row:")
X_train.head(1).T

# ## 3. Hyperparameter Tuning & Final Training

from xgboost import train, DMatrix
from sklearn.metrics import confusion_matrix, classification_report

# 3.1 Best hyperparameters (from prior search)
#   subsample=1.0, n_estimators=200, max_depth=12,
#   learning_rate=0.15, colsample_bytree=0.6

params = {
    'objective':'binary:logistic',
    'eval_metric':'logloss',
    'max_depth':12,
    'learning_rate':0.15,
    'subsample':1.0,
    'colsample_bytree':0.6,
    'scale_pos_weight': float((y_train==0).sum()/(y_train==1).sum()),
    'base_score':float(y_train.mean())
}

# 3.2 Prepare DMatrices
features = [
    'amt_log','category_te','hour_sin','hour_cos','is_night',
    'city_pop_log','pop_size_code','amt_x_catTE','amt_x_hour_sin'
]
dtr = DMatrix(X_train[features], label=y_train)
dvl = DMatrix(X_test[features],  label=y_test)

# 3.3 Train with early stopping
bst = train(
    params, dtr, num_boost_round=500,
    evals=[(dtr,'train'),(dvl,'eval')],
    early_stopping_rounds=10, verbose_eval=False
)

# 3.4 Evaluate
preds = (bst.predict(DMatrix(X_test[features])) >= 0.70).astype(int)
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))
print("\nClassification Report:")
print(classification_report(y_test, preds, digits=4))

# ## 4. Model Interpretation (SHAP)

import shap

# 4.1 Load explainer & SHAP values
explainer = shap.Explainer(bst)
shap_vals  = explainer(X_test[features])

# 4.2 Summary plot
shap.summary_plot(shap_vals, X_test[features])

# ## 5. Deploy & Export Artifacts

import joblib
import xgboost as xgb

# 5.1 Save Booster
bst.save_model("streamlit_app/fraud_slim.json")
# 5.2 Save feature list & mappings
joblib.dump(features,    "streamlit_app/slim_features.joblib")
joblib.dump(cat_rates,   "streamlit_app/category_rates.joblib")
joblib.dump(le_category,"streamlit_app/le_category.joblib")
joblib.dump(uf_names,    "streamlit_app/uf_names.joblib")

# 5.3 Snippet: wiring into Streamlit
print("# In your app.py:\n" + \
      "# booster = xgb.Booster(); booster.load_model('fraud_slim.json')\n" + \
      "# features = joblib.load('slim_features.joblib')\n" + \
      "# … build df = pd.DataFrame([...])[features]\n" + \
      "# prob = booster.predict(xgb.DMatrix(df))[0]\n" + \
      "# st.metric('Fraud Probability', f'{prob:.1%}')")
