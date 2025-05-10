import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def preprocess(df, cat_rates):
    df = df.copy()
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

    # Convert to datetime
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek

    # Drop high-cardinality or irrelevant columns
    df = df.drop(columns=[
        'trans_date_trans_time', 'cc_num', 'first', 'last', 'street',
        'zip', 'dob', 'trans_num', 'unix_time'
    ])

    # Label encode categorical variables
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # New: amt_log10 & amt_raw_k
    df["amt_log10"] = np.log10(df["amt"].clip(lower=1))
    df["amt_raw_k"] = df["amt"] / 1_000

    # New: amt_pop_ratio & is_business_hour (if you added them)
    df["amt_pop_ratio"]   = df["amt"] / df["city_pop"]
    df["is_business_hour"]= df["hour"].between(8,20).astype(int)

    # 1) Log10 + raw‐k
    df["amt_log10"]  = np.log10(df["amt"].clip(lower=1))
    df["amt_raw_k"]  = df["amt"] / 1_000

    # 2) Target‐encode category (assuming you have cat_rates in scope)
    df["category_te"] = df["category"].map(cat_rates)  # or however you did it

    # 3) Re-run your cyclical and night flags
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["is_night"]   = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(int)

    # 4) City‐pop log
    df["city_pop_log"] = np.log1p(df["city_pop"])

    # 5) amt × category_te, amt × hour_sin
    df["amt_x_catTE"]    = np.log1p(df["amt"]) * df["category_te"]
    df["amt_x_hour_sin"] = np.log1p(df["amt"]) * df["hour_sin"]

    # 6) pop_size_code (assuming you have defined those bins earlier)
    df["pop_size_code"] = pd.cut(
        df["city_pop"],
        bins=[0, 10_000, 100_000, 1_000_000, np.inf],
        labels=[0,1,2,3]
    ).cat.codes

    return df
