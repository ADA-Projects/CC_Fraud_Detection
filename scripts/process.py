import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def preprocess(df):
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

    return df
