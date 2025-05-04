print("Using process.py from scripts")

import pandas as pd

def preprocess(df):
    df = df.copy()

    # Drop irrelevant or high-cardinality identifier columns
    df = df.drop(columns=[
        'Unnamed: 0', 'cc_num', 'first', 'last', 'trans_num',
        'street', 'dob', 'unix_time'
    ], errors='ignore')

    # Convert datetime and extract useful features
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['month'] = df['trans_date_trans_time'].dt.month
    df = df.drop(columns=['trans_date_trans_time'])

    # Categorical encoding
    categorical_cols = [
        'merchant', 'category', 'gender', 'city', 'state', 'job'
    ]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df
