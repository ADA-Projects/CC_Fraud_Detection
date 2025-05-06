import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

    return df
