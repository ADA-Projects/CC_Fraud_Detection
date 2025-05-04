import pandas as pd
def preprocess(df):
    df = df.copy()
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df = pd.get_dummies(df, columns=['category', 'gender', 'state', 'job', 'merchant'], drop_first=True)
    df = df.drop(columns=['trans_date_trans_time', 'first', 'last', 'cc_num'], errors='ignore')
    return df