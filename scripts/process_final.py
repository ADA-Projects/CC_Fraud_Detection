import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def polprocess(df, cat_rates):
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
    df["category_te"].fillna(cat_rates.mean(), inplace=True)
    df['hour_sin']      = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos']      = np.cos(2*np.pi*df['hour']/24)
    df['is_night']      = ((df['hour']>=22)|(df['hour']<6)).astype(int)
    df['city_pop_log']  = np.log1p(df['city_pop'])
    df['amt_x_catTE']   = df['amt_log'] * df['category_te']
    df['amt_x_hour_sin']= df['amt_log'] * df['hour_sin']
    return df