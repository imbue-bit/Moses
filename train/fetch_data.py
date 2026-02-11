import tushare as ts
import pandas as pd
import numpy as np
import os

TOKEN = 'YOUR_TUSHARE_TOKEN_HERE'
ASSET_CODE = '000300.SH'
START_DATE = '20150101'
END_DATE = '20231231'
SAVE_PATH = 'data/training_data.csv'

def fetch_and_process():
    ts.set_token(TOKEN)
    pro = ts.pro_api()
    df = ts.pro_bar(ts_code=ASSET_CODE, asset='I', start_date=START_DATE, end_date=END_DATE)
    if df is None or df.empty:
        print("Failed to fetch data. Check your Token or Asset Code.")
        return
    df = df.sort_values('trade_date').reset_index(drop=True)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['log_ret'].rolling(window=20).std()
    df['target_return_5d'] = df['close'].shift(-5) / df['close'] - 1.0
    df['target_signal'] = df['target_return_5d'].clip(-0.05, 0.05) / 0.05
    df.dropna(inplace=True)
    
    if not os.path.exists('data'):
        os.makedirs('data')
    
    df.to_csv(SAVE_PATH, index=False)

if __name__ == '__main__':
    fetch_and_process()