import sys
import subprocess
import pkg_resources
import os
from myFunctions import install_packages
install_packages()


## importing the packages
import pandas as pd
import numpy as np
import warnings
import joblib
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)
from tabulate import tabulate

### folders 
input_dir = os.path.join('..', 'data', 'preprocessed')
output_dir = os.path.join('..', 'data', 'features')

### Script Functions
# Function to calculate OBV
def calculate_obv(df):
    """
    Calculate the On-Balance Volume (OBV) for each asset and return a DataFrame with columns 'OBV_<asset>'.
    
    OBV is calculated by summing the volume when the price closes higher than the previous close 
    and subtracting it when the price closes lower. If the price is unchanged, OBV remains the same.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing columns for close prices and tick volumes.

    Returns:
    - pd.DataFrame: A DataFrame containing OBV for each asset.
    """
    print("\nCreating OBV indicator...\n")
    assets = {col.split('_')[1] for col in df.columns if col.startswith("close_")}
    for asset in assets:
        close_col = f"close_{asset}"
        volume_col = f"tick_volume_{asset}"
        if close_col in df.columns and volume_col in df.columns:
            obv = [0]
            for i in range(1, len(df)):
                if df[close_col].iloc[i] > df[close_col].iloc[i - 1]:
                    obv.append(obv[-1] + df[volume_col].iloc[i])
                elif df[close_col].iloc[i] < df[close_col].iloc[i - 1]:
                    obv.append(obv[-1] - df[volume_col].iloc[i])
                else:
                    obv.append(obv[-1])
            df[f"OBV_{asset}"] = obv
            print(f"OBV_{asset}")


    return df

# Function to calculate RSI
def calculate_rsi(df, period=14):
    """
    Calculate the Relative Strength Index (RSI) for each asset and return a DataFrame with columns 'RSI_<asset>'.
    
    RSI measures the speed and magnitude of price movements. It is calculated by dividing 
    the average gain by the average loss over a specified period and scaling the result to a 0-100 range.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing columns for close prices.
    - period (int): The lookback period for RSI calculation (default: 14).

    Returns:
    - pd.DataFrame: A DataFrame containing RSI for each asset.
    """
    print("\nCreating RSI indicator\n")
    assets = {col.split('_')[1] for col in df.columns if col.startswith("close_")}
    for asset in assets:
        close_col = f"close_{asset}"
        if close_col in df.columns:
            delta = df[close_col].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()

            rs = avg_gain / avg_loss.replace(0, float('inf'))
            rsi = 100 - (100 / (1 + rs))

            rsi[(avg_gain == 0) & (avg_loss == 0)] = None
            rsi = rsi.fillna(method='ffill')

            df[f"RSI_{asset}"] = rsi
            print(f"RSI_{asset}")
    return df


# Function to calculate ATR
def calculate_atr(df, period=14):
    """
    Calculate the Average True Range (ATR) for each asset and return a DataFrame with columns 'ATR_<asset>'.
    
    ATR measures market volatility by calculating the average of the True Range (TR), 
    which is the greatest of the following:
    - High - Low
    - High - Previous Close
    - Low - Previous Close

    Parameters:
    - df (pd.DataFrame): A DataFrame containing columns for high, low, and close prices.
    - period (int): The lookback period for ATR calculation (default: 14).

    Returns:
    - pd.DataFrame: A DataFrame containing ATR for each asset.
    """

    print("\nCreating ATR indicador...")
    assets = {col.split('_')[1] for col in df.columns if col.startswith("high_")}
    for asset in assets:
        high_col = f"high_{asset}"
        low_col = f"low_{asset}"
        close_col = f"close_{asset}"
        
        if high_col in df.columns and low_col in df.columns and close_col in df.columns:
            high = df[high_col]
            low = df[low_col]
            close = df[close_col].shift(1)
            
            tr = pd.concat([high - low, abs(high - close), abs(low - close)], axis=1).max(axis=1)
            atr = tr.rolling(window=period, min_periods=1).mean()
            df[f"ATR_{asset}"] = atr
            print(f"ATR_{asset}")
    return df

# Function to calculate VWAP
def calculate_vwap(df):
    """
    Calculate the Volume-Weighted Average Price (VWAP) for each asset and return a DataFrame with columns 'VWAP_<asset>'.
    
    VWAP is calculated by dividing the cumulative sum of price * volume by the cumulative sum of volume.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing columns for close prices and tick volumes.

    Returns:
    - pd.DataFrame: A DataFrame containing VWAP for each asset.
    """
    
    print("\nCreating VWAP indicador...\n")
    assets = {col.split('_')[1] for col in df.columns if col.startswith("close_")}
    for asset in assets:
        close_col = f"close_{asset}"
        volume_col = f"tick_volume_{asset}"
        
        if close_col in df.columns and volume_col in df.columns:
            cum_vol = df[volume_col].cumsum()
            cum_price_vol = (df[close_col] * df[volume_col]).cumsum()
            df[f"VWAP_{asset}"] = cum_price_vol / cum_vol
            print(f"VWAP_{asset}")
    return df

# Function to calculate EMA
def calculate_emas(df, periods=[9, 21, 55]):
    """
    Calculate Exponential Moving Averages (EMAs) for each asset and return a DataFrame with columns 'EMA<period>_<asset>'.
    
    EMAs give more weight to recent prices compared to older ones. The function calculates EMAs for 
    multiple periods for each asset.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing columns for close prices.
    - periods (list): A list of periods for EMA calculation (default: [9, 21, 55]).

    Returns:
    - pd.DataFrame: A DataFrame containing EMAs for each asset and period.
    """
    print('\nCreating EMAs\n')
    assets = {col.split('_')[1] for col in df.columns if col.startswith("close_")}
    for asset in assets:
        close_col = f"close_{asset}"
        if close_col in df.columns:
            for period in periods:
                df[f"EMA{period}_{asset}"] = df[close_col].ewm(span=period, adjust=False).mean()
                print(f'EMA{period}_{asset}')
    
    return df

### Reading data from input dir
df = pd.read_parquet(f'{input_dir}\data.parquet')
print('df shape', df.shape)

### Processing features
df = calculate_obv(df)
df = calculate_rsi(df)
df = calculate_atr(df)
df = calculate_vwap(df)
df = calculate_emas(df)

### Cleaning nan inputed by indicators
first_non_null_date = df[df.notna().all(axis=1)].iloc[0]['time']
next_day = first_non_null_date + pd.Timedelta(days=1)
next_day = next_day.replace(hour=9, minute=0, second=0, microsecond=0)  # Linha alterada
next_day_row = df[df['time'].dt.date == next_day.date()].iloc[0]
df = df[df['time'] >= next_day]

### Showing processed data information
logs = [
    ['Processed data shape', df.shape],
    ['First date', next_day],
    ['NaN values found', df[df.isna().any(axis=1)].shape]
]

print(tabulate(logs, headers=['Description', 'Value'], tablefmt='pretty'))

### saving file do output dir
os.makedirs(output_dir, exist_ok=True)
df.to_parquet(f'{output_dir}/features.parquet')

features = df.columns.drop([col for col in df.columns if 'target' in col or 'time' in col])
joblib.dump(features, f'{output_dir}/features.pkl')
print(features)
