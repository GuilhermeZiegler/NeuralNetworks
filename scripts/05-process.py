import sys
import os

sys.path.append(os.path.abspath('../scripts'))
import myFunctions
from myFunctions import install_packages, save_table 
install_packages()

### required packages
import joblib
import pandas as pd

### folders 
input_target = os.path.join('..', 'data', 'target').replace("/", "\\")
input_features = os.path.join('..', 'data', 'features').replace("/", "\\")
output_dir = os.path.join('..', 'data', 'processed').replace("/", "\\")
output_features =  os.path.join('..', 'data', 'features').replace("/", "\\")


### Script Functions

def pivot_data(df, target=None):
    """
    Pivot a DataFrame by hour for each day, excluding columns containing the target keyword.

    Parameters:
    df (pd.DataFrame): Input DataFrame with a 'time' column containing datetime values.
    target (str, optional): Keyword to exclude columns containing this value from the pivot. Default is None.

    Returns:
    pd.DataFrame: Pivoted DataFrame with columns for each hour of the day and variables suffixed with '_hhmm'.

    Example:
    >>> data = {
    ...     'time': ['2022-06-02 09:00:00', '2022-06-02 09:15:00', '2022-06-02 09:30:00'],
    ...     'open_BGI$': [313.94, 313.17, 312.85],
    ...     'close_BGI$': [313.75, 312.73, 313.00],
    ... }
    >>> df = pd.DataFrame(data)
    >>> df['time'] = pd.to_datetime(df['time'])
    >>> print(pivoted)
    date       open_BGI$_0900  open_BGI$_0915  open_BGI$_0930  close_BGI$_0900  close_BGI$_0915  close_BGI$_0930
    2022-06-02        313.94          313.17          312.85           313.75          312.73          313.00
    """
    df['date'] = pd.to_datetime(df['time']).dt.date
    df['hour'] = pd.to_datetime(df['time']).dt.strftime('%H%M')

    columns_to_pivot = [col for col in df.columns if col not in ['time', 'date', 'hour']]
    pivoted = df.pivot(index='date', columns='hour', values=columns_to_pivot)
    pivoted.columns = [f"{col}_{hour}" for col, hour in pivoted.columns]

    pivoted.reset_index(inplace=True)
    return pivoted


### reading features
df_features = pd.read_parquet(os.path.join(input_features, 'features.parquet'))
df_daily_target = pd.read_parquet(os.path.join(input_target, 'daily_target.parquet'))
df_timestamp_target = pd.read_parquet(os.path.join(input_target, 'timestamp_target.parquet'))
df_daily = df_features.copy()
df_timestamp = df_features.copy()

### merging features and timestamps targets
print(df_timestamp_target.shape)
df_timestamp = df_timestamp.merge(
    df_timestamp_target[['time', 'close_price_target', 'open_price_target', 'behavior_target']],
    on='time',
    how='inner'
)
print(df_timestamp.shape)
df_timestamp.dropna(inplace=True)

### pivoting timestamps to columns in order to train daily model 
df_daily =  pivot_data(df_daily.copy())
df_daily_target.reset_index(inplace=True)


### merging features and daily  targets
df_daily = df_daily.merge(
    df_daily_target[['day', 'close_price_target', 'open_price_target', 'behavior_target']],
    left_on='date', right_on='day', how='inner'
)
df_daily.dropna(inplace=True)
df_daily.drop(columns=['day'], inplace=True)

### Saving data and tables

save_table(df_timestamp.head(6), title = 'Exemplo do Target timestamp para o fechamento, abertura e comportamento do mercado')
save_table(df_daily.head(6), title = 'Exemplo do Target diário para o fechamento, abertura e comportamento do mercado')
os.makedirs(output_dir, exist_ok=True)

daily_name = 'df_daily.parquet'
timestamp_name = 'df_timestamp.parquet'

daily_path = os.path.join(output_dir, daily_name).replace("/", "\\")
timestamp_path = os.path.join(output_dir, timestamp_name).replace("/", "\\")

df_daily.to_parquet(daily_path)
df_timestamp.to_parquet(timestamp_path)

print(f'{daily_name} saved to {daily_path}')
print(f'{timestamp_name} saved to {timestamp_path}')

features = df_daily.columns.drop([col for col in df_daily.columns if 'target' in col or 'day' in col or 'date' in col])
joblib.dump(features, f'{output_features}/daily_features.pkl')
print(f'df_daily_features saved to {output_features}')

features = df_timestamp.columns.drop([col for col in df_timestamp.columns if 'target' in col or 'time' in col])
joblib.dump(features, f'{output_features}/timestamp_15min_features.pkl')
print(f'df_timestamp_features saved to {output_features}')




