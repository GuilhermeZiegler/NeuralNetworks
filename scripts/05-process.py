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
input_target = os.path.join('..', 'data', 'target')
input_features = os.path.join('..', 'data', 'features')
output_dir = os.path.join('..', 'data', 'processed')
output_features =  os.path.join('..', 'data', 'features')


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
D1_df_target = pd.read_parquet(os.path.join(input_target, 'D1_df_target.parquet'))
M15_df_target = pd.read_parquet(os.path.join(input_target, 'M15_df_target.parquet'))
D1_df = df_features.copy()
M15_df = df_features.copy()

### merging features and timestamps targets
print(M15_df_target.shape)
M15_df = M15_df.merge(
    M15_df_target[['time', 'close_price_target', 'open_price_target', 'behavior_target']],
    on='time',
    how='inner'
)
print(M15_df.shape)
M15_df.dropna(inplace=True)

### pivoting timestamps to columns in order to train daily model 
D1_df =  pivot_data(D1_df.copy())
D1_df_target.reset_index(inplace=True)


### merging features and daily  targets
D1_df = D1_df.merge(
    D1_df_target[['day', 'close_price_target', 'open_price_target', 'behavior_target']],
    left_on='date', right_on='day', how='inner'
)
D1_df.dropna(inplace=True)
D1_df.drop(columns=['day'], inplace=True)

### Saving data and tables

save_table(M15_df.head(6), title = 'Exemplo do Target timestamp para o fechamento, abertura e comportamento do mercado')
save_table(D1_df.head(6), title = 'Exemplo do Target diário para o fechamento, abertura e comportamento do mercado')
os.makedirs(output_dir, exist_ok=True)

D1_name = 'D1_df.parquet'
M15_name = 'M15_df.parquet'

D1_path = os.path.join(output_dir, D1_name)
M15_path = os.path.join(output_dir, M15_name)

D1_df.to_parquet(D1_path)
M15_df.to_parquet(M15_path)

print(f'{D1_name} saved to {D1_path}')
print(f'{M15_path} saved to {M15_path}')

features = D1_df.columns.drop([col for col in D1_df.columns if 'target' in col or 'day' in col or 'date' in col])
joblib.dump(features, f'{output_features}/D1_features.pkl')
print(f'D1_df_features saved to {output_features}')

features = M15_df.columns.drop([col for col in M15_df.columns if 'target' in col or 'time' in col])
joblib.dump(features, f'{output_features}/M15_features.pkl')
print(f'M15_df_features saved to {output_features}')




