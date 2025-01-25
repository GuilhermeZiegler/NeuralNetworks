import sys
import subprocess
import pkg_resources
import os
from myFunctions import install_packages, save_table
install_packages()
## importing the packages
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)

### Seting folders
input_dir = os.path.join('..', 'data', 'features')
output_dir = os.path.join('..', 'data', 'target')

### Script Functions
def generate_targets(df, asset, timeframe=None):
    """
    Generate targets based on the closing and opening prices of the specified asset.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing asset data.
    asset (str): Name of the asset for which to calculate the targets.
    timeframe (str): Timeframe for grouping data. If 'day', group by day; 
                     otherwise, calculate targets for all timestamps.
    
    Returns:
    pd.DataFrame: DataFrame with opening and closing prices, and targets.

    Example:
    With `timeframe='day'`:
    | time                | open_BGI$ | close_BGI$ |
    |---------------------|-----------|------------|
    | 2024-12-01 09:00:00 | 323.57    | 322.07     |
    | 2024-12-01 17:45:00 | 313.94    | 288.87     |
    | 2024-12-02 09:00:00 | 318.39    | 287.36     |

    Returns:
    | day        | open_BGI$ | close_BGI$ | close_price_target | open_price_target | behavior_target |
    |------------|-----------|------------|--------------------|-------------------|-----------------|
    | 2024-12-01 | 323.57    | 288.87     | 287.36             | 318.39            | 0               |
    | 2024-12-02 | 318.39    | 287.36     | ...                | ...               | ...             |
    
    With `timeframe=None`:
    | time                | open_BGI$ | close_BGI$ |
    |---------------------|-----------|------------|
    | 2024-12-01 09:00:00 | 323.57    | 322.07     |
    | 2024-12-01 17:45:00 | 313.94    | 288.87     |
    | 2024-12-02 09:00:00 | 318.39    | 287.36     |

    Returns:
    | time                | open_BGI$ | close_BGI$ | close_price_target | open_price_target | behavior_target |
    |---------------------|-----------|------------|--------------------|-------------------|-----------------|
    | 2024-12-01 09:00:00 | 323.57    | 322.07     | 288.87             | 313.94            | 0               |
    | 2024-12-01 17:45:00 | 313.94    | 288.87     | 287.36             | 318.39            | 0               |
    """
    close_col = f'close_{asset}'
    open_col = f'open_{asset}'
    
    if close_col not in df.columns or open_col not in df.columns:
        raise KeyError(f'Columns for {asset} ({open_col}, {close_col}) not found in the DataFrame')

    if timeframe == 'day':
        df['day'] = pd.to_datetime(df['time']).dt.date
        aux_open = df.groupby('day').first()[[open_col]]
        aux_close = df.groupby('day').last()[[close_col]]
        target_df = pd.concat([aux_open, aux_close], axis=1)
        target_df['close_price_target'] = target_df[close_col].shift(-1)
        target_df['open_price_target'] = target_df[open_col].shift(-1)
        target_df['behavior_target'] = (target_df['close_price_target'] > target_df[close_col]).astype(int)
    else:
        target_df = df[['time', open_col, close_col]].copy()
        target_df['close_price_target'] = target_df[close_col].shift(-1)
        target_df['open_price_target'] = target_df[open_col].shift(-1)
        target_df['behavior_target'] = (target_df['close_price_target'] > target_df[close_col]).astype(int)

    return target_df

### reading features
df = pd.read_parquet(f'{input_dir}/features.parquet')
print('df shape: ', df.shape)

### Generating Targets for timestamp model and daily models
D1_df_target = generate_targets(df, asset='BGI$', timeframe='day')
M15_df_target = generate_targets(df, asset='BGI$')

### Saving data and tables
os.makedirs(output_dir, exist_ok=True)
M15_df_target.to_parquet(f'{output_dir}/target_M15_df.parquet')
D1_df_target.to_parquet(f'{output_dir}/D1_df_target.parquet')

save_table(D1_df_target.head(6), title = 'Exemplo do Target D1 para o fechamento, abertura e comportamento do mercado')
save_table(M15_df_target.head(6), title = 'Exemplo do Target M15 para o fechamento, abertura e comportamento do mercado')