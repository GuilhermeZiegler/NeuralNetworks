import sys
import subprocess
import pkg_resources

def install_packages():
    required_packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib",
        "pyarrow",
        "fastparquet",
        "plotly",
        "matplotlib"
    ]
    
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}

    for package in required_packages:
        if package.lower() not in installed_packages:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        else:
            print(f"{package} is already installed.")
    
    print("All packages are verified")

install_packages()
import pandas as pd
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

input_file = '..//data//assets//'
output_file = '..//data//preprocess//'
table_dir='..//results//tables//'

def save_table(df: pd.DataFrame, 
               title: str = 'Table', 
               table_dir: str = '../results/tables/'):
    """
    Saves the DataFrame as a CSV file. If a file with the exact title exists,
    it will overwrite the existing file. Otherwise, it will create a new file
    with the next available number.

    Args:
        df (pd.DataFrame): DataFrame to be saved.
        title (str): Title to be used in the CSV filename.
        output_path (str): Path where the file will be saved.
    """
    os.makedirs(os.path.join(table_dir, 'csv'), exist_ok=True)
    csv_path = os.path.join(table_dir, 'csv')
    existing_files = [f for f in os.listdir(csv_path) if title in f and f.endswith('.csv')]
    if existing_files:
        file_name_csv = existing_files[0]
        csv_output_path = os.path.join(csv_path, file_name_csv)
    else:
        num = len([f for f in os.listdir(csv_path) if f.startswith('Tabela_')])
        num += 1
        file_name_csv = f"Tabela_{num}_{title}.csv"
        csv_output_path = os.path.join(csv_path, file_name_csv)

    df.to_csv(csv_output_path, index=False)
    print(f"Tabela saved as CSV: {csv_output_path}")
    
def check_data(input_dir: str):
    """
    Loads all .parquet files from the specified directory, processes them by:
    - Extracting the ticker symbol from the filename (everything before the first '_').
    - Creating a 'date' column with unique dates and dropping duplicates.
    - Storing the ticker, first date, last date, and the number of columns in the DataFrame for each processed file.

    Args:
        directory_path (str): The path to the directory containing the .parquet files.

    Returns:
        pd.DataFrame: A DataFrame containing the summary information (ticker, first date, last date, and shape).
    """
    parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]
    summary_data = [] 
    for parquet_file in parquet_files:
        ticker = parquet_file.split('_')[0] 
        file_path = os.path.join(input_dir, parquet_file)
        df = pd.read_parquet(file_path)
        df['date'] = df['time'].dt.strftime('%Y-%m-%d')
        df['time_of_trade'] = df['time'].dt.strftime('%H:%M:%S')
        first_date = df['date'].min()
        last_date = df['date'].max()
        first_trade =  df['time_of_trade'].min()
        last_trade = df['time_of_trade'].max()
        summary_data.append({
            'ticker': ticker,
            'first_date': first_date,
            'last_date': last_date,
            'fist_trade': first_trade,
            'last_trade': last_trade,
            'rows': df.shape[0],
            'columns': df.shape[1],
            'unique_dates': len(df.date.unique())
        })
    
    data_info_df = pd.DataFrame(summary_data)
    save_table(data_info_df, title='Visualizaçao das séries de dados escolhidas')
    return data_info_df

def elegant_inputer(df: pd.DataFrame, 
                    start_date: str = '2022-06-01 09:00:00', 
                    end_date: str = '2024-11-22 17:45:00',
                    timeframe: str = '15T', 
                    lookback: int = 180):
    """
    Generates missing time intervals for the provided date range and timeframe, 
    adds them to the original dataframe, and fills missing values with forward fill method.
    
    Parameters:
    df (DataFrame): Original dataframe containing the time and other columns.
    start_date (str): Start date and time of the range (default '2022-06-01 09:00:00').
    end_date (str): End date and time of the range (default '2024-11-22 17:45:00').
    timeframe (str): Time interval for generating timestamps (default '15T' for 15 minutes).
    lookback (int): The number of previous valid entries to use for filling missing data (default 5).
    
    Returns:
    DataFrame: A dataframe with the missing time intervals added and missing values filled.
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    time_intervals = pd.date_range(start=start_date, end=end_date, freq=timeframe)
    df_aux = pd.DataFrame(time_intervals, columns=['time'])
    df_aux['date'] = df_aux['time'].dt.date
    df['date'] = df['time'].dt.date
    valid_dates = df['date'].unique()
    df_aux = df_aux[df_aux['date'].isin(valid_dates)]
    df_inputed = pd.merge(df_aux, df, on=['date', 'time'], how='left')
    df_inputed['tick_volume'].fillna(0, inplace=True)
    df_inputed['real_volume'].fillna(0, inplace=True)
    df_inputed = df_inputed.sort_values(by='time').reset_index(drop=True)
    cols_to_ffill = ['open','high','low','close','spread']
    df_inputed[cols_to_ffill] = df_inputed[cols_to_ffill].fillna(method='ffill', limit=lookback)
    first_trade = start_date.time()
    last_trade = end_date.time()
    df_inputed = df_inputed[(df_inputed['time'].dt.time >= first_trade) & (df_inputed['time'].dt.time <= last_trade)]

    return df_inputed

def process_data(input_dir: str, 
                 output_dir: str):
    """
    Processes all .parquet files in the specified directory, concatenating DataFrames by aligning on the 'time' column
    and appending the ticker as a suffix to each column name.

    Args:
        input_dir (str): Path to the directory containing .parquet files.
        output_dir (str): Path to save the processed DataFrame.
    """
    parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]
    processed_df = pd.DataFrame()
    for parquet_file in parquet_files:
        ticker = parquet_file.split('_')[0]
        file_path = os.path.join(input_dir, parquet_file)
        df = pd.read_parquet(file_path)
        df = elegant_inputer(df)
        df.drop(columns='date', inplace=True)
        df = df.rename(columns=lambda col: f"{col}_{ticker}" if col != "time" else col)
        processed_df = pd.concat([processed_df, df], axis=1)

    processed_df = processed_df.loc[:, ~processed_df.columns.duplicated()]
    processed_df.to_parquet(f'{output_dir}/data.parquet')
    return processed_df

