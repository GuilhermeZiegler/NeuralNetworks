import sys
import subprocess
import pkg_resources
import os
import pandas as pd

def install_packages():
    required_packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib",
        "pyarrow",
        "fastparquet",
        "plotly",
        "matplotlib",
        "seaborn",
        "MetaTrader5",
        "tabulate",
        "optuna",
        "torch",
        "tqdm",
        "shap",
        "kaleido",
        "statsmodels", 
        "re"
    ]
    
    print(f'Installing required packages: {required_packages}')
    # Checking installed packages
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}

    # Install missing packages
    for package in required_packages:
        try:
            if package.lower() not in installed_packages:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            else:
                print(f"{package} is already installed.")
        except Exception as e:
            print(f"Error installing {package}: {e}")
            continue  # Continue with other packages, log the error but don't stop the process
    
    print("All packages are verified.")

    
from statsmodels.tsa.stattools import adfuller


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
        table_dir (str): Path where the file will be saved.
    """
    os.makedirs(os.path.join(table_dir, 'csv'), exist_ok=True)
    
    csv_path = os.path.join(table_dir, 'csv')

    existing_files = [f for f in os.listdir(csv_path) if title in f and f.endswith('.csv')]
    
    if existing_files:
        file_name_csv = existing_files[0]
        csv_output_path = os.path.join(csv_path, file_name_csv).replace("/", "\\")
    else:
        num = len([f for f in os.listdir(csv_path) if f.startswith('Tabela_')])
        num += 1
        file_name_csv = f"Tabela_{num}_{title}.csv"
        csv_output_path = os.path.join(csv_path, file_name_csv).replace("/", "\\")

    df.to_csv(csv_output_path, index=False)
    print(f"Table saved as CSV: {csv_output_path}")


def make_stationary(df: pd.DataFrame(), 
                    max_diffs:int):
    """
    Make the given dataframe stationary by applying differencing.

    Args:
        df (pd.DataFrame): DataFrame containing time series data.
        N_MAX (int): Maximum number of differencing iterations to perform.

    Returns:
        pd.DataFrame: DataFrame with columns showing the stationarity status after each differencing iteration.
    """

    diff_df = df.copy()
    result_df = pd.DataFrame(index=diff_df.columns)
    stationary_set = set()

    for i in range(max_diffs):
        if i != 0:
            diff_df = diff_df.diff()
            diff_df.dropna(inplace=True)

        _, stationary_cols, non_stationary_cols = adfuller_test(diff_df)
        stationary_set.update(stationary_cols)

        for col in diff_df.columns:
            if col in stationary_set:
                if f'stationary_{i}' not in result_df.columns:
                    result_df[f'stationary_{i}'] = ''
                if result_df.loc[col, f'stationary_{i}'] != 'S':
                    result_df.loc[col, f'stationary_{i}'] = 'S'

        if len(non_stationary_cols) == 0:
            print(f"All variables became stationary after {i} differences.")
            break
        elif i == max_diffs - 1:
            print(f"Maximum iterations ({max_diffs}) reached, some variables are still non-stationary.")
            
    print(result_df)

def adfuller_test(df, critical_level='5%'):
    # Check if the critical level is valid
    if critical_level not in ['1%', '5%', '10%']:
        raise ValueError("The critical_level parameter must be one of: '1%', '5%', or '10%'")

    non_stationary = []  # List to store non-stationary variables
    result_df = pd.DataFrame(columns=['Variable', 'ADF Statistic', 'p-value', '1%', '5%', '10%', 'res'])

    # Loop through each column (variable)
    for col in df.columns:
        result = adfuller(df[col].values)
        adf_stat = result[0]
        p_value = result[1]
        critical_vals = result[4]

        # Compare the ADF statistic with the selected critical value
        is_stationary = 1 if adf_stat < critical_vals[critical_level] else 0

        # Append to the non_stationary list if not stationary
        if is_stationary == 0:
            non_stationary.append(col)

        # Collect results for each variable
        row = {
            'Variable': col,
            'ADF Statistic': adf_stat,
            'p-value': p_value,
            '1%': critical_vals['1%'],
            '5%': critical_vals['5%'],
            '10%': critical_vals['10%'],
            'res': is_stationary  # 1 for Stationary, 0 for Non-Stationary
        }

        # Append row to result_df
        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

    # Return non_stationary variables and result_df
    return non_stationary, result_df

def stationary_window_adf_multi_columns(df, window_size, method, offset_type='M'):
    results = {}
    max_index = df.index.max()
    min_index = df.index.min()
    offset_mapping = {
        'D': 'days', 'BD': 'weekday', 'W': 'weeks', 'M': 'months', 'Y': 'years',
        'MIN': 'minutes', 'H': 'hours', 'S': 'seconds', 'MS': 'microseconds',
        'MS_START': 'months=window_size, day=1', 'MS_END': 'months=window_size, day=1'
    }
    
    if offset_type not in offset_mapping:
        supported_options = ', '.join(offset_mapping.keys())
        raise ValueError(f"The offset_type parameter must be one of the supported options: {supported_options}")
    else:
        if offset_type in ['MS_START', 'MS_END']:
            offset_value = offset_mapping[offset_type].replace('window_size', str(window_size))
            offset = pd.DateOffset(eval(offset_value))
        else:
            offset = pd.DateOffset(**{offset_mapping[offset_type]: window_size})
    
    for column in df.columns:
        df_column = df[column]
        start_date = min_index
        end_date = start_date + offset
        column_results = []
        
        if method == 'constant':
            while end_date <= max_index:
                df_slice = df_column.loc[start_date:end_date]
                result = adfuller(df_slice)
                p_value = result[1]
                is_stationary = p_value <= 0.05
                column_results.append((start_date, end_date, p_value, is_stationary, "constant"))
                start_date = end_date
                end_date += offset
        elif method == 'forward':
            while end_date <= max_index:
                df_slice = df_column.loc[start_date:end_date]
                result = adfuller(df_slice)
                p_value = result[1]
                is_stationary = p_value <= 0.05
                column_results.append((start_date, end_date, p_value, is_stationary, method))
                end_date += offset
        elif method == 'back':
            end_date = max_index
            start_date = end_date - offset
            while start_date >= min_index:
                df_slice = df_column.loc[start_date:end_date]
                result = adfuller(df_slice)
                p_value = result[1]
                is_stationary = p_value <= 0.05
                column_results.append((start_date, end_date, p_value, is_stationary, method))
                start_date -= offset

        results[column] = pd.DataFrame(column_results, columns=['Start Date', 'End Date', 'p-value', 'Is Stationary', 'Method'])

    return results

def johansen_cointegration_test(df, y_var, coint_vars, det_order=-1, k_ar_diff=0, critical_level="5%"):
    col = {'1%': 0, '5%': 1, '10%': 2}.get(critical_level)
    if col is None:
        raise ValueError("Critical level must be '1%', '5%' or '10%'.")

    result = {}
    start_date = df.index.min()
    end_date = df.index.max()
    y_variable = df[y_var]
    other_vars = df[coint_vars]
    cointegrated_vars = ', '.join(coint_vars)
    series_list = {"Interest Variable": y_variable}

    for var in coint_vars:
        series_list[var] = other_vars[var]

    data = pd.DataFrame(series_list)
    johansen_result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
    eigenvalues = johansen_result.eig
    trace_stats = johansen_result.lr1
    max_eigen_stats = johansen_result.lr2
    trace_critical_vals = johansen_result.cvt
    max_eigen_critical_vals = johansen_result.cvm

    trace_cointegration = max_eigen_cointegration = ranking_trace = ranking_eigenvalue = 0

    for i in range(len(trace_stats)):
        if trace_stats[i] >= trace_critical_vals[i][col]:
            trace_cointegration = 1
            ranking_trace += 1
        if max_eigen_stats[i] >= max_eigen_critical_vals[i][col]:
            max_eigen_cointegration = 1
            ranking_eigenvalue += 1

    result = {
        "Start Date": start_date,
        "End Date": end_date,
        "Interest Variable": y_variable.name,
        "Cointegrated Variables": cointegrated_vars,
        "Cointegration (Trace)": trace_cointegration,
        "Ranking (Trace)": ranking_trace,
        "Cointegration (Max Eigenvalue)": max_eigen_cointegration,
        "Ranking (Max Eigenvalue)": ranking_eigenvalue,
        "Eigenvalues": eigenvalues
    }

    result_df = pd.DataFrame([result])
    return result_df


