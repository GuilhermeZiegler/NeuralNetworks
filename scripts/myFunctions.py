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
        "MetaTrader5",
        "tabulate",
        "optuna",
        "torch",
        "tqdm",
        "shap",
        "kaleido"
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



