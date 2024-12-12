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
        "MetaTrader5"
    ]
    
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}

    for package in required_packages:
        if package.lower() not in installed_packages:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        else:
            print(f"{package} is already installed.")
    
    print("All packages are verified")

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

