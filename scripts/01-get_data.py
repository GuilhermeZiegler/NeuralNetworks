# For this script to work, MetaTrader5 must be logged in on the machine.
# Currently, Clear has a free version. Log in to your real account.
# Configure the script with login, password, and server information!
# Required installations
# login variables


login = int(input('YourAccountNumber: '))   # must be a integer
server = input('YourSever: ') # a string like 'ClearInvestimentos-CLEAR')
password = input('YouPassword: ')

import sys
import subprocess
import os

from myFunctions import install_packages
install_packages()

## importing the packages
import pandas as pd
from datetime import datetime
import MetaTrader5 as mt5
from datetime import datetime

## script functions
def connect_to_mt5(login: int = None,
                   server: str = "YourServer", 
                   password: str = "YourPassword") -> bool:
    """
    Connects to MetaTrader 5 using the provided login, server, and password.
    
    Parameters:
    - login: MetaTrader login (default: your provided login).
    - server: MetaTrader server (default: your provided server).
    - password: MetaTrader password (default: your provided password).
    """
    if not mt5.initialize(login=login, password=password, server=server):
        print("Failed to connect to MetaTrader 5")
    else:
        print("Successfully connected to MetaTrader 5")
  
def download_and_save_from_mt5(symbols: list = None,
                               timeframe = mt5.TIMEFRAME_M15,
                               start_date: datetime = None,
                               end_date: datetime = datetime.now(),
                               save_dir: str = None):
                            
    """
    Downloads historical data for multiple assets from MetaTrader 5 and saves the data as CSV and Parquet files.
    
    Parameters:
    - symbols: List of asset symbols to download.
    - timeframe: Timeframe for the data (e.g., M15, M1, H1, etc.).
    - start_date: Start date for historical data.
    - end_date: End date for historical data.
    - save_dir: Directory to save the CSV and Parquet files.
    - login: MetaTrader login.
    - server: MetaTrader server.
    - password: MetaTrader password.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    for symbol in symbols:
        try:
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
            if rates is None or len(rates) == 0:
                print(f"No data found for asset: {symbol}")
                continue

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            first_date = df['time'].iloc[0].strftime('%Y%m%d')
            filename_csv = os.path.join(save_dir, f"{symbol}_data_{first_date}.csv").replace("/", "\\")
            filename_parquet = os.path.join(save_dir, f"{symbol}_data_{first_date}.parquet").replace("/", "\\")
            df.to_csv(filename_csv, index=False)
            df.to_parquet(filename_parquet, index=False)
            print(f"Data for {symbol} saved to {filename_csv} and {filename_parquet}")

        except Exception as e:
            print(f"Error downloading data for {symbol}: {e}")
    mt5.shutdown()

# List of assets
assets = [
    'AGFS',    # Iagro
    'BGI$',    # Future Catle
    'IND$',    # Future bovespa index
    'IBOV',    # Bovespa Index
    'ICON',    # Consumer Index
    'ICF$',    # Future Coffee
    'CCM$',    # Future Corn
    'IVVB11',  # ETF (S&P500)
    'GOLD11',  # Future Gold
    'DOL$',    # Future Dollar
    'DI1$',    # Future Selic Proxy
    'IFIX'     # Real state index 

]

# starting date to downlaod data
start_date = datetime(2022,6,1)

# Directory to save the data
data_dir = os.path.join('..', 'data', 'assets').replace("/", "\\")


# Calling functions to stablish connection with mt5
connect_to_mt5(login, server, password)
download_and_save_from_mt5(symbols=assets, 
                           timeframe= mt5.TIMEFRAME_M15,
                           start_date = start_date,
                           save_dir =data_dir )
