# import sys
# import subprocess
# def install_packages():
#     required_packages = [
#         "MetaTrader5",
#         "numpy",
#         "scikit-learn",
#         "joblib"
#     ]

#     for package in required_packages:
#         try:
#             __import__(package)
#         except ImportError:
#             print(f"Instalando {package}...")
#             subprocess.check_call([sys.executable, "-m", "pip", "install", package])
#     print("Todos os pacotes estão instalados.")

# install_packages()

import MetaTrader5 as mt5

if not mt5.initialize():
    print(f"Erro ao inicializar MetaTrader 5: {mt5.last_error()}")
else:
    print("MetaTrader 5 inicializado com sucesso.")

# Fechar a conexão
mt5.shutdown()