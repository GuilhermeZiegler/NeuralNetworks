
import sys
import os
import myFunctions
from myFunctions import install_packages, save_table 
### packages required
install_packages()

### importing required packages
from tabulate import tabulate
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import joblib
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import optuna
import optuna.visualization as vis
import os
import joblib
import shap
import matplotlib.pyplot as plt


## Model Classes
# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Using the last output of the sequence
        return out

# GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        gru_out, hn = self.gru(x)
        out = self.fc(gru_out[:, -1, :])  # Using the last output of the sequence
        return out

# CNN-LSTM Model
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, conv_filters):
        super(CNNLSTMModel, self).__init__()
        self.conv1d = nn.Conv1d(input_size, conv_filters, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(conv_filters, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # For Conv1d, we need [batch, channels, seq_len]
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)  # Returning to [batch, seq_len, channels]
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Using the last output of the sequence
        return out

# CNN-GRU Model
class CNNGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, conv_filters):
        super(CNNGRUModel, self).__init__()
        self.conv1d = nn.Conv1d(input_size, conv_filters, kernel_size=3, padding=1)
        self.gru = nn.GRU(conv_filters, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # For Conv1d, we need [batch, channels, seq_len]
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)  # Returning to [batch, seq_len, channels]
        gru_out, hn = self.gru(x)
        out = self.fc(gru_out[:, -1, :])  # Using the last output of the sequence
        return out

def scale_features(df, exclude_columns=['date', 'day']):
### PROCESSING DATA TO ARRAYS FOR WINDOWS SIZE AND LOOK FORWARDdef scale_features
    """
    Escalona todas as colunas do DataFrame, exceto as especificadas em exclude_columns.

    Parameters:
        df (pd.DataFrame): O DataFrame contendo os dados.
        exclude_columns (list): Lista de colunas que não serão escalonadas.

    Returns:
        df_scaled (pd.DataFrame): DataFrame com as colunas escalonadas.
        scalers (dict): Dicionário contendo os escaladores para cada coluna escalonada.
    """
    scalers = {}
    columns_to_scale = [col for col in df.columns if col not in exclude_columns]
    
    df_scaled = df.copy()
    for col in columns_to_scale:
        scaler = MinMaxScaler()
        df_scaled[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler
    
    return df_scaled, scalers

def data_to_array(df, window_size, target, features):
    """
    Prepares X and y with targets shifted for the next day after the window.

    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        window_size (int): The window size (e.g., 7 days).
        target (str): The target column name (e.g., 'close_price_target').
        features (list): List of feature column names (e.g., ['open', 'high', 'low', 'close']).

    Returns:
        X (np.ndarray): Input features.
        y (np.ndarray): Target values.
        y_dates (np.ndarray): Dates associated with the targets.
    """
    X = []
    y = []
    y_dates = []

    for i in range(len(df) - window_size):
        # Access the target column directly by its name
        target_value = df.iloc[i + window_size][target]
        y.append(target_value)
        y_dates.append(df.iloc[i + window_size]['date'])
        
        # Prepare the features using the provided column names
        X.append(df.iloc[i:i + window_size][features].values)

    return np.array(X), np.array(y), np.array(y_dates)

### SPLITING DATA TO TRAIN AND TEST
def segment_data(df, test_size=0.15):
    """
    Segments the data into training and test sets based on the test size percentage.
    """
    test_len = int(len(df) * test_size)
    train_data = df[:-test_len]  # 85% for training
    test_data = df[-test_len:]   # 15% for testing
    
    return train_data,test_data


### LOADING HYPERPARAMETERS
def parse_best_params_file(params_file_path):
    best_params = {}
    with open(params_file_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            if '.' in value:
                best_params[key] = float(value)
            elif value.isdigit():
                best_params[key] = int(value)
            else:
                best_params[key] = int(value) if value.isdigit() else value
    return best_params

### FUNCTION TO SAVE TRAINED MODEL
def save_model(model, train_dir, study_name):
    """
    Save the trained model to the specified directory.

    Parameters:
    model (torch.nn.Module): The trained PyTorch model.
    train_dir (str): The directory where the model should be saved.
    study_name (str): The name of the study, used to create a subdirectory for the model.

    Returns:
    None
    """
    model_path = os.path.join(train_dir, study_name, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

def save_loss_plot(losses, study_name, train_dir):
    """
    Save the training loss plot to the specified directory.

    Parameters:
    losses (list): List of training losses recorded at each epoch.
    study_name (str): The name of the study, used to create a subdirectory for the plot.
    train_dir (str): The directory where the loss plot should be saved.

    Returns:
    None
    """
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    loss_plot_path = os.path.join(train_dir, study_name, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss plot saved at {loss_plot_path}")

def save_shap_plots(model, X_train, feature_names, study_name, train_dir):
    """
    Generate and save SHAP plots for model interpretation.

    Parameters:
    model (torch.nn.Module): The trained PyTorch model.
    X_train (np.ndarray): The training data used for generating SHAP values.
    feature_names (list): List of feature names for labeling SHAP plots.
    study_name (str): The name of the study, used to create a subdirectory for the plots.
    train_dir (str): The directory where the SHAP plots should be saved.

    Returns:
    None
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    shap_summary_path = os.path.join(train_dir, study_name, 'shap_summary.png')
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
    plt.savefig(shap_summary_path)
    plt.close()
    print(f"SHAP summary plot saved at {shap_summary_path}")

    shap_dependent_path = os.path.join(train_dir, study_name, 'shap_dependence.png')
    shap.dependence_plot(0, shap_values.values, X_train, show=False)
    plt.savefig(shap_dependent_path)
    plt.close()
    print(f"SHAP dependence plot saved at {shap_dependent_path}")


def train_evaluate_model(X_train, y_train, X_test, y_test, target, window_size, look_forward, model_type, study_name, train_dir, best_params):
    """
    Train and evaluate the model on the provided data, then save the results.

    Parameters:
    X_train (np.ndarray): The training features.
    y_train (np.ndarray): The training target values.
    X_test (np.ndarray): The test features.
    y_test (np.ndarray): The test target values.
    target (str): The target variable name.
    window_size (int): The size of the sliding window for time series data.
    look_forward (int): The number of time steps to predict.
    model_type (str): The type of model to train ('LSTM', 'GRU', 'CNN-LSTM', 'CNN-GRU').
    study_name (str): The name of the study.
    train_dir (str): The directory to save the model and plots.
    best_params (dict): Dictionary of hyperparameters for training.

    Returns:
    float: The error metric (MSE for regression or accuracy for classification).
    """
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    epochs = best_params['epochs']
    batch_size = best_params['batch_size']
    learning_rate = best_params['learning_rate']
    dropout = best_params['dropout']
    num_layers = best_params['num_layers']
    hidden_size = best_params['hidden_size']
    conv_filters = best_params['conv_filters'] if model_type in ['CNN-LSTM', 'CNN-GRU'] else None

    if model_type == 'LSTM':
        model = LSTMModel(input_size=X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    elif model_type == 'GRU':
        model = GRUModel(input_size=X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    elif model_type == 'CNN-LSTM':
        model = CNNLSTMModel(input_size=X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, conv_filters=conv_filters)
    elif model_type == 'CNN-GRU':
        model = CNNGRUModel(input_size=X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, conv_filters=conv_filters)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss() if 'price' in target else torch.nn.BCELoss()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
    
    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_losses.append(total_loss / len(train_loader))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_losses[-1]}")

    model.eval()
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    predictions = []
    true_values = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            predictions.append(output.numpy())
            true_values.append(y_batch.numpy())

    predictions = np.concatenate(predictions, axis=0)
    true_values = np.concatenate(true_values, axis=0)

    if 'price' in target:
        error = mean_squared_error(true_values, predictions)
    else:
        predictions = (predictions > 0.5).astype(int)
        error = -accuracy_score(true_values, predictions)

    save_model(model, train_dir, study_name)
    save_loss_plot(epoch_losses, study_name, train_dir)

    feature_names = X_train.numpy().columns.tolist() if hasattr(X_train, 'columns') else [f'Feature {i}' for i in range(X_train.shape[1])]
    save_shap_plots(model, X_train, feature_names, study_name, train_dir)

    loss_decay_file = os.path.join(train_dir, study_name, f"{study_name}_loss_decay.pkl")
    with open(loss_decay_file, 'wb') as f:
        joblib.dump({'epoch_losses': epoch_losses}, f)

    return error

def train_models(models_dir, input_dir, input_dir_features, train_dir, target_type):
    """
    Train models for each study using the parameters stored in the model directory.

    Parameters:
    models_dir (str): The directory containing the model study folders.
    input_dir (str): The directory containing the data.
    input_dir_features (str): The directory containing the feature files.
    target_type (str): The type of target variable ('daily' or 'timestamp').

    Returns:
    None
    """

    train_dir = os.path.join('..',train_dir, target_type).replace("/", "\\")
    df_daily = pd.read_parquet(os.path.join(input_dir, 'df_daily.parquet')).replace("/", "\\")
    df_timestamp = pd.read_parquet(os.path.join(input_dir, 'df_timestamp.parquet')).replace("/", "\\")
    
    if target_type == 'daily':
        df = df_daily.copy()
        features = joblib.load(os.path.join(input_dir_features, 'daily_features.pkl'))
    else:
        df = df_timestamp.copy()
        features = joblib.load(os.path.join(input_dir_features, '15min_timestamp_features.pkl'))
    
    exclude_columns = ['date', 'day']
    df_scaled, scalers = scale_features(df, exclude_columns)
    train_data, test_data = segment_data(df_scaled, test_size=0.15)

    for study_name in os.listdir(models_dir):
        study_path = os.path.join(models_dir, study_name)

        if os.path.isdir(study_path) and f'{study_name}_best_params.pkl' in os.listdir(study_path):
            study_name_parts = study_name.split('_')
            model_type = study_name_parts[0]
            window_size = int(study_name_parts[study_name_parts.index('window') + 1])
            look_forward = int(study_name_parts[study_name_parts.index('look_forward') + 1])
            target = study_name_parts[study_name_parts.index('target') + 1]
            best_params_file = os.path.join(study_path, f'{study_name}_best_params.pkl')
            best_params = parse_best_params_file(best_params_file)

            X_train, y_train = data_to_array(train_data, window_size, target, features)
            X_test, y_test = data_to_array(test_data, window_size, target, features)

            error = train_evaluate_model(
                X_train, y_train, X_test, y_test, target, window_size, look_forward,
                model_type, study_name, train_dir, best_params
            )
            print(f"Study: {study_name}, Window: {window_size}, Look Forward: {look_forward}, Error: {error}")

# Call the function to process all studies
train_models(
    models_dir=os.path.join('..', 'data', 'models').replace("/", "\\"),
    input_dir=os.path.join('..', 'data', 'processed').replace("/", "\\"),
    input_dir_features=os.path.join('..', 'data', 'features').replace("/", "\\"),
    train_dir = os.path.join('..', 'data', 'train').replace("/", "\\"),
    target_type='daily' 
     # Change to 'timestamp' or 'daily' depending on your needs
)

train_models(
    models_dir=os.path.join('..', 'data', 'models').replace("/", "\\"),
    input_dir=os.path.join('..', 'data', 'processed').replace("/", "\\"),
    input_dir_features=os.path.join('..', 'data', 'features').replace("/", "\\"),
    train_dir = os.path.join('..', 'data', 'train').replace("/", "\\"),
    target_type='timestamp'  # Change to 'timestamp' or 'daily' depending on your needs
)
