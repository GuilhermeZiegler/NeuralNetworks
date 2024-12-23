import sys
import os
import myFunctions
from myFunctions import install_packages, save_table 
### packages required
install_packages()

### importing required packages
from tabulate import tabulate
import pandas as pd
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import optuna
import optuna.visualization as vis
import os
import joblib


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


def train_evaluate_model(trial, X_train, y_train, X_test, y_test, target, window_size, look_forward, model_type, study_name, model_dir):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    epochs = trial.suggest_int('epochs', 10, 20)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    num_layers = trial.suggest_int('num_layers', 2, 3)
    hidden_size = trial.suggest_int('hidden_size', 16, 128)

    if model_type in ['CNN-LSTM', 'CNN-GRU']:
        conv_filters = trial.suggest_categorical('conv_filters', [32, 64, 128])

    # Initialize model based on type
    if model_type == 'LSTM':
        model = LSTMModel(input_size=X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    elif model_type == 'GRU':
        model = GRUModel(input_size=X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    elif model_type == 'CNN-LSTM':
        model = CNNLSTMModel(input_size=X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, conv_filters=conv_filters)
    elif model_type == 'CNN-GRU':
        model = CNNGRUModel(input_size=X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, conv_filters=conv_filters)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss() if 'price' in target else nn.BCELoss()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)

    average_losses = {epoch: [] for epoch in range(epochs)}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            if 'behavior' in target:
                output = torch.sigmoid(output)
                
            output = output.view(-1, 1) 
            y_batch = y_batch.view(-1, 1)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        average_losses[epoch].append(avg_loss)

    # Evaluation on test data
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

    # Calculate error based on target
    if 'price' in target:
        error = mean_squared_error(true_values, predictions)
    else:
        predictions = (predictions > 0.5).astype(int)
        error = -accuracy_score(true_values, predictions)

    # Save loss decay file
    loss_decay_file = os.path.join(model_dir, f"{study_name}_loss_decay.pkl")
    with open(loss_decay_file, 'wb') as f:
        joblib.dump(average_losses, f)

    # Plot and save loss decay graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), [np.mean(loss) for loss in average_losses.values()], label='Average Training Loss')
    plt.title(f'Loss Decay Over Epochs for {study_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image
    loss_plot_file = os.path.join(model_dir, f"{study_name}_loss_decay_plot.png")
    plt.savefig(loss_plot_file)
    plt.close()

    return error

def optimize_models(df, targets, features, windows, look_forwards, max_samples=100):
    """
    Optimizes models using Optuna for hyperparameter tuning. Now prepares the data within the function.
    """
    study_results = {}
    input_dir = os.path.join('..', 'data', 'models')
    input_dir = input_dir.replace("/", "\\")  # Ensure correct path format
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    
    exclude_columns = ['date', 'day']
    df_scaled, scalers = scale_features(df, exclude_columns)
    train_data, test_data = segment_data(df_scaled, test_size=0.15)
    
    for window_size in windows:
        for look_forward in look_forwards:
            for target in targets:
                X_train, y_train, y_dates = data_to_array(train_data.copy(), 
                                                          window_size,
                                                          target, 
                                                          features)

                X_test, y_test, _ = data_to_array(test_data.copy(), 
                                                  window_size, 
                                                  target, 
                                                  features)

                model_names = ['LSTM', 'GRU', 'CNN-LSTM', 'CNN-GRU']
                for model_type in tqdm(model_names):
                    study_name = f"{model_type}_window_{window_size}_look_forward_{look_forward}_{target}"
                    model_dir = os.path.join(input_dir, study_name).replace("/", "\\")
                    
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    
                    # Creating the Optuna study
                    study = optuna.create_study(direction='minimize', study_name=study_name)
                    study.optimize(
                        lambda trial: train_evaluate_model(
                            trial, X_train, y_train, X_test, y_test, target, window_size, look_forward, model_type, study_name, model_dir
                        ),
                        n_trials=max_samples
                    )

                    study_file = os.path.join(model_dir, f"{study_name}_study.pkl").replace("/", "\\")
                    joblib.dump(study, study_file)
                    print(f'{study_name} saved to {study_file}')


                    best_params = study.best_params
                    best_trial_index = study.best_trial.number 
                    best_trial_value = study.best_value  

                    best_params_dict = {
                        'best_params': best_params,
                        'best_trial_index': best_trial_index,
                        'best_trial_value': best_trial_value
                    }
                    params_file = os.path.join(model_dir, f"{study_name}_best_params.pkl").replace("/", "\\")
                    with open(params_file, "wb") as f:
                        joblib.dump(best_params_dict, f)

                    try:
                        fig_optimization = vis.plot_optimization_history(study)
                        fig_optimization.write_image(os.path.join(model_dir, f"{study_name}_optimization_history.png"))

                        fig_importances = vis.plot_param_importances(study)
                        fig_importances.write_image(os.path.join(model_dir, f"{study_name}_param_importances.png"))

                        fig_slice = vis.plot_slice(study)
                        fig_slice.write_image(os.path.join(model_dir, f"{study_name}_slice_plot.png"))

                        print(f"Images saved to {model_dir}")
                    except Exception as e:
                        print(f"Error in plotting images for {study_name}: {e}")


                    print(f"Saved to {model_dir}:")
                    print(f"Best hyperparameters for {model_type}, window size: {window_size}, look forward: {look_forward}, target {target}: {best_params}")
                    print(f"Best trial index: {best_trial_index}, Best trial value: {best_trial_value}")

                    study_results[study_name] = {
                        "study": study,
                        "best_params": best_params,
                        "best_trial_index": best_trial_index,
                        "best_trial_value": best_trial_value,
                        "directory": model_dir
                    }

    return study_results


### Loading data to optuna
input_dir = os.path.join('..', 'data', 'processed')
features_name = 'daily_features.pkl'
input_dir_features = os.path.join('..', 'data', 'features')

df_daily = pd.read_parquet(os.path.join(input_dir, 'df_daily.parquet')).replace("/", "\\")
df_timestamp = pd.read_parquet(os.path.join(input_dir, 'df_timestamp.parquet')).replace("/", "\\")
daily_features = joblib.load(os.path.join(input_dir_features, 'daily_features.pkl'))
timnestamp_features = joblib.load(os.path.join(input_dir_features, '15min_timestamp_features.pkl'))

features = daily_features
targets = ['close_price_target', 'open_price_target', 'behavior_target']
windows = [7, 15, 30, 45, 60] 
look_forwards = [1]  

# Call the optimize_models function
study_results = optimize_models(df_daily, targets,features, windows, look_forwards, max_samples=100)
