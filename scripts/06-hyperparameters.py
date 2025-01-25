import sys
import os
import myFunctions
from myFunctions import install_packages, save_table 
### packages required
install_packages()

### importing manipulation packages
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt

### importing torch packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

### importing optuna packages
import optuna
import optuna.visualization as vis

### importing metrics packages
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error, accuracy_score

## Model Classes
# LSTM Model
class LSTMModel(nn.Module):
    """
    LSTM-based model for time-series tasks, supporting regression and classification.

    Args:
        input_size (int): Number of input features per time step.
        hidden_size (int): Number of hidden units in each LSTM layer.
        num_layers (int): Number of LSTM layers.
        dropout (float): Dropout rate for regularization.
        target (str): Target type ('price' for regression, 'behavior' for classification).
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout, target):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        if 'behavior' in target:
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 2)
            )
        else:
            self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden[-1])
        return output

class GRUModel(nn.Module):
    """
    GRU-based model for time-series tasks, supporting regression and classification.

    Args:
        input_size (int): Number of input features per time step.
        hidden_size (int): Number of hidden units in each GRU layer.
        num_layers (int): Number of GRU layers.
        dropout (float): Dropout rate for regularization.
        target (str): Target type ('price' for regression, 'behavior' for classification).
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout, target):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        if 'behavior' in target:
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 2)
            )
        else:
            self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, hidden = self.gru(x)
        output = self.fc(hidden[-1])
        return output

class CNNLSTMModel(nn.Module):
    """
    Hybrid CNN-LSTM model for time-series tasks, supporting regression and classification.

    Args:
        input_size (int): Number of input features per time step.
        hidden_size (int): Number of hidden units in each LSTM layer.
        num_layers (int): Number of LSTM layers.
        dropout (float): Dropout rate for regularization.
        conv_filters (int): Number of filters in the 1D convolutional layer.
        target (str): Target type ('price' for regression, 'behavior' for classification).
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout, conv_filters, target):
        super(CNNLSTMModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=conv_filters, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(conv_filters, hidden_size, num_layers, batch_first=True, dropout=dropout)
        if 'behavior' in target:
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 2)
            )
        else:
            self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden[-1])
        return output

class CNNGRUModel(nn.Module):
    """
    Hybrid CNN-GRU model for time-series tasks, supporting regression and classification.

    Args:
        input_size (int): Number of input features per time step.
        hidden_size (int): Number of hidden units in each GRU layer.
        num_layers (int): Number of GRU layers.
        dropout (float): Dropout rate for regularization.
        conv_filters (int): Number of filters in the 1D convolutional layer.
        target (str): Target type ('price' for regression, 'behavior' for classification).
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout, conv_filters, target):
        super(CNNGRUModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=conv_filters, kernel_size=3, stride=1, padding=1)
        self.gru = nn.GRU(conv_filters, hidden_size, num_layers, batch_first=True, dropout=dropout)
        if 'behavior' in target:
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 2)
            )
        else:
            self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)
        _, hidden = self.gru(x)
        output = self.fc(hidden[-1])
        return output

def scale_features(df, exclude_columns=['date', 'day']):
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

    Args:
        df (pd.DataFrame): The DataFrame to be segmented.
        test_size (float): The percentage of data to be used for testing.

    Returns:
        pd.DataFrame: Training data.
        pd.DataFrame: Testing data.
    """

    # Calculates the size of the test set
    test_len = int(len(df) * test_size)

    # Segments the data
    train_data = df[:-test_len]  # 85% for training
    test_data = df[-test_len:]   # 15% for testing

    return train_data,test_data


def train_evaluate_model(trial, X_train, y_train, X_test, y_test, target, window_size, look_forward, model_type, study_name, model_dir):
    """
    Train and evaluate a model using the given parameters and data.

    Args:
        trial (optuna.Trial): The current optuna trial to optimize hyperparameters.
        X_train (ndarray): The training input data (features).
        y_train (ndarray): The target labels for the training data.
        X_test (ndarray): The test input data (features).
        y_test (ndarray): The target labels for the test data.
        target (str): Target type ('price' for regression, 'behavior' for classification).
        window_size (int): The size of the sliding window for time-series data.
        look_forward (int): The number of steps ahead to predict.
        model_type (str): The type of model ('LSTM', 'GRU', 'CNN-LSTM', 'CNN-GRU').
        study_name (str): Name of the optuna study for logging purposes.
        model_dir (str): Directory to save the model and trial results.

    Returns:
        float: The evaluation error (either MSE or AUC depending on the task).
    """
    # Set up the device for computation (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert input data to PyTorch tensors and move them to the appropriate device
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long if 'behavior' in target else torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long if 'behavior' in target else torch.float32).to(device)

    # Suggest hyperparameters using Optuna
    epochs = trial.suggest_int('epochs',350, 1000)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    num_layers = trial.suggest_int('num_layers', 2, 3)
    hidden_size = trial.suggest_int('hidden_size', 16, 128)

    if model_type in ['CNN-LSTM', 'CNN-GRU']:
        conv_filters = trial.suggest_categorical('conv_filters', [32, 64, 128])

    # Initialize the model based on the selected model type
    if model_type == 'LSTM':
        model = LSTMModel(input_size=X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, target=target).to(device)
    elif model_type == 'GRU':
        model = GRUModel(input_size=X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, target=target).to(device)
    elif model_type == 'CNN-LSTM':
        model = CNNLSTMModel(input_size=X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, conv_filters=conv_filters, target=target).to(device)
    elif model_type == 'CNN-GRU':
        model = CNNGRUModel(input_size=X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, conv_filters=conv_filters, target=target).to(device)

    # Define the optimizer and the loss function (CrossEntropyLoss for classification, MSELoss for regression)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss() if 'behavior' in target else nn.MSELoss()

    # Create DataLoader for training data
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)

    # Initialize a dictionary to track loss at each epoch
    average_losses = {epoch: [] for epoch in range(epochs)}

    # Training loop with dimension adjustments
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)

            if 'behavior' in target:
               output = output.view(-1, 2)
            else:
               output = output.view(-1)

            y_batch = y_batch.view(-1)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        average_losses[epoch].append(avg_loss)

        # Report the loss to Optuna for pruning and optimization
        trial.report(avg_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Print the loss every 50 epochs
        if epoch % 50 == 0:
            print(f'Epoch {epoch} loss: {avg_loss}')

    # Create a directory to save trial results
    trials_dir = os.path.join(model_dir, 'trials')
    if not os.path.exists(trials_dir):
        os.makedirs(trials_dir)

    # Optionally, plot and save the loss decay curve (this part is commented out for now)
    # plot_loss_decay(average_losses, epochs, study_name, trials_dir, trial.number)

    # Evaluate the model on the test data
    model.eval()
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    predictions = []
    true_values = []

    with torch.no_grad():
      for X_batch, y_batch in test_loader:
          output = model(X_batch)

          if 'behavior' in target:
              _, predicted = torch.max(output, 1)

          else:
              predicted = output.squeeze()
              predicted = predicted.reshape(-1)
              print(len(predicted))

          predictions.append(predicted.cpu().numpy())
          true_values.append(y_batch.cpu().numpy())

    if any(p.size == 0 for p in predictions) or any(t.size == 0 for t in true_values):
      print(f"Error: One or more arrays in 'predictions' or 'true_values' are empty. Pruning trial {trial.number}.")
      raise optuna.exceptions.TrialPruned()

    predictions = np.concatenate(predictions, axis=0)
    true_values = np.concatenate(true_values, axis=0)

    # Calculate error (Mean Squared Error for regression or AUC for classification)
    if 'price' in target:
        error = mean_squared_error(true_values, predictions)
    else:
        error = roc_auc_score(y_test.cpu().numpy(), predictions)

    # Save loss decay data to a file
    loss_decay_file = os.path.join(trials_dir, f"{study_name}_trial_{trial.number}_loss_decay.pkl")
    with open(loss_decay_file, 'wb') as f:
        joblib.dump(average_losses, f)

    return error


def optimize_models(df: pd.DataFrame,
                    targets: list,
                    features: list,
                    look_backs: list,
                    look_forwards: list,
                    target_dir: str = None,
                    models: list = None,
                    max_samples: int = 100):
    study_results = {}

    if target_dir is not None:
        input_dir = os.path.join('..', 'data', 'hyperparameters', target_dir)
    else:
        input_dir = os.path.join('..', 'data', 'hyperparameters')

    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    exclude_columns = ['date', 'day']
    df_scaled, _ = scale_features(df, exclude_columns)
    train_data, test_data = segment_data(df_scaled, test_size=0.15)

    for window_size in look_backs:
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

                if models is None:
                    models = ['LSTM', 'GRU', 'CNN-LSTM', 'CNN-GRU']

                for model_type in tqdm(models):
                    study_name = f"{model_type}_look_back_{window_size}_look_forward_{look_forward}_{target}"
                    model_dir = os.path.join(input_dir, study_name)

                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)

                    # Define optimization direction
                    direction = 'minimize' if 'price' in target else 'maximize'

                    # Creating the Optuna study with a pruner

                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=350, interval_steps=1)
                    study = optuna.create_study(direction=direction, study_name=study_name, pruner=pruner)
                    study.optimize(
                        lambda trial: train_evaluate_model(
                            trial, X_train, y_train, X_test, y_test, target, window_size, look_forward, model_type, study_name, model_dir
                        ),
                        n_trials=max_samples
                    )

                    study_file = os.path.join(model_dir, f"{study_name}_study.pkl")
                    joblib.dump(study, study_file)
                    print(f"{study_name} saved to {study_file}")

                    best_params = study.best_params
                    best_trial_index = study.best_trial.number
                    best_trial_value = study.best_value

                    best_params_dict = {
                        'best_params': best_params,
                        'best_trial_index': best_trial_index,
                        'best_trial_value': best_trial_value
                    }
                    params_file = os.path.join(model_dir, f"{study_name}_best_params.pkl")
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
                    print(f"Best hyperparameters for {model_type}, lookback: {window_size}, look forward: {look_forward}, target {target}: {best_params}")
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
features_name = 'D1_features.pkl'
input_dir_features = os.path.join('..', 'data', 'features')

D1_df = pd.read_parquet(os.path.join(input_dir, 'D1_df.parquet'))
M15_df = pd.read_parquet(os.path.join(input_dir, 'M15_df.parquet'))
D1_features = joblib.load(os.path.join(input_dir_features, 'D1_features.pkl'))
M15_features = joblib.load(os.path.join(input_dir_features, 'M15_features.pkl'))

models = ['LSTM', 'GRU', 'CNN-LSTM', 'CNN-GRU'] # define one or more models
targets = ['close_price_target', 'open_price_target', 'behavior_target']
windows = [7, 15, 30, 45, 60] 
look_forwards = [1]  

# Call the optimize_models function with acess to a GPU otherwise it's definetly not going to work
study_results = optimize_models(D1_df,
                                targets=targets,
                                features=D1_features, 
                                windows=windows, 
                                look_forwards=look_forwards,
                                max_samples=50)
