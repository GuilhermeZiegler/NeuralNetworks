import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import optuna.visualization as vis
import os
import joblib

# Modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Usando a última saída da sequência
        return out

# Modelo GRU
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        gru_out, hn = self.gru(x)
        out = self.fc(gru_out[:, -1, :])  # Usando a última saída da sequência
        return out

# Modelo CNN-LSTM
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, conv_filters):
        super(CNNLSTMModel, self).__init__()
        self.conv1d = nn.Conv1d(input_size, conv_filters, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(conv_filters, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Para a camada Conv1d, precisamos de [batch, channels, seq_len]
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)  # Voltar para [batch, seq_len, channels]
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Usando a última saída da sequência
        return out

# Modelo CNN-GRU
class CNNGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, conv_filters):
        super(CNNGRUModel, self).__init__()
        self.conv1d = nn.Conv1d(input_size, conv_filters, kernel_size=3, padding=1)
        self.gru = nn.GRU(conv_filters, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Para a camada Conv1d, precisamos de [batch, channels, seq_len]
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)  # Voltar para [batch, seq_len, channels]
        gru_out, hn = self.gru(x)
        out = self.fc(gru_out[:, -1, :])  # Usando a última saída da sequência
        return out

# Função para treinar e avaliar o modelo
def train_evaluate_model(trial, X_train, y_train, X_test, y_test, target, window_size, look_forward, model_type):
    # Sugestões de hiperparâmetros do Optuna
    epochs = trial.suggest_int('epochs', 10, 20)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_uniform('dropout', 0.2, 0.5)

    # Para modelos híbridos, incluir a sugestão para o número de filtros (no caso, o número de canais de saída da camada convolucional)
    if model_type in ['CNN-LSTM', 'CNN-GRU']:
        conv_filters = trial.suggest_categorical('conv_filters', [32, 64, 128])  # Ajuste o número de filtros aqui

    # Definição do modelo com base no tipo escolhido
    if model_type == 'LSTM':
        model = LSTMModel(input_size=X_train.shape[1], 
                          hidden_size=hidden_size, 
                          num_layers=num_layers,
                          dropout=dropout)
    elif model_type == 'GRU':
        model = GRUModel(input_size=X_train.shape[1], 
                         hidden_size=hidden_size, 
                         num_layers=num_layers,
                         dropout=dropout)
    elif model_type == 'CNN-LSTM':
        model = CNNLSTMModel(input_size=X_train.shape[1], 
                             hidden_size=hidden_size, 
                             num_layers=num_layers,
                             dropout=dropout,
                             conv_filters=conv_filters)  # Passando o número de filtros para a CNN
    elif model_type == 'CNN-GRU':
        model = CNNGRUModel(input_size=X_train.shape[1], 
                            hidden_size=hidden_size, 
                            num_layers=num_layers,
                            dropout=dropout,
                            conv_filters=conv_filters)  # Passando o número de filtros para a CNN

    # Usando Adam como otimizador
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Criar o DataLoader com o tamanho do batch
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
    
    # Treinando o modelo
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    # Avaliar o modelo
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
    mse = mean_squared_error(true_values, predictions)
    
    return mse

def optimize_models(X_train, y_train, X_test, y_test, targets, windows, look_forwards, max_samples=100):
    study_results = {}
    base_dir = "..//hiperparametros"
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for window_size in windows:
        for look_forward in look_forwards:
            for target in targets:
                model_names = ['LSTM', 'GRU', 'CNN-LSTM', 'CNN-GRU']
                for model_type in model_names:
                    study_name = f"{model_type}_window{window_size}_look{look_forward}_target{target}"
                    model_dir = os.path.join(base_dir, study_name)
                    
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    
                    # Criar o estudo Optuna
                    study = optuna.create_study(direction='minimize', study_name=study_name)
                    study.optimize(
                        lambda trial: train_evaluate_model(
                            trial, X_train, y_train, X_test, y_test, target, window_size, look_forward, model_type
                        ),
                        n_trials=max_samples
                    )
                    
                    # Salvar o estudo completo
                    study_file = os.path.join(model_dir, f"{study_name}_study.pkl")
                    joblib.dump(study, study_file)
                    
                    # Salvar os hiperparâmetros ótimos
                    best_params = study.best_params
                    params_file = os.path.join(model_dir, f"{study_name}_best_params.txt")
                    with open(params_file, "w") as f:
                        for param, value in best_params.items():
                            f.write(f"{param}: {value}\n")
                    
                    # Gerar e salvar gráficos
                    try:
                        fig_optimization = vis.plot_optimization_history(study)
                        fig_optimization.write_image(os.path.join(model_dir, f"{study_name}_optimization_history.png"))
                        
                        fig_importances = vis.plot_param_importances(study)
                        fig_importances.write_image(os.path.join(model_dir, f"{study_name}_param_importances.png"))
                        
                        fig_slice = vis.plot_slice(study)
                        fig_slice.write_image(os.path.join(model_dir, f"{study_name}_slice_plot.png"))
                        
                        print(f"Gráficos salvos em {model_dir}")
                    except Exception as e:
                        print(f"Erro ao gerar gráficos para {study_name}: {e}")
                    
                    # Log para conferência
                    print(f"Salvo em {model_dir}")
                    print(f"Melhores parâmetros para {model_type} com janela {window_size}, look forward {look_forward}, target {target}: {best_params}")
                    
                    # Salvar os resultados no dicionário
                    study_results[study_name] = {
                        "study": study,
                        "best_params": best_params,
                        "directory": model_dir
                    }
    
    return study_results



