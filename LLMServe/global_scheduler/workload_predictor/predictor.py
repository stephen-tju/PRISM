import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import matplotlib.pyplot as plt


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class mLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(mLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ih = nn.Linear(input_size, 4 * hidden_size)
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size)
        self.W_m = nn.Linear(input_size, hidden_size)
        self.U_m = nn.Linear(hidden_size, hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input, hx=None):
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            cx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        else:
            hx, cx = hx

        m = self.tanh(self.W_m(input) * self.U_m(hx))

        gates = self.W_ih(input) + self.W_hh(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = self.sigmoid(ingate)
        forgetgate = self.sigmoid(forgetgate)
        cellgate = self.tanh(cellgate)
        outgate = self.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)

        hy = outgate * self.tanh(cy)

        return hy, cy

class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False):
        super(mLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.cells.append(mLSTMCell(input_size, hidden_size))
            else:
                self.cells.append(mLSTMCell(hidden_size, hidden_size))

    def forward(self, input, hx=None):
        if self.batch_first:
            input = input.transpose(0, 1)

        seq_len, batch_size, _ = input.size()

        if hx is None:
            hx = [None] * self.num_layers
        else:
            hx = list(zip(*hx))

        outputs = []
        for t in range(seq_len):
            x = input[t]
            for layer in range(self.num_layers):
                hx[layer] = self.cells[layer](x, hx[layer])
                x = hx[layer][0]
            outputs.append(x)

        outputs = torch.stack(outputs)

        if self.batch_first:
            outputs = outputs.transpose(0, 1)

        hx = list(zip(*hx))
        return outputs, hx


class mLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mLSTMModel, self).__init__()
        self.mLSTM = mLSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.mLSTM(x)
        out = self.fc(out[:, -1, :])
        return out


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class WorkloadPredictor:
    def __init__(self, train_data, sequence_length=25, hidden_dim=64, num_epochs=50, batch_size=32, lr=0.001, seed=42):
        set_seed(seed)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.sequence_length = sequence_length

        X = []
        y = []
        self.min_val, self.max_val = np.min(train_data['y']), np.max(train_data['y'])
        data = (train_data['y'] - self.min_val) / (self.max_val - self.min_val)

        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        X = np.array(X)

        X_train = np.reshape(X, (X.shape[0], X.shape[1], 1))
        y_train = np.array(y)

        train_dataset = TimeSeriesDataset(X_train, y_train)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.model = mLSTMModel(input_dim=1, hidden_dim=hidden_dim, output_dim=1).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()
                y_pred = self.model(X_batch).squeeze()
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

        self.history = data[-sequence_length:].tolist()
        self.history_buf = []  # stack

    def predict_next_window(self):
        input_data = np.array(self.history).reshape(1, self.sequence_length, 1)
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(input_tensor).squeeze().cpu().numpy()
        # logger.debug(f"History value list: {self.history} -> Prediction: {prediction}")
        prediction = prediction * (self.max_val - self.min_val) + self.min_val

        return prediction

    def add_history(self, ground_truth):
        scaled_ground_truth = (ground_truth - self.min_val) / (self.max_val - self.min_val)
        # self.history.pop(0)
        self.history_buf.append(self.history.pop(0))
        self.history.append(scaled_ground_truth)

    def update_history(self, ground_truth):
        scaled_ground_truth = (ground_truth - self.min_val) / (self.max_val - self.min_val)
        self.history[-1] = scaled_ground_truth 

    def rollback_history(self, step):
        for _ in range(step):
            self.history.pop()
            self.history.insert(0, self.history_buf.pop())
        # May not use all popped values, need to clear the buffer
        self.history_buf = []


# if __name__ == '__main__':
#     dataset_name = "code"
#     dataset_path = f"../../../data/workloads/Azure_{dataset_name}/cleaned.csv"
#     df = pd.read_csv(dataset_path)
#     time_window = 5 * 60
#     df['TimeWindow'] = (df['Timestamp'] // time_window) * time_window
#     grouped_data = df.groupby('TimeWindow').agg({
#         'Request tokens': 'sum',
#         'Response tokens': 'sum'
#     }).reset_index()[1:-1]


#     # for ploting
#     groundtruth_datas = []
#     prediction_1_datas = []
#     prediction_2_datas = []
#     history_length = 25

#     prompt_data = grouped_data[['Request tokens']]
#     prompt_data.columns = ['y']
#     train_size = int(len(prompt_data) * 0.5)
#     train_data = prompt_data[:train_size]
#     test_data = prompt_data[train_size:]['y'].values
#     print(f"Training prompt predictor")
#     predictor = WorkloadPredictor(train_data)
#     mean_test_y = test_data.mean()
#     predictions_1 = []
#     predictions_2 = []
#     for i, ground_truth in enumerate(test_data):
#         # Step 1
#         prediction_1 = predictor.predict_next_window()
#         predictions_1.append(prediction_1)
#         predictor.add_history(prediction_1)
#         # Step 2
#         prediction_2 = predictor.predict_next_window()
#         predictions_2.append(prediction_2)
#         # rollback
#         predictor.update_history(ground_truth)
    
#     groundtruth_datas.append(train_data['y'].values.tolist()[-history_length:] + test_data.tolist())
#     prediction_1_datas.append(predictions_1)
#     prediction_2_datas.append(predictions_2)
#     mae = mean_absolute_error(test_data, predictions_1)
#     print(f"MAE: {mae}")
#     print(f"MAE percentage: {mae / mean_test_y}")

#     response_data = grouped_data[['Response tokens']]
#     response_data.columns = ['y']
#     train_size = int(len(response_data) * 0.5)
#     train_data = response_data[:train_size]
#     test_data = response_data[train_size:]['y'].values
#     print(f"Training response predictor")
#     predictor = WorkloadPredictor(train_data)
#     mean_test_y = test_data.mean()
#     predictions_1 = []
#     predictions_2 = []
#     for ground_truth in test_data:
#         # Step 1
#         prediction_1 = predictor.predict_next_window()
#         predictions_1.append(prediction_1)
#         predictor.add_history(prediction_1)
#         # Step 2
#         prediction_2 = predictor.predict_next_window()
#         predictions_2.append(prediction_2)
#         # rollback
#         predictor.update_history(ground_truth)
    
#     groundtruth_datas.append(train_data['y'].values.tolist()[-history_length:] + test_data.tolist())
#     prediction_1_datas.append(predictions_1)
#     prediction_2_datas.append(predictions_2)
#     mae = mean_absolute_error(test_data, predictions_1)
#     print(f"MAE: {mae}")
#     print(f"MAE percentage: {mae / mean_test_y}")


#     print("Ploting prediction results")
#     fig, axes = plt.subplots(2, 1, figsize=(40, 8))
#     for i, ax in enumerate(axes):
#         groundtruth = groundtruth_datas[i]
#         times = list(range(len(groundtruth)))
#         ax.plot(times, groundtruth, '-', marker='o', label=None, color="blue", markersize=2)
        
#         for j, _ in enumerate(prediction_2_datas[i]):
#             ax.plot(
#                 [history_length + j, history_length + j + 1, history_length + j + 2], 
#                 [groundtruth[history_length + j], prediction_1_datas[i][j], prediction_2_datas[i][j]], 
#                 '-', marker='o', label=f"prediction-{j}", color="red", markersize=2)

#     axes[0].set_title("Prompt tokens prediction")
#     axes[1].set_title("Response tokens prediction")
#     fig.tight_layout()
#     plt.savefig(f'./Azure_{dataset_name}_mLSTM_prediction.png', dpi=200)



def train_and_predict(data):
    train_size = int(len(data) * 0.5)
    train_data = data[:train_size]
    test_data = data[train_size:]['y'].values
    predictor = WorkloadPredictor(train_data)

    predictions_1 = []
    predictions_2 = []
    for ground_truth in test_data:
        # Step 1
        prediction_1 = predictor.predict_next_window()
        predictions_1.append(prediction_1)
        predictor.add_history(prediction_1)
        # Step 2
        prediction_2 = predictor.predict_next_window()
        predictions_2.append(prediction_2)
        # Rollback
        predictor.update_history(ground_truth)
    return train_data['y'].values, test_data, predictions_1, predictions_2


def preprocess_data(df, data_type):
    if data_type == 'prompt':
        data = df[['Request tokens']]
    elif data_type == 'response':
        data = df[['Response tokens']]
    else:
        raise ValueError("data_type must be either 'prompt' or 'response'")
    data.columns = ['y']
    return data


def process_dataset(dataset_name):
    print(f"Processing dataset: {dataset_name}")
    dataset_path = f"../../../data/workloads/Azure_{dataset_name}/cleaned.csv"
    df = pd.read_csv(dataset_path)
    time_window = 5 * 60
    df['TimeWindow'] = (df['Timestamp'] // time_window) * time_window
    grouped_data = df.groupby('TimeWindow').agg({
        'Request tokens': 'sum',
        'Response tokens': 'sum'
    }).reset_index()[1:-1]

    # for ploting
    groundtruth_datas = []
    prediction_1_datas = []
    prediction_2_datas = []
    history_length = 25

    for data_type in ['prompt', 'response']:
        processed_data = preprocess_data(grouped_data, data_type)
        train_data, test_data, predictions_1, predictions_2 = train_and_predict(processed_data)
        
        error_percentage_1 = mean_absolute_error(test_data, predictions_1) / test_data.mean()
        error_percentage_2 = mean_absolute_error(test_data, predictions_2) / test_data.mean()
        print(f"Error Percentage for {data_type}s (Step 1): {error_percentage_1:.6f}")
        print(f"Error Percentage for {data_type}s (Step 2): {error_percentage_2:.6f}")

        groundtruth_datas.append(train_data.tolist()[-history_length:] + test_data.tolist())
        prediction_1_datas.append(predictions_1)
        prediction_2_datas.append(predictions_2)

    print("Ploting prediction results")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(40, 8))
    for i, ax in enumerate(axes):
        groundtruth = groundtruth_datas[i]
        times = list(range(len(groundtruth)))
        ax.plot(times, groundtruth, '-', marker='o', label=None, color="blue", markersize=2)
        
        for j, _ in enumerate(prediction_2_datas[i]):
            ax.plot(
                [history_length + j, history_length + j + 1, history_length + j + 2], 
                [groundtruth[history_length + j], prediction_1_datas[i][j], prediction_2_datas[i][j]], 
                '-', marker='o', label=None, color="red", markersize=2)
        ax.set_xlabel("Time")
        ax.set_ylabel("Tokens")

    axes[0].set_title("Prompt tokens prediction")
    axes[1].set_title("Response tokens prediction")
    fig.tight_layout()
    plt.savefig(f'./Azure_{dataset_name}_LSTM_prediction.png', dpi=200)


if __name__ == '__main__':
    for dataset_name in ['code']: #, 'conv']:
        process_dataset(dataset_name)
