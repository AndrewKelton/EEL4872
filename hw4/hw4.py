''' 
* ----------------- *
* Andrew Kelton     *
* Homework 4        *
* EEL 4872          *
* Dr. Gurupur       *
* April 5, 2025     *
* ----------------- *

Forecasts the temperature in Fort Myers, FL over the next
5 days using and RNN. Reads input from 'fm-march-weather.csv'
and extracts the features: 
 - Temperature Avg (°F)
 - Dew Point Avg (°F)
 - Humidity Avg (%)
to train and test the data to predict the temperature.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


TIME_STEP=5
N_PAST=30
TARGET='Temperature Avg (°F)'
FEATURES=['Temperature Avg (°F)','Dew Point Avg (°F)','Humidity Avg (%)']
NUM_FEATURES=3

# Creates sequences and targets from DataFrame, returns X and y
def create_sequences(df : pd.DataFrame, output_size=1) -> np.ndarray | np.ndarray:
    data = df[FEATURES].values
    target = df[TARGET].values

    X, y = [], []

    for i in range(len(df) - TIME_STEP - output_size + 1):
        X_seq = data[i:i + TIME_STEP]
        y_seq = target[i + TIME_STEP:i + TIME_STEP + output_size]
        X.append(X_seq)
        y.append(y_seq)

    X=np.array(X) # torch.tensor(np.array(X), dtype=torch.float32)
    y=np.array(y) # torch.tensor(np.array(y), dtype=torch.float32)

    return X, y

# Split data into testing and training sets
def split_data(X : np.array, y : np.array, test_ratio=0.8) -> np.ndarray | np.ndarray | np.ndarray | np.ndarray:
    
    # chronologically split data
    split_idx = int(len(X) * test_ratio)
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

# Main 
def main():

    # extract data
    df = pd.read_csv('fm-march-weather.csv')
    df = df.dropna(subset=FEATURES)
    df = df.reset_index(drop=True)

    # normalize target
    target_scaler = MinMaxScaler()
    df[TARGET] = target_scaler.fit_transform(df[[TARGET]])

    # split data
    X, y = create_sequences(df, output_size=5)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # normalize features
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train.reshape(-1, NUM_FEATURES)).reshape(-1, TIME_STEP,NUM_FEATURES)
    X_test=scaler.transform(X_test.reshape(-1, NUM_FEATURES)).reshape(-1,TIME_STEP,NUM_FEATURES)

    # convert to pytorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)#.view(-1,1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)#.view(-1,1)

    # RNN class to predict temperature
    class TemperatureRNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(TemperatureRNN, self).__init__()
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.rnn(x)
            out = self.fc(out[:, -1, :])
            return out

    # model parameters
    input_size = NUM_FEATURES
    hidden_size = 64
    output_size = 5

    # initialize model
    model = TemperatureRNN(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train model
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epoch}], Loss: {loss.item():.4f}')

    # make predictions with test data
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)

    # reverse normalization and get real values
    predictions_rescaled = target_scaler.inverse_transform(predictions.detach().numpy())[0]
    actual_temp = target_scaler.inverse_transform(y_test_tensor.detach().numpy())[0]

    # remove dimensions
    predictions_rescaled = predictions_rescaled.flatten()
    actual_temp = actual_temp.flatten()

    # calculate accuracy
    percentage_error = np.abs((predictions_rescaled - actual_temp) / actual_temp) * 100
    mean_percentage_error = np.mean(percentage_error)
    accuracy_percentage = 100 - mean_percentage_error
    print(f'Mean Percentage Error: {mean_percentage_error:.2f}%')
    print(f'Accuracy: {accuracy_percentage:.2f}%')

    # graph the results
    plt.plot(range(1, 6), actual_temp, marker='o', label='Actual', color='red',)
    plt.plot(range(1, 6), predictions_rescaled, marker='x', label='Predicted', color='blue')
    for i in range(len(actual_temp)):
        plt.text(i + 1, actual_temp[i], f'{actual_temp[i]:.2f}', fontsize=10, 
             verticalalignment='top', horizontalalignment='right', color='red')
        plt.text(i + 1, predictions_rescaled[i], f'{predictions_rescaled[i]:.2f}', 
             fontsize=9, verticalalignment='bottom', horizontalalignment='left', color='blue')

    plt.text(1, max(actual_temp[-1], predictions_rescaled[-1]) + 5, f'Accuracy: {accuracy_percentage:.2f}%', 
         fontsize=14, color='green', weight='bold', verticalalignment='bottom')

    plt.xlabel('Day')
    plt.ylabel('Temperature (°F)')
    plt.title('5 Day Forecast')
    plt.xticks([1, 2, 3, 4, 5])
    plt.ylim(min(actual_temp[-1].min(), predictions_rescaled[-1].min()) - 10, 
         max(actual_temp[-1].max(), predictions_rescaled[-1].max()) + 10) 
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
