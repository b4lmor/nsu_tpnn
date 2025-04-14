import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error

from network import Network

data = pd.read_csv('../dataset/laptop_prices.csv')

data = pd.get_dummies(data, columns=['Brand'], drop_first=True)

X = data.drop('Price', axis=1).values
y = data['Price'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

price_scaler = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = price_scaler.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = price_scaler.transform(y_test.reshape(-1, 1))

test_data_for_monitoring = list(zip(X_test, y_test_scaled))

network = Network(sizes=[X_train.shape[1], 64, 32, 32, 1])
network.SGD(training_data=list(zip(X_train, y_train_scaled)),
           epochs=100,
           mini_batch_size=10,
           eta=0.7,
           test_data=test_data_for_monitoring)

predictions_scaled = np.array([network.feedforward(x.reshape(-1, 1)) for x in X_test])
predictions = price_scaler.inverse_transform(predictions_scaled.reshape(-1, 1))

mape = mean_absolute_percentage_error(y_test, predictions) * 100
accuracy = 100 - mape

print(f"Accuracy: {accuracy:.2f}%")

print("\nПримеры предсказаний:")
for i in range(min(10, len(y_test))):
    print(f"Pred: {predictions[i][0]:.2f} | Actual: {y_test[i][0]:.2f} | Error: {abs(predictions[i][0] - y_test[i][0]):.2f}")