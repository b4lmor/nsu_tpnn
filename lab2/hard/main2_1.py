import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

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

network = Network(sizes=[X_train.shape[1], 64, 32, 32, 1])
network.SGD(training_data=list(zip(X_train, y_train_scaled)), epochs=10, mini_batch_size=10, eta=0.1)

predictions_scaled = np.array([network.feedforward(x.reshape(-1, 1)) for x in X_test])
predictions = price_scaler.inverse_transform(predictions_scaled.reshape(-1, 1))

print("Первые 10 предсказанных цен:")
for i in range(10):
    print(f"Предсказанная цена: {predictions[i][0]:.2f}, Фактическая цена: {y_test[i][0]:.2f}")
