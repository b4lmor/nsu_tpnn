import numpy as np
import pandas as pd
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from network import Network

data = pd.read_csv('../dataset/laptop_prices.csv')

data = pd.get_dummies(data, columns=['Brand'], drop_first=True)

bins, labels = [0, 10000, 20000, 30000, np.inf], [0, 1, 2, 3]

data['Price_Category'] = pd.cut(data['Price'], bins=bins, labels=labels)

y = to_categorical(data['Price_Category'])

X = data.drop(['Price', 'Price_Category'], axis=1).values

X_train, X_test, y_train, y_test, y_train_original, y_test_original \
    = train_test_split(X, y, data['Price'].values, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

training_data = list(zip(X_train, y_train))

network = Network([X_train.shape[1], 64, 32, len(labels)])
network.SGD(training_data, epochs=30, mini_batch_size=10, eta=0.1)

test_data = list(zip(X_test, y_test))

predictions = [np.argmax(network.feedforward(x.reshape(-1, 1))) for x in X_test]

print("Первые 10 предсказанных категорий:")
pretty = lambda ind: f'[{bins[ind]} - {bins[ind+1]}]'
for i in range(10):
    print(f'Predicted: {pretty(predictions[i])}, Actual: {y_test_original[i]:.2f}')
