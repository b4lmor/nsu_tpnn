import numpy as np
import pandas as pd
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from network import Network

# 1. Загрузка и подготовка данных
data = pd.read_csv('../dataset/laptop_prices.csv')
data = pd.get_dummies(data, columns=['Brand'], drop_first=True)

# 2. Создание категорий цен
bins, labels = [0, 10000, 20000, 30000, np.inf], [0, 1, 2, 3]
data['Price_Category'] = pd.cut(data['Price'], bins=bins, labels=labels)
y = to_categorical(data['Price_Category'])

# 3. Подготовка признаков
X = data.drop(['Price', 'Price_Category'], axis=1).values

# 4. Разделение данных
X_train, X_test, y_train, y_test, y_train_original, y_test_original \
    = train_test_split(X, y, data['Price'].values, test_size=0.2)

# 5. Масштабирование данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Подготовка данных для сети
training_data = list(zip(X_train, y_train))
test_data = list(zip(X_test, y_test))

# 7. Создание и обучение сети
network = Network([X_train.shape[1], 64, 32, len(labels)])
network.SGD(training_data, epochs=40, mini_batch_size=10, eta=0.5, test_data=test_data, categorical=True, stop_criteria=90)

# 8. Предсказание и вычисление точности
predictions = [np.argmax(network.feedforward(x.reshape(-1, 1))) for x in X_test]
true_labels = [np.argmax(y) for y in y_test]
accuracy = accuracy_score(true_labels, predictions)

print(f"Accuracy: {accuracy*100:.2f}%")

pretty = lambda ind: f'[{bins[ind]} - {bins[ind+1]}]'
for i in range(10):
    print(f'Predicted: {pretty(predictions[i])}, Actual: {y_test_original[i]:.2f}')