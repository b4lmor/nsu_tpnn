import pandas as pd
from keras import Sequential
from keras.src.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Загрузка и препроцессинг данных
data = pd.read_csv('../dataset/laptop_prices.csv')

data = pd.get_dummies(data, columns=['Brand'], drop_first=True)

X = data.drop('Price', axis=1).values
# X = data[['Storage_Capacity']].values

y = data['Price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Создание модели
model = Sequential()

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error')

# 3. Обучение модели
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

y_pred = model.predict(X_test)

# TODO выводить доп. информацию об эпохах, + критерий остановки

# TODO add metrics
for i in range(10):
    print(f'Predicted: {y_pred[i][0]}, Actual: {y_test[i]}')
