import pandas as pd
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dense
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Загрузка и препроцессинг данных
data = pd.read_csv('../dataset/laptop_prices.csv')

data = pd.get_dummies(data, columns=['Brand'], drop_first=True)

X = data.drop('Price', axis=1).values
y = data['Price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

model.compile(loss='mean_squared_error', optimizer='adam')

# 3. Обучение модели с callback и выводом доп. информации
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

print("\nTraining history:")
print(f"Final training loss: {history.history['loss'][-1]:.2f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.2f}")
print(f"Training stopped after {len(history.history['loss'])} epochs")

# 4. Предсказание и оценка модели
y_pred = model.predict(X_test)

mape = mean_absolute_percentage_error(y_test, y_pred) * 100
accuracy = 100 - mape

print(f"\nModel Accuracy (100 - MAPE): {accuracy:.2f}%")
print(f"Mean Absolute Percentage Error: {mape:.2f}%")

print("\nSample predictions:")
for i in range(min(10, len(y_test))):
    print(f'Predicted: {y_pred[i][0]:.2f}, Actual: {y_test[i]:.2f}, Difference: {abs(y_pred[i][0] - y_test[i]):.2f}')
