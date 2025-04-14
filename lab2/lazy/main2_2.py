import pandas as pd
import numpy as np
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dense
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. Загрузка и препроцессинг данных
data = pd.read_csv('../dataset/laptop_prices.csv')

data = pd.get_dummies(data, columns=['Brand'], drop_first=True)

# 2. Преобразуем целевую переменную в категории
bins, labels = [0, 10000, 20000, 30000, np.inf], [0, 1, 2, 3]
y = pd.cut(data['Price'], bins=bins, labels=labels)
y_original = data['Price'].values  # Сохраняем оригинальные цены для вывода

y_categorical = to_categorical(y)

# 3. Выбираем признаки
X = data.drop('Price', axis=1).values

# 4. Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test, y_train_original, y_test_original = train_test_split(
    X, y_categorical, y_original, test_size=0.2, random_state=42
)

# 5. Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Создание модели
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))  # Выходной слой для 4 классов

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 7. Обучение модели с callback и выводом информации
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Вывод информации об обучении
print("\nTraining history:")
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Training stopped after {len(history.history['loss'])} epochs")

# 8. Предсказание на тестовых данных
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# 9. Вычисление метрик
accuracy = accuracy_score(y_true_classes, y_pred_classes) * 100

print(f"\nModel Accuracy: {accuracy:.2f}%")

# 10. Вывод первых нескольких предсказаний с красивым форматированием
print("\nSample predictions:")
pretty = lambda ind: f'[{bins[ind]} - {bins[ind+1]}]'
for i in range(10):
    print(f'Predicted: {pretty(y_pred_classes[i])}, Actual: {y_test_original[i]:.2f}')
