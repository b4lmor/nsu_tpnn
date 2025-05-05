import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM
from tensorflow.keras.optimizers import Adam


def load_data(filepath):
    """Загрузка и подготовка данных"""

    df = pd.read_csv(filepath, parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df


def prepare_data(df, target_col, feature_cols, window_size=24, horizon=1):
    """Нормализация данных и создание временных окон"""

    # Нормализация
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[[target_col] + feature_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=[target_col] + feature_cols, index=df.index)

    # Создание временных окон
    X, y = [], []
    for i in range(len(scaled_df) - window_size - horizon + 1):
        X.append(scaled_df[feature_cols].values[i:(i + window_size)])
        y.append(scaled_df[target_col].iloc[i + window_size + horizon - 1])

    return np.array(X), np.array(y), scaler


def split_data(X, y, train_ratio=0.8):
    """Разделение данных на обучающую и тестовую выборки"""

    train_size = int(train_ratio * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test


def build_model(model_type, window_size, n_features):
    """Создание модели RNN/GRU/LSTM"""

    model = Sequential()

    if model_type == 'rnn':
        model.add(SimpleRNN(50, activation='tanh', input_shape=(window_size, n_features)))

    elif model_type == 'gru':
        model.add(GRU(50, activation='tanh', input_shape=(window_size, n_features)))

    else:  # lstm
        model.add(LSTM(50, activation='tanh', input_shape=(window_size, n_features)))

    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def train_and_evaluate(models, X_train, y_train, X_test, y_test, scaler):
    """Обучение моделей и оценка результатов"""

    histories = {}
    predictions = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        histories[name] = model.fit(X_train, y_train,
                                    epochs=30,
                                    batch_size=32,
                                    validation_data=(X_test, y_test),
                                    verbose=1)

        # Предсказания
        pred = model.predict(X_test)

        # Обратное масштабирование
        dummy = np.zeros((len(pred), 1 + len(feature_cols)))
        dummy[:, 0] = pred.flatten()
        pred_inv = scaler.inverse_transform(dummy)[:, 0]

        dummy[:, 0] = y_test
        y_test_inv = scaler.inverse_transform(dummy)[:, 0]

        predictions[name] = {
            'predictions': pred_inv,
            'true_values': y_test_inv,
            'rmse': np.sqrt(mean_squared_error(y_test_inv, pred_inv))
        }

    return histories, predictions


def plot_results(predictions):
    """Визуализация результатов"""

    plt.figure(figsize=(14, 7))
    for name, res in predictions.items():
        plt.plot(res['predictions'][:200], label=f'{name} Predictions')

    plt.plot(predictions['LSTM']['true_values'][:200], label='True Values', alpha=0.6)
    plt.title('Energy Consumption Prediction Comparison')
    plt.ylabel('Energy Usage (kWh)')
    plt.xlabel('Time Step')
    plt.legend()
    plt.show()


def plot_comparison(predictions, n_points=200):
    """Визуализация сравнения всех моделей (исправленная версия)"""
    plt.figure(figsize=(15, 10))

    # Основной график предсказаний
    plt.subplot(2, 1, 1)
    true_values = next(iter(predictions.values()))['true_values']  # Берем истинные значения из любой модели

    plt.plot(true_values[:n_points],
             label='Истинные значения',
             color='black',
             linewidth=2,
             alpha=0.8)

    # Цвета для моделей
    colors = {
        'RNN': '#FF5252',  # Красный
        'GRU': '#4CAF50',  # Зеленый
        'LSTM': '#2196F3'  # Синий
    }

    for name, res in predictions.items():
        plt.plot(res['predictions'][:n_points],
                 label=f'{name} (RMSE: {res["rmse"]:.2f})',
                 color=colors[name],
                 alpha=0.7,
                 linewidth=1.5)

    plt.title('Сравнение моделей прогнозирования энергопотребления')
    plt.ylabel('Потребление (kWh)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # График ошибок
    plt.subplot(2, 1, 2)
    for name, res in predictions.items():
        error = res['predictions'][:n_points] - true_values[:n_points]
        plt.plot(error,
                 label=f'{name}',
                 color=colors[name],
                 alpha=0.6)

    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.title('Ошибки предсказаний')
    plt.ylabel('Ошибка (kWh)')
    plt.xlabel('Временные точки (15-минутные интервалы)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def main():
    # Параметры
    filepath = '../steel_industry_data.csv'
    target_col = 'Usage_kWh'
    feature_cols = [
        'Lagging_Current_Reactive.Power_kVarh',
        'Leading_Current_Reactive_Power_kVarh',
        'CO2(tCO2)',
        'NSM'
    ]
    window_size = 24  # 6 часов при 15-минутных интервалах
    horizon = 4  # Прогноз на 1 час вперед

    # Основной поток
    print("1. Loading and preparing data...")
    df = load_data(filepath)
    X, y, scaler = prepare_data(df, target_col, feature_cols, window_size, horizon)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n2. Building models...")
    models = {
        'RNN': build_model('rnn', window_size, len(feature_cols)),
        'GRU': build_model('gru', window_size, len(feature_cols)),
        'LSTM': build_model('lstm', window_size, len(feature_cols))
    }

    print("\n3. Training and evaluating models...")
    histories, predictions = train_and_evaluate(models, X_train, y_train, X_test, y_test, scaler)

    print("\n4. Results:")
    plot_comprasion(predictions)
    
    for name, res in predictions.items():
        print(f"{name} RMSE: {res['rmse']:.2f} kWh")

    print("\n5. Plotting results...")
    plot_results(predictions)


if __name__ == "__main__":
    main()
