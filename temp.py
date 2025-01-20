import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from pycoingecko import CoinGeckoAPI
import datetime

# Инициализация клиента CoinGecko API
cg = CoinGeckoAPI()

# Запрашиваем данные для криптовалюты (например, BTC)
crypto_id = 'bitcoin'  # Используем ID криптовалюты для CoinGecko (например, bitcoin для BTC)
currency = 'usd'  # В какой валюте запрашиваем цену (например, USD)

# Получаем исторические данные (за последний год)
data = cg.get_coin_market_chart_range_by_id(id=crypto_id, vs_currency=currency, 
                                            from_timestamp=int((datetime.datetime.now() - datetime.timedelta(days=365)).timestamp()),  # 2 года
                                            to_timestamp=int(datetime.datetime.now().timestamp()))

# Преобразуем данные в DataFrame
prices = data['prices']  # Список цен
prices_df = pd.DataFrame(prices, columns=['Timestamp', 'Close'])

# Преобразуем временные метки в datetime
prices_df['Date'] = pd.to_datetime(prices_df['Timestamp'], unit='ms')
prices_df = prices_df[['Date', 'Close']]  # Оставляем только колонку с датой и ценой

# Преобразуем данные в числовой формат
prices_df['Close'] = pd.to_numeric(prices_df['Close'])

# Преобразование данных
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices_df['Close'].values.reshape(-1, 1))

# Создание обучающих и тестовых данных
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Функция для создания данных для LSTM
def create_dataset(data, time_step=60):
    x, y = [], []
    for i in range(len(data) - time_step - 1):
        x.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(x), np.array(y)

# Создание обучающих и тестовых наборов
time_step = 60
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)

# Форматирование данных для LSTM
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# Построение улучшенной модели LSTM с Dropout для предотвращения переобучения
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))  # Добавлен слой Dropout
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))  # Добавлен слой Dropout
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели с увеличением эпох
model.fit(x_train, y_train, epochs=20, batch_size=32)  # Увеличено количество эпох до 20

# Прогнозирование
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.plot(prices_df['Date'][train_size:], prices_df['Close'][train_size:], color='blue', label='Actual Price')
plt.plot(prices_df['Date'][train_size + time_step + 1:], predictions, color='red', label='Predicted Price')
plt.legend()
plt.title(f'Price Prediction for {crypto_id.capitalize()}')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.show()
