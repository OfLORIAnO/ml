import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Параметры обучения
EPOCHS = 500
BATCH_SIZE = 16

# Загрузка данных
boston_housing = keras.datasets.boston_housing
(raw_x_train, y_train), (raw_x_test, y_test) = boston_housing.load_data()

# Вывод всех raw_x_train в консоль
for i, row in enumerate(raw_x_train):
    print(f"raw_x_train[{i}]: {row}")

# Стандартизация данных
x_mean = np.mean(raw_x_train, axis=0)
x_stddev = np.std(raw_x_train, axis=0)
x_train = (raw_x_train - x_mean) / x_stddev
x_test = (raw_x_test - x_mean) / x_stddev

# Создание модели
model = Sequential()
model.add(Dense(64, activation="relu", input_shape=[13]))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="linear"))

# Компиляция модели
model.compile(
    loss="mean_squared_error", optimizer="adam", metrics=["mean_absolute_error"]
)

# Обучение модели
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
    shuffle=True,
)

# Предсказания
predictions = model.predict(x_test)
for i in range(4):
    print(f"Prediction: {predictions[i]}, true value: {y_test[i]}")
