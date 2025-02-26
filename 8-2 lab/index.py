import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
import numpy as np
import pandas as pd

# Параметры обучения
EPOCHS = 500
BATCH_SIZE = 16

# Загрузка данных
boston_housing = keras.datasets.boston_housing
(raw_x_train, y_train), (raw_x_test, y_test) = boston_housing.load_data()

# Стандартизация данных
x_mean = np.mean(raw_x_train, axis=0)
x_stddev = np.std(raw_x_train, axis=0)
x_train = (raw_x_train - x_mean) / x_stddev
x_test = (raw_x_test - x_mean) / x_stddev

# Конфигурации моделей
configs = {
    "Base Model": lambda: Sequential(
        [
            Dense(64, activation="relu", input_shape=[13]),
            Dense(64, activation="relu"),
            Dense(1, activation="linear"),
        ]
    ),
    "L2 Regularization (0.1)": lambda: Sequential(
        [
            Dense(64, activation="relu", input_shape=[13], bias_regularizer=l2(0.1)),
            Dense(64, activation="relu", bias_regularizer=l2(0.1)),
            Dense(1, activation="linear"),
        ]
    ),
    "L2 + Dropout (0.2)": lambda: Sequential(
        [
            Dense(64, activation="relu", input_shape=[13], bias_regularizer=l2(0.1)),
            Dropout(0.2),
            Dense(64, activation="relu", bias_regularizer=l2(0.1)),
            Dense(1, activation="linear"),
        ]
    ),
    "More Neurons (128)": lambda: Sequential(
        [
            Dense(128, activation="relu", input_shape=[13], bias_regularizer=l2(0.1)),
            Dense(128, activation="relu", bias_regularizer=l2(0.1)),
            Dense(1, activation="linear"),
        ]
    ),
    "Final Model (Dropout 0.3)": lambda: Sequential(
        [
            Dense(128, activation="relu", input_shape=[13], bias_regularizer=l2(0.1)),
            Dense(128, activation="relu", bias_regularizer=l2(0.1)),
            Dropout(0.3),
            Dense(1, activation="linear"),
        ]
    ),
}

# Запуск всех конфигураций
for name, model_fn in configs.items():
    model = model_fn()
    model.compile(
        loss="mean_squared_error", optimizer="adam", metrics=["mean_absolute_error"]
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        shuffle=True,
    )

    # Получение финальных ошибок
    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]

    # Вывод данных в нужном формате
    print(
        f"{name}: Ошибка обучения = {final_train_loss:.4f}, Ошибка теста = {final_val_loss:.4f}"
    )
