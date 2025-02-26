import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

results = []

# Запуск всех конфигураций
for name, model_fn in configs.items():
    print(f"Training: {name}")
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
    final_train_mae = history.history["mean_absolute_error"][-1]
    final_val_mae = history.history["val_mean_absolute_error"][-1]

    results.append(
        [name, final_train_loss, final_val_loss, final_train_mae, final_val_mae]
    )

# Вывод результатов
results_df = pd.DataFrame(
    results, columns=["Model", "Train MSE", "Test MSE", "Train MAE", "Test MAE"]
)

# Визуализация результатов
plt.figure(figsize=(12, 6))

# График MSE
plt.subplot(1, 2, 1)
plt.bar(results_df["Model"], results_df["Train MSE"], alpha=0.7, label="Train MSE")
plt.bar(results_df["Model"], results_df["Test MSE"], alpha=0.7, label="Test MSE")
plt.xticks(rotation=45, ha="right")
plt.ylabel("MSE")
plt.title("Mean Squared Error Comparison")
plt.legend()

# График MAE
plt.subplot(1, 2, 2)
plt.bar(results_df["Model"], results_df["Train MAE"], alpha=0.7, label="Train MAE")
plt.bar(results_df["Model"], results_df["Test MAE"], alpha=0.7, label="Test MAE")
plt.xticks(rotation=45, ha="right")
plt.ylabel("MAE")
plt.title("Mean Absolute Error Comparison")
plt.legend()

plt.tight_layout()
plt.show()
