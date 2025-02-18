import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# --- Загрузка и предобработка данных ---
cifar_dataset = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar_dataset.load_data()

# Нормализация
mean = np.mean(train_images)
stddev = np.std(train_images)
train_images = (train_images - mean) / stddev
test_images = (test_images - mean) / stddev

# One-hot encoding меток
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# --- Определение конфигураций моделей ---
configs = {
    "Conf1": [
        Conv2D(64, (5, 5), strides=(2, 2), activation="relu", padding="same"),
        Conv2D(64, (3, 3), strides=(2, 2), activation="relu", padding="same"),
        Flatten(),
        Dense(10, activation="softmax"),
    ],
    "Conf2": [
        Conv2D(64, (3, 3), strides=(2, 2), activation="relu", padding="same"),
        Conv2D(16, (2, 2), strides=(2, 2), activation="relu", padding="same"),
        Flatten(),
        Dense(10, activation="softmax"),
    ],
    "Conf3": [
        Conv2D(64, (3, 3), strides=(2, 2), activation="relu", padding="same"),
        Conv2D(16, (2, 2), strides=(2, 2), activation="relu", padding="same"),
        Dropout(0.2),
        Flatten(),
        Dense(10, activation="softmax"),
    ],
    "Conf4": [
        Conv2D(64, (4, 4), strides=(1, 1), activation="relu", padding="same"),
        Conv2D(64, (2, 2), strides=(2, 2), activation="relu", padding="same"),
        Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same"),
        MaxPooling2D(pool_size=(2, 2)),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Flatten(),
        Dense(10, activation="softmax"),
    ],
    "Conf5": [
        Conv2D(64, (4, 4), strides=(1, 1), activation="relu", padding="same"),
        Conv2D(64, (2, 2), strides=(2, 2), activation="relu", padding="same"),
        Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same"),
        Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same"),
        MaxPooling2D(pool_size=(2, 2)),
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Flatten(),
        Dense(10, activation="softmax"),
    ],
    "Conf6": [
        Conv2D(64, (4, 4), strides=(1, 1), activation="tanh", padding="same"),
        Conv2D(64, (2, 2), strides=(2, 2), activation="tanh", padding="same"),
        Conv2D(32, (3, 3), strides=(1, 1), activation="tanh", padding="same"),
        Conv2D(32, (3, 3), strides=(1, 1), activation="tanh", padding="same"),
        MaxPooling2D(pool_size=(2, 2)),
        Dense(64, activation="tanh"),
        Dense(64, activation="tanh"),
        Dropout(0.2),
        Flatten(),
        Dense(10, activation="softmax"),
    ],
}

EPOCHS = 5
BATCH_SIZE = 16
histories = {}

# --- Обучение моделей ---
for name, layers in configs.items():
    model = Sequential([keras.layers.Input(shape=(32, 32, 3))] + layers)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    print(f"🔹 Обучение модели {name} ...")
    history = model.fit(
        train_images,
        train_labels,
        validation_data=(test_images, test_labels),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    histories[name] = history

# --- Визуализация результатов ---
plt.figure(figsize=(12, 6))
for name, history in histories.items():
    plt.plot(history.history["val_loss"], label=f"{name} (Test Loss)")
plt.xlabel("Эпохи")
plt.ylabel("Ошибка")
plt.legend()
plt.title("Сравнение ошибок тестирования для разных конфигураций")
plt.show()

plt.figure(figsize=(12, 6))
for name, history in histories.items():
    plt.plot(history.history["loss"], label=f"{name} (Train Loss)")
plt.xlabel("Эпохи")
plt.ylabel("Ошибка")
plt.legend()
plt.title("Сравнение ошибок обучения для разных конфигураций")
plt.show()
