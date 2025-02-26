import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical  # type: ignore
import numpy as np
import logging
import matplotlib.pyplot as plt
import random


# Настройка логирования для уменьшения вывода предупреждений
tf.get_logger().setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Фиксация случайных значений для воспроизводимости
tf.random.set_seed(7)

# Глобальные параметры обучения
EPOCHS = 20  # Количество эпох обучения (добавлено мной)
BATCH_SIZE = 32  # Размер батча для обучения (добавлено мной)

# Загрузка и подготовка данных MNIST
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Стандартизация входных данных (добавлено мной)
mean = np.mean(train_images)
stddev = np.std(train_images)
train_images = (train_images - mean) / stddev
test_images = (test_images - mean) / stddev

# One-hot кодирование меток
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# Инициализация весов (Glorot, так как N = 15 нечётное) (добавлено мной)
initializer = keras.initializers.GlorotUniform()

# Определение модели с тремя скрытыми слоями и функцией активации tanh (добавлено мной)
model = keras.Sequential(
    [
        keras.layers.Flatten(
            input_shape=(28, 28)
        ),  # Преобразование 28x28 вектор в одномерный массив
        keras.layers.Dense(
            64, kernel_initializer=initializer, bias_initializer="zeros"
        ),
        keras.layers.BatchNormalization(),  # Пакетная нормализация перед активацией (добавлено мной)
        keras.layers.Activation(
            "tanh"
        ),  # Используем tanh, так как N нечётное (добавлено мной)
        keras.layers.Dense(
            32, kernel_initializer=initializer, bias_initializer="zeros"
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("tanh"),
        keras.layers.Dense(
            16, kernel_initializer=initializer, bias_initializer="zeros"
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("tanh"),
        keras.layers.Dense(
            10,
            activation="sigmoid",  # На выходе сигмоида для вероятностей классов
            kernel_initializer=initializer,
            bias_initializer="zeros",
        ),
    ]
)

# Компиляция модели с SGD оптимизатором и MSE функцией потерь (изменено мной)
opt = keras.optimizers.SGD(
    learning_rate=0.005
)  # Уменьшил скорость обучения до 0.005 (добавлено мной)
model.compile(loss="MSE", optimizer=opt, metrics=["accuracy"])

# Обучение модели
history = model.fit(
    train_images,
    train_labels,
    validation_data=(test_images, test_labels),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
    shuffle=True,
)

# Оценка модели на тестовых данных
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)

# Вывод результатов
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


def plot_training_progress(history):
    epochs_range = range(1, len(history.history["loss"]) + 1)

    # График функции потерь
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history.history["loss"], label="Training Loss")
    plt.plot(epochs_range, history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # График точности
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history.history["accuracy"], label="Training Accuracy")
    plt.plot(epochs_range, history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)


def plot_predictions(model, test_images, test_labels):
    indices = random.sample(
        range(len(test_images)), 9
    )  # Выбираем 9 случайных изображений (добавлено мной)
    images = test_images[indices]
    labels = np.argmax(test_labels[indices], axis=1)  # Истинные метки
    predictions = model.predict(images)  # Предсказания модели
    predicted_labels = np.argmax(predictions, axis=1)  # Предсказанные метки

    fig, axes = plt.subplots(3, 6, figsize=(15, 10))
    for i in range(9):
        # Отображение изображения цифры
        axes[i // 3, i % 3 * 2].imshow(images[i], cmap="gray")
        axes[i // 3, i % 3 * 2].set_title(
            f"True: {labels[i]}, Pred: {predicted_labels[i]}"
        )
        axes[i // 3, i % 3 * 2].axis("off")

        # Отображение вероятностей предсказания
        axes[i // 3, i % 3 * 2 + 1].bar(range(10), predictions[i])
        axes[i // 3, i % 3 * 2 + 1].set_xticks(range(10))
        axes[i // 3, i % 3 * 2 + 1].set_ylim(0, 1)
        axes[i // 3, i % 3 * 2 + 1].set_title("Predicted Probabilities")
        axes[i // 3, i % 3 * 2 + 1].grid(True)

    plt.tight_layout()
    plt.show()


plot_training_progress(history)
plot_predictions(model, test_images, test_labels)
