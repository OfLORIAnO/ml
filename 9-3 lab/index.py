import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import logging

tf.get_logger().setLevel(logging.ERROR)

# Загрузка датасета CIFAR-10
cifar_dataset = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar_dataset.load_data()

# Выбираем случайные 30 индексов
random_indices = np.random.choice(len(train_images), 100, replace=False)

# Количество колонок и строк
cols = 10
rows = 10

# Создание графика
plt.figure(figsize=(cols * 2, rows * 2))
for i, idx in enumerate(random_indices):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(train_images[idx])
    plt.axis("off")

plt.tight_layout()
plt.show()

# Вывод информации о картинках
print("Train Images Shape:", train_images.shape)
print("Test Images Shape:", test_images.shape)
print("Data Type:", train_images.dtype)
print("Min Pixel Value:", train_images.min())
print("Max Pixel Value:", train_images.max())
