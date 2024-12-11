import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Загрузка данных MNIST
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]

# Преобразование меток в числовой формат
y = y.astype(int)

# Разделение данных на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Нормализация данных (стандартизация)
mean = np.mean(x_train)
stddev = np.std(x_train)
x_train = (x_train - mean) / stddev
x_test = (x_test - mean) / stddev

# One-hot encoding для меток
encoder = OneHotEncoder(categories="auto", sparse_output=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

# Проверим размеры данных
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")


# Визуализация примера из набора данных
def visualize_sample(images, labels, index):
    plt.imshow(images[index].reshape(28, 28), cmap="gray")
    plt.title(f"Label: {np.argmax(labels[index])}")
    plt.show()


# Пример визуализации
visualize_sample(x_train, y_train, 0)
