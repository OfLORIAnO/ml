import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# Загрузка MNIST
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]


# Преобразуем метки в целые числа
y = y.astype(np.uint8)

# Нормализация данных
X = X / 255.0

# Метки для задачи распознавания цифры 4
y_binary = (y == 4).astype(np.uint8)

# Разделение данных
X_train, X_test = X[:5000], X[5000:12000]
y_train, y_test = y_binary[:5000], y_binary[5000:12000]

################################
# Берём любое изображение из данных (например, первое)
index = 0  # Индекс изображения, которое хотим показать
image = X_train[index].reshape(28, 28)  # Преобразуем в 28x28

# Показываем изображение
plt.imshow(image, cmap="gray")  # Выводим в оттенках серого
plt.title(f"Label: {y[index]}")  # Подписываем метку
plt.axis("off")  # Отключаем оси
plt.show()
################################


# Сигмоида
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Производная сигмоиды
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


# Функция потерь
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(
        y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9)
    )


# Нейронная сеть
class SimpleNN:
    def __init__(self, input_size, hidden_size):
        # Инициализация весов и смещений
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        dZ2 = self.A2 - y.reshape(-1, 1)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dZ1 = np.dot(dZ2, self.W2.T) * sigmoid_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1


# Инициализация сети
model = SimpleNN(input_size=784, hidden_size=16)

# Гиперпараметры
epochs = 100
learning_rate = 0.1

# Обучение
for epoch in range(epochs):
    output = model.forward(X_train)
    loss = binary_cross_entropy(y_train, output)
    model.backward(X_train, y_train, learning_rate)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# Тестирование
y_pred = model.forward(X_test)
y_pred_class = (y_pred >= 0.5).astype(int)
accuracy = np.mean(y_pred_class.flatten() == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
