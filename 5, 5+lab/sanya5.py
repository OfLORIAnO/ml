import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Загружаем MNIST
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X, y = mnist["data"], mnist["target"].astype(np.uint8)

# Выделяем только цифры младшего разряда номера варианта
VARIANT_LAST_DIGIT = 7  # Младший разряд номера моего варианта
X_digit = X
y_digit_binary = (y == VARIANT_LAST_DIGIT).astype(int)  # Бинарные метки

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_digit, y_digit_binary, test_size=0.2, random_state=42
)
# Нормализация данных
X_train = X_train / 255.0
X_test = X_test / 255.0


# Сигмоида и её производная
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Простой класс для нейросети с бинарной классификацией
class ImbaNeuroNetwork:
    def __init__(self, input_size, hidden_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)  # Сигмоида
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)  # Сигмоида для бинарной классификации
        return self.A2

    def backward(self, X, y_true, y_pred, learning_rate):
        m = X.shape[0]
        dZ2 = y_pred - y_true.reshape(-1, 1)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2


# Обучение модели
nn = ImbaNeuroNetwork(input_size=28 * 28, hidden_size=128)
epochs = 1000
learning_rate = 0.1

for epoch in range(epochs):
    y_pred = nn.forward(X_train)

    y_train_reshaped = y_train.reshape(-1, 1)

    nn.backward(X_train, y_train, y_pred, learning_rate)

    if (epoch + 1) % 100 == 0:
        print(f"Эпоха {epoch + 1}")

# Тестирование модели
y_test_pred = nn.forward(X_test)
y_test_pred_binary = (y_test_pred > 0.5).astype(int)

# Общая точность
accuracy = np.mean(y_test_pred_binary.flatten() == y_test)
print(f"Общая точность на тестовых данных: {accuracy:.4f}")

# Визуализация 10 случайных изображений с предсказаниями
random_indices = np.random.choice(X_test.shape[0], 10, replace=False)
fig, axes = plt.subplots(1, 10, figsize=(20, 5))
fig.suptitle("Случайные изображения с вероятностью определения цифры 7")

for i, idx in enumerate(random_indices):
    ax = axes[i]
    ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
    probability = y_test_pred[idx, 0]
    ax.set_title(f"{probability:.2f}")
    ax.axis("off")

plt.tight_layout()
plt.show()
