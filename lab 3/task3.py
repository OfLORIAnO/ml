import numpy as np


# Класс многослойной нейронной сети с обратным распространением ошибки
class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1):
        """
        Инициализирует нейронную сеть с заданными размерами слоев.

        :param layer_sizes: Список, где каждый элемент определяет количество нейронов в соответствующем слое.
        :param learning_rate: Скорость обучения для обновления весов.
        """
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        # Инициализация весов и смещений для каждого слоя
        for i in range(len(layer_sizes) - 1):
            self.weights.append(
                np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1
            )
            self.biases.append(np.random.randn(layer_sizes[i + 1]) * 0.1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        """
        Прямое распространение сигнала через сеть.

        :param x: Входные данные.
        :return: Выход сети и значения активаций для каждого слоя.
        """
        activations = [x]
        for w, b in zip(self.weights, self.biases):
            x = self.sigmoid(np.dot(x, w) + b)
            activations.append(x)
        return activations

    def backward(self, activations, y):
        """
        Обратное распространение ошибки для обновления весов и смещений.

        :param activations: Список активаций для каждого слоя.
        :param y: Истинное значение (цель).
        """
        deltas = [activations[-1] - y]

        # Вычисление дельт для всех слоев
        for i in reversed(range(len(self.weights) - 1)):
            deltas.append(
                deltas[-1].dot(self.weights[i + 1].T)
                * self.sigmoid_derivative(activations[i + 1])
            )

        deltas.reverse()

        # Обновление весов и смещений
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * activations[i].T.dot(deltas[i])
            self.biases[i] -= self.learning_rate * deltas[i].mean(axis=0)

    def train(self, X, y, epochs=1000):
        """
        Обучение нейронной сети на основе входных данных и целевых значений.

        :param X: Входные данные.
        :param y: Целевые значения.
        :param epochs: Количество эпох обучения.
        """
        for epoch in range(epochs):
            activations = self.forward(X)
            self.backward(activations, y)

    def predict(self, x):
        """
        Предсказание результата для входных данных x.

        :param x: Входные данные.
        :return: Результат предсказания (0 или 1).
        """
        output = self.forward(x)[-1]
        return (output > 0.5).astype(int)


# Настройки сети
layer_sizes = [3, 4, 4, 1]  # Входной слой (3), два скрытых слоя (4), выходной слой (1)
nn = NeuralNetwork(layer_sizes, learning_rate=0.1)

# Данные для обучения
X = np.array(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]
)
y = np.array([[0], [1], [0], [1], [1], [1], [1], [1]])  # Логическая функция (пример)

# Обучение
nn.train(X, y, epochs=5000)

# Тестирование
print("Результаты предсказаний:")
for inputs in X:
    result = nn.predict(inputs.reshape(1, -1))
    print(f"Input: {inputs} -> Output: {result[0][0]}")
