import numpy as np


# Сигмоида и её производная
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Вычисляет сигмоиду для заданного массива x."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Вычисляет производную сигмоиды для заданного массива x."""
    return x * (1 - x)


# tanh и её производная
def tanh(x: np.ndarray) -> np.ndarray:
    """Вычисляет гиперболический тангенс для заданного массива x."""
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Вычисляет производную гиперболического тангенса для заданного массива x."""
    return 1 - x**2


# Нейронная сеть с обратным распространением ошибки
class NeuralNetwork:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        learning_rate: float = 0.1,
    ) -> None:
        """
        Инициализирует нейронную сеть.

        :param input_size: Количество входных нейронов.
        :param hidden_size: Количество нейронов в скрытом слое.
        :param output_size: Количество выходных нейронов.
        :param learning_rate: Скорость обучения (шаг градиентного спуска).
        """
        self.learning_rate = learning_rate

        # Инициализация весов
        self.input_weights: np.ndarray = np.random.uniform(
            -1.0, 1.0, (hidden_size, input_size + 1)
        )  # Для скрытого слоя
        self.output_weights: np.ndarray = np.random.uniform(
            -1.0, 1.0, (output_size, hidden_size + 1)
        )  # Для выходного слоя

    def forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        """
        Выполняет прямой проход через сеть.

        :param inputs: Входные данные.
        :return: Выход сети.
        """
        # Добавляем bias к входам
        inputs = np.append(1, inputs)

        # Вычисляем выход скрытого слоя
        self.hidden_input: np.ndarray = np.dot(self.input_weights, inputs)
        self.hidden_output: np.ndarray = tanh(self.hidden_input)

        # Добавляем bias к скрытым выходам
        hidden_with_bias = np.append(1, self.hidden_output)

        # Вычисляем выход выходного слоя
        self.output_input: np.ndarray = np.dot(self.output_weights, hidden_with_bias)
        self.output_output: np.ndarray = sigmoid(self.output_input)

        return self.output_output

    def backward_pass(self, inputs: np.ndarray, target: float) -> None:
        """
        Выполняет обратное распространение ошибки.

        :param inputs: Входные данные.
        :param target: Целевое значение.
        """
        # Вычисляем ошибку выходного слоя
        output_error = target - self.output_output
        output_delta = output_error * sigmoid_derivative(self.output_output)

        # Добавляем bias к скрытым выходам
        hidden_with_bias = np.append(1, self.hidden_output)

        # Обновляем веса выходного слоя
        self.output_weights += self.learning_rate * np.outer(
            output_delta, hidden_with_bias
        )

        # Вычисляем ошибку скрытого слоя
        hidden_error = np.dot(self.output_weights[:, 1:].T, output_delta)
        hidden_delta = hidden_error * tanh_derivative(self.hidden_output)

        # Добавляем bias к входам
        inputs = np.append(1, inputs)

        # Обновляем веса скрытого слоя
        self.input_weights += self.learning_rate * np.outer(hidden_delta, inputs)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int) -> None:
        """
        Обучает нейронную сеть.

        :param X: Массив входных данных.
        :param y: Массив целевых значений.
        :param epochs: Количество эпох обучения.
        """
        for epoch in range(epochs):
            for inputs, target in zip(X, y):
                self.forward_pass(inputs)
                self.backward_pass(inputs, target)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Выполняет предсказание на основе входных данных.

        :param inputs: Входные данные.
        :return: Выход сети.
        """
        return self.forward_pass(inputs)


# Данные для обучения логической функции
X_train: np.ndarray = np.array(
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
# Целевые значения
y_train: np.ndarray = np.array([0, 1, 1, 1, 0, 0, 1, 1])

# Создаем нейронную сеть
nn = NeuralNetwork(input_size=3, hidden_size=2, output_size=1, learning_rate=0.1)

# Обучаем сеть
nn.train(X_train, y_train, epochs=100)

# Тестирование сети
print("Результаты предсказаний логической функции:")
for inputs in X_train:
    output = nn.predict(inputs)
    print(f"Input: {inputs} -> Output: {output.round()}")
