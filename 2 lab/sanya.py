import numpy as np
import random

# Устанавливаем начальные параметры
np.random.seed(100)
random.seed(100)
LEARNING_RATE = 0.1


# Функции активации
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Класс для нейронной сети
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Инициализация весов
        self.weights_input_hidden = (
            2 * np.random.rand(self.input_size, self.hidden_size) - 1
        )
        self.weights_hidden_output = (
            2 * np.random.rand(self.hidden_size, self.output_size) - 1
        )

    def forward_pass(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden)
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output)
        self.final_output = sigmoid(self.final_input)
        return self.final_output

    def backward_pass(self, X, y):
        # Вычисляем ошибку
        output_error = y - self.final_output
        output_delta = output_error * sigmoid_derivative(self.final_output)

        # Обновляем веса между скрытым и выходным слоями
        self.weights_hidden_output += (
            np.dot(self.hidden_output.T, output_delta) * LEARNING_RATE
        )

        # Вычисляем ошибку скрытого слоя
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Обновляем веса между входным и скрытым слоями
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * LEARNING_RATE

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            self.forward_pass(X)
            self.backward_pass(X, y)


# Данные для логических операций
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([[0], [0], [0], [1]])

X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([[0], [1], [1], [1]])

X_not = np.array([[0], [1]])
y_not = np.array([[1], [0]])

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])


# Создаем и обучаем сети для AND, OR, NOT
nn_and = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn_and.train(X_and, y_and, epochs=10000)

nn_or = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn_or.train(X_or, y_or, epochs=10000)

nn_not = NeuralNetwork(input_size=1, hidden_size=4, output_size=1)
nn_not.train(X_not, y_not, epochs=10000)

nn_xor = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn_xor.train(X_xor, y_xor, epochs=10000)


# Функция для вычисления выражения ((x1 AND x2) OR NOT(x2 OR x3)) XOR x1
def compute_expression(x1, x2, x3):
    # Вычисляем AND
    and_result = nn_and.forward_pass(np.array([[x1, x2]]))

    # Вычисляем OR
    or_result = nn_or.forward_pass(np.array([[x2, x3]]))

    # Вычисляем NOT
    not_result = nn_not.forward_pass(np.array([[or_result[0][0]]]))

    # Вычисляем (x1 AND x2) OR NOT(x2 OR x3)
    or_combined_result = nn_or.forward_pass(
        np.array([[and_result[0][0], not_result[0][0]]])
    )

    # Вычисляем XOR с x1
    final_result = nn_xor.forward_pass(np.array([[or_combined_result[0][0], x1]]))

    return np.round(final_result)


# Проверяем на всех возможных комбинациях x1, x2, x3
print("Results for ((x1 AND x2) OR NOT(x2 OR x3)) XOR x1:")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        for x3 in [0, 1]:
            result = compute_expression(x1, x2, x3)
            print(f"x1={x1}, x2={x2}, x3={x3} -> {result[0][0]}")
