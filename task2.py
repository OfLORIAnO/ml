import numpy as np
import random
import matplotlib.pyplot as plt


# Класс для одного нейрона
class Neuron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.randn(input_size + 1)  # Включаем bias
        self.learning_rate = learning_rate

    def compute_output(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]  # bias
        return 1 if summation >= 0 else 0  # Пороговая активация (бинарная)

    def update_weights(self, inputs, error):
        self.weights[1:] += self.learning_rate * error * inputs  # Обновление весов
        self.weights[0] += self.learning_rate * error  # Обновление bias


# Класс для персептрона
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.neuron = Neuron(input_size, learning_rate)

    def train(self, X, y, epochs=100):
        errors = []
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in zip(X, y):
                prediction = self.neuron.compute_output(inputs)
                error = target - prediction
                self.neuron.update_weights(inputs, error)
                total_error += abs(error)
            errors.append(total_error)
        return errors

    def predict(self, inputs):
        return self.neuron.compute_output(inputs)


# Функции для логических операций (используем персептроны)
def and_perceptron(X, y, epochs=100):
    perceptron = Perceptron(input_size=2)
    perceptron.train(X, y, epochs)
    return perceptron


def or_perceptron(X, y, epochs=100):
    perceptron = Perceptron(input_size=2)
    perceptron.train(X, y, epochs)
    return perceptron


def not_perceptron(X, y, epochs=100):
    perceptron = Perceptron(input_size=1)
    perceptron.train(X, y, epochs)
    return perceptron


# Создаем персептроны для каждого шага логической функции
def create_perceptrons():
    # Таблица истинности для AND, OR, NOT
    X_and_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    X_not = np.array([[0], [1]])

    # Обучение персептронов для AND, OR и NOT
    and_p = and_perceptron(X_and_or, np.array([0, 0, 0, 1]))  # x1 AND x2
    or_p = or_perceptron(X_and_or, np.array([0, 1, 1, 1]))  # x2 OR x3
    not_p = not_perceptron(X_not, np.array([1, 0]))  # NOT (x2 OR x3)

    return and_p, or_p, not_p


# Реализуем финальную логическую функцию: ((x1 AND x2) OR NOT(x2 OR x3)) XOR x1
def final_function(and_p, or_p, not_p, X):
    results = []
    for x in X:
        x1, x2, x3 = x[0], x[1], x[2]

        # Шаг 1: x1 AND x2
        and_result = and_p.predict([x1, x2])

        # Шаг 2: x2 OR x3
        or_result = or_p.predict([x2, x3])

        # Шаг 3: NOT (x2 OR x3)
        not_result = not_p.predict([or_result])

        # Шаг 4: (x1 AND x2) OR NOT(x2 OR x3)
        final_or = and_result or not_result

        # Шаг 5: XOR с x1
        xor_result = final_or ^ x1

        results.append(xor_result)

    return results


# Таблица истинности для входов x1, x2, x3
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

# Создаем персептроны
and_p, or_p, not_p = create_perceptrons()

# Вычисляем финальную функцию
final_results = final_function(and_p, or_p, not_p, X)

# Проверка результатов
print("Final Function Results:")
for inputs, result in zip(X, final_results):
    print(f"Input: {inputs} -> Output: {result}")
