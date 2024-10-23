import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# Класс для одного персептрона
class Perceptron:
    def __init__(self, input_size: int, learning_rate: float = 0.1) -> None:
        """
        Инициализирует персептрон с заданным количеством входов и скоростью обучения.

        :param input_size: Количество входов (без учета bias).
        :param learning_rate: Скорость обучения для обновления весов.
        """
        self.weights = np.random.randn(input_size + 1)  # Включаем bias (доп. вес)
        self.learning_rate = learning_rate

    def predict(self, inputs: np.ndarray) -> int:
        """
        Выполняет предсказание для входных данных на основе текущих весов.

        :param inputs: Входные данные (включая bias).
        :return: Результат предсказания (1 или 0).
        """
        # Вычисляем взвешенную сумму
        summation = np.dot(inputs, self.weights)
        # Применяем пороговую активацию: если сумма >= 0, возвращаем 1, иначе 0
        return 1 if summation >= 0 else 0

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> None:
        """
        Обучает персептрон на основе входных данных и целевых значений.

        :param X: Массив входных данных (включает bias в качестве первого элемента).
        :param y: Массив целевых значений (результатов).
        :param epochs: Количество эпох обучения.
        """
        for epoch in range(epochs):
            for inputs, target in zip(X, y):
                prediction = self.predict(inputs)  # Делаем предсказание
                error = target - prediction  # Вычисляем ошибку
                if error != 0:
                    # Обновляем веса для всех входов, кроме bias
                    self.weights[1:] += self.learning_rate * error * inputs[1:]
                    # Обновляем bias (первый вес)
                    self.weights[0] += self.learning_rate * error


# Создаем персептроны для логических операций AND, OR, NOT
and_perceptron = Perceptron(input_size=2)
or_perceptron = Perceptron(input_size=2)
not_perceptron = Perceptron(input_size=1)

# Данные для обучения персептронов
X_and_or = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])  # Включаем bias (1)
X_not = np.array([[1, 0], [1, 1]])  # Входы для NOT, включаем bias (1)
y_and = np.array([0, 0, 0, 1])  # Результаты для AND
y_or = np.array([0, 1, 1, 1])  # Результаты для OR
y_not = np.array([1, 0])  # Результаты для NOT

# Обучаем персептроны
and_perceptron.train(X_and_or, y_and, epochs=100)
or_perceptron.train(X_and_or, y_or, epochs=100)
not_perceptron.train(X_not, y_not, epochs=100)


# Логическая функция: (NOT(x1) AND x2) OR x3
def logical_function(x1, x2, x3) -> int:
    """
    Вычисляет логическую функцию (NOT(x1) AND x2) OR x3 с использованием персептронов.

    :param x1: Входное значение x1 (0 или 1).
    :param x2: Входное значение x2 (0 или 1).
    :param x3: Входное значение x3 (0 или 1).
    :return: Результат логической функции (0 или 1).
    """
    not_result = not_perceptron.predict(np.array([1, x1]))  # NOT(x1)
    and_result = and_perceptron.predict(np.array([1, not_result, x2]))  # NOT(x1) AND x2
    final_result = or_perceptron.predict(
        np.array([1, and_result, x3])
    )  # (NOT(x1) AND x2) OR x3
    return final_result


# Тестирование на всех возможных входах
X_test = np.array(
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

print("Результаты предсказаний логической функции (NOT(x1) AND x2) OR x3:")
for inputs in X_test:
    x1, x2, x3 = inputs
    result = logical_function(x1, x2, x3)
    print(f"Input: {inputs} -> Output: {result}")


# Визуализация логической функции с использованием networkx
def visualize_network() -> None:
    """
    Визуализирует структуру логической функции (NOT(x1) AND x2) OR x3 с весами на ребрах.
    """
    # Создаем граф
    G = nx.DiGraph()

    # Добавляем вершины
    G.add_nodes_from(["x1", "x2", "x3", "NOT", "AND", "OR", "y"])

    # Добавляем рёбра с весами (веса обученные персептронами)
    edges = [
        ("x1", "NOT", not_perceptron.weights[0]),  # NOT(x1)
        ("NOT", "AND", and_perceptron.weights[0]),  # NOT(x1) AND x2
        ("x2", "AND", and_perceptron.weights[1]),  # x2 -> AND
        ("x3", "OR", or_perceptron.weights[1]),  # x3 -> OR
        ("AND", "OR", or_perceptron.weights[0]),  # (NOT(x1) AND x2) -> OR
        ("OR", "y", 1.0),  # Финальный выход y
    ]

    # Добавляем рёбра в граф с атрибутом веса
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    # Ручное расположение узлов
    pos = {
        "x1": (-2, 2),
        "x2": (-2, 0),
        "x3": (-2, -2),
        "NOT": (0, 2),
        "AND": (2, 1),
        "OR": (4, 0),
        "y": (6, 0),
    }

    # Рисуем граф
    plt.figure(figsize=(10, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=1000,
        node_color="lightblue",
        font_size=8,
        font_weight="bold",
        arrows=True,
    )

    # Подписываем веса на рёбрах
    edge_labels = {(u, v): f"w={w:.2f}" for u, v, w in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Показываем граф
    plt.show()


# Визуализация сети
visualize_network()
