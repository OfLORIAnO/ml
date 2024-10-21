import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# Класс для одного персептрона
class Perceptron:
    def __init__(self, input_size: int, learning_rate: float = 0.1) -> None:
        self.weights = np.random.randn(input_size + 1)  # Включаем bias
        self.learning_rate = learning_rate

    def predict(self, inputs: np.ndarray) -> int:
        inputs = np.append(1, inputs)  # 1 для bias
        summation = np.dot(inputs, self.weights)
        return 1 if summation >= 0 else 0

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> None:
        for epoch in range(epochs):
            for inputs, target in zip(X, y):
                prediction = self.predict(inputs)
                error = target - prediction
                if error != 0:
                    self.weights[1:] += self.learning_rate * error * inputs
                    self.weights[0] += self.learning_rate * error  # bias


# Создаем персептроны для логических операций
and_perceptron = Perceptron(input_size=2)
or_perceptron = Perceptron(input_size=2)
not_perceptron = Perceptron(input_size=1)

# Данные для обучения персептронов
X_and_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Входы для AND и OR
X_not = np.array([[0], [1]])  # Входы для NOT
y_and = np.array([0, 0, 0, 1])  # Результаты для AND
y_or = np.array([0, 1, 1, 1])  # Результаты для OR
y_not = np.array([1, 0])  # Результаты для NOT

# Обучаем персептроны
and_perceptron.train(X_and_or, y_and, epochs=100)
or_perceptron.train(X_and_or, y_or, epochs=100)
not_perceptron.train(X_not, y_not, epochs=100)


# Логическая функция: (NOT(x1) AND x2) OR x3
def logical_function(x1, x2, x3):
    not_result = not_perceptron.predict([x1])  # NOT(x1)
    and_result = and_perceptron.predict([not_result, x2])  # NOT(x1) AND x2
    final_result = or_perceptron.predict([and_result, x3])  # (NOT(x1) AND x2) OR x3
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


# Теперь создаем граф с networkx для визуализации логической функции
def visualize_network():
    # Создаем граф
    G = nx.DiGraph()

    # Добавляем вершины
    G.add_nodes_from(["x1", "x2", "x3", "NOT", "AND", "OR", "y"])

    # Добавляем рёбра с весами (веса условные, чтобы показать структуру)
    edges = [
        ("x1", "NOT", not_perceptron.weights[0]),  # NOT(x1)
        ("NOT", "AND", and_perceptron.weights[0]),  # NOT(x1) AND x2
        ("x2", "AND", and_perceptron.weights[1]),
        ("x3", "OR", or_perceptron.weights[1]),  # x3 -> OR
        ("AND", "OR", or_perceptron.weights[0]),  # (NOT(x1) AND x2) -> OR
        ("OR", "y", 1.0),  # Финальный выход
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
