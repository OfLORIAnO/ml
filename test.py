import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx

# Устанавливаем начальные параметры
random.seed(100)
LEARNING_SPEED = 0.1

# Функции для рисования графиков


# Рисование графика для операций AND, OR, XOR
def draw_graph(w, func=None):
    x2 = 1 + 0.1
    x1 = 0 - 0.1
    y1 = -w[1] * x1 / w[2] - w[0] / w[2]
    y2 = -w[1] * x2 / w[2] - w[0] / w[2]
    plt.ylabel("x2")
    plt.xlabel("x1")
    plt.axis([-0.1, 1.1, -0.1, 1.1])

    if func is not None:
        for y in (0, 1):
            for x in (0, 1):
                res = func([1, x, y])
                if res == 1:
                    plt.plot(x, y, "r+")
                else:
                    plt.plot(x, y, "b_")

    plt.plot((x1, x2), (y1, y2))


# Рисование графика для операции NOT
def draw_not_graph(w, func=None):
    x0 = -w[0] / w[1]
    y1 = -0.1
    y2 = 1.1
    plt.xlabel("x1")
    plt.axis([-0.1, 1.1, -0.1, 1.1])

    if func is not None:
        for x in (0, 1):
            res = func([1, x])
            if res == 1:
                plt.plot(x, 0, "r+")
            else:
                plt.plot(x, 0, "b_")

    plt.plot((x0, x0), (y1, y2))


# Класс для создания и обучения персептронов
class Perceptron:
    def __init__(self, cnt: int):
        self.w = [random.uniform(-1, 1) for _ in range(cnt)]
        self.learning_w = []

    def compute(self, x: list):
        z = np.dot(x, self.w)
        z = np.sign(z)
        return 0 if z == -1 else 1

    def train(self, xs: list, ys: list):
        self.indices = list(range(len(xs)))
        all_correct = False
        while not all_correct:
            self.learning_w.append(self.w.copy())
            all_correct = True
            random.shuffle(self.indices)
            for i in self.indices:
                x = xs[i]
                y = ys[i]
                o = self.compute(x)
                if o != y:
                    all_correct = False
                    for j in range(0, len(self.w)):
                        self.w[j] += LEARNING_SPEED * (y - o) * x[j]

        print(f"Final weights for perceptron: {self.w}")


# Инициализация и тренировка персептронов для логических операций
andP = Perceptron(3)
andP.train([(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)], [0, 0, 0, 1])  # AND

orP = Perceptron(3)
orP.train([(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)], [0, 1, 1, 1])  # OR

notP = Perceptron(2)
notP.train([(1, 0), (1, 1)], [1, 0])  # NOT


# Функция для реализации XOR через AND, OR, NOT
def xor_compute(x):
    or_result = orP.compute(x)
    and_result = andP.compute(x)
    not_and_result = notP.compute([1, and_result])
    return andP.compute([1, or_result, not_and_result])


# Вспомогательные функции для логики
def and_op(x1, x2):
    return andP.compute([1, x1, x2])


def or_op(x1, x2):
    return orP.compute([1, x1, x2])


def not_op(x):
    return notP.compute([1, x])


def xor_op(x1, x2):
    return xor_compute([1, x1, x2])


# Окончательная функция для вычисления результата
def final_compute(x1, x2, x3):
    condition1 = and_op(x1, x2)
    condition2 = not_op(or_op(x2, x3))
    inner_result = or_op(condition1, condition2)
    return xor_op(inner_result, x1)


# Рисование графиков для каждой логической операции
for i in andP.learning_w:
    draw_graph(i, andP.compute)
plt.show()
draw_graph(andP.w, andP.compute)
plt.show()

for i in orP.learning_w:
    draw_graph(i, orP.compute)
plt.show()
draw_graph(orP.w, orP.compute)
plt.show()

for i in notP.learning_w:
    draw_not_graph(i, notP.compute)
plt.show()
draw_not_graph(notP.w, notP.compute)
plt.show()

# Вывод таблицы истинности
print("x1 x2 x3 | Result")
for x in (0, 1):
    for y in (0, 1):
        for z in (0, 1):
            print(f"{x}  {y}  {z}  |   {final_compute(x, y, z)}")

import networkx as nx
import matplotlib.pyplot as plt


def visualize_network() -> None:
    """
    Визуализирует структуру логической функции с весами на рёбрах.
    """
    # Создаем граф
    G = nx.DiGraph()

    # Добавляем вершины для всех промежуточных операций
    G.add_nodes_from(["x1", "x2", "x3", "AND1", "OR1", "NOT1", "AND2", "OR2", "y"])

    # Добавляем рёбра с весами (веса обученные персептронами)
    edges = [
        ("x1", "AND1", andP.w[0]),  # x1 -> AND1
        ("x2", "AND1", andP.w[1]),  # x2 -> AND1
        ("x3", "OR1", orP.w[2]),  # x3 -> OR1
        ("x2", "OR1", orP.w[0]),  # x2 -> OR1
        ("OR1", "NOT1", notP.w[1]),  # OR1 -> NOT1
        ("AND1", "OR2", orP.w[1]),  # AND1 -> OR2
        ("NOT1", "AND2", andP.w[1]),  # NOT1 -> AND2
        ("x1", "AND2", andP.w[0]),  # x1 -> AND2
        ("OR2", "y", orP.w[1]),  # OR2 -> y
        ("AND2", "y", orP.w[2]),  # AND2 -> y
    ]

    # Добавляем рёбра в граф с атрибутом веса
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    # Ручное расположение узлов (можно подправить для лучшей визуализации)
    pos = {
        "x1": (-2, 2),
        "x2": (-2, 0),
        "x3": (-2, -2),
        "AND1": (0, 1),
        "OR1": (0, -1),
        "NOT1": (2, -1),
        "AND2": (2, 1),
        "OR2": (4, 0),
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
