import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List


def visualize_training_process(
    labels: List[float],
    features: List[Tuple[float, float, float]],
    final_weights: List[float],
    weights_history: List[List[float]],
):
    # Визуализация изменений разделяющих линий во время обучения
    plt.figure(figsize=(10, 6))

    # Отображение обучающих данных с разными классами
    for i, (f1, f2) in enumerate([(f[1], f[2]) for f in features]):
        marker_style = "+" if labels[i] == 1 else "_"
        plt.scatter(
            f1, f2, color="red" if labels[i] == 1 else "blue", marker=marker_style
        )

    # Показ изменения разделяющих линий по мере обучения
    x_range = np.linspace(-2, 2, 100)

    # Список цветов для визуализации прогресса
    colors = ["#34568B", "#88B04B", "#F7CAC9", "#92A8D1"]
    num_colors = len(colors)

    # Прорисовка линий для каждой итерации обучения
    for iteration, current_weights in enumerate(weights_history):
        selected_color = colors[iteration % num_colors]
        y_values = [
            -(current_weights[0] + current_weights[1] * x) / current_weights[2]
            for x in x_range
        ]
        plt.plot(x_range, y_values, color=selected_color, linewidth=2)

    # Финальная линия после обучения
    final_y_values = -(final_weights[0] + final_weights[1] * x_range) / final_weights[2]
    plt.plot(
        x_range, final_y_values, color="green", label="Финальная модель", linewidth=3
    )

    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Эволюция разделяющей линии в процессе обучения")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Визуализация финальной разделяющей линии после обучения
    plt.figure(figsize=(10, 6))

    # Отображение точек данных с соответствующими метками
    for i, (f1, f2) in enumerate([(f[1], f[2]) for f in features]):
        marker_style = "+" if labels[i] == 1 else "_"
        plt.scatter(
            f1, f2, color="red" if labels[i] == 1 else "blue", marker=marker_style
        )

    # Финальная линия разделения: w0 + w1 * x1 + w2 * x2 = 0
    final_y_values = -(final_weights[0] + final_weights[1] * x_range) / final_weights[2]
    plt.plot(x_range, final_y_values, color="green")

    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Финальная разделяющая линия после обучения")
    plt.grid(True)
    plt.show()


def show_learning(w):
    print("w0 =", "%5.2f" % w[0], ", w1 =", "%5.2f" % w[1], ", w2 =", "%5.2f" % w[2])
