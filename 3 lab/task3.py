import numpy as np

# Инициализация параметров
np.random.seed(3)  # Фиксация случайного состояния
LEARNING_RATE = 0.1  # Скорость обучения
index_list = [0, 1, 2, 3]  # Индексы обучающих примеров

# Обучающие данные (входы и истинные значения)
x_train = np.array(
    [
        [1.0, -1.0, -1.0],  # x1 = 1.0, x2 = -1.0
        [1.0, -1.0, 1.0],  # x1 = 1.0, x2 = 1.0
        [1.0, 1.0, -1.0],  # x1 = -1.0, x2 = -1.0
        [1.0, 1.0, 1.0],  # x1 = -1.0, x2 = 1.0
    ]
)

y_train = np.array(
    [
        0.0,  # Целевое значение для первого примера
        1.0,  # Целевое значение для второго примера
        1.0,  # Целевое значение для третьего примера
        0.0,  # Целевое значение для четвертого примера
    ]
)


# Инициализация весов нейронов
def neuron_w(input_count):
    weights = np.zeros(input_count + 1)  # Включая вес смещения
    for i in range(1, input_count + 1):
        weights[i] = np.random.uniform(-1.0, 1.0)  # Рандомизация весов
    return weights


n_w = [neuron_w(2), neuron_w(2), neuron_w(2)]  # Веса всех нейронов
n_y = [0, 0, 0]  # Выходы нейронов
n_error = [0, 0, 0]  # Ошибки нейронов


# Функция прямого прохода
def forward_pass(x):
    global n_y
    n_y[0] = np.tanh(np.dot(n_w[0], x))  # Выход первого скрытого нейрона
    n_y[1] = np.tanh(np.dot(n_w[1], x))  # Выход второго скрытого нейрона

    n2_inputs = np.array([1.0, n_y[0], n_y[1]])  # Входы для выходного нейрона
    z2 = np.dot(n_w[2], n2_inputs)  # Линейная комбинация для выходного нейрона
    n_y[2] = 1.0 / (1.0 + np.exp(-z2))  # Сигмоидальная активация


# Функция обратного прохода
def backward_pass(y_truth):
    global n_error
    # Вычисление ошибки для выходного нейрона
    error_prime = y_truth - n_y[2]
    derivative = n_y[2] * (1.0 - n_y[2])  # Производная сигмоиды
    n_error[2] = error_prime * derivative  # Градиент

    # Ошибки для скрытых нейронов
    derivative = 1.0 - n_y[0] ** 2  # Производная tanh
    n_error[0] = n_w[2][1] * n_error[2] * derivative

    derivative = 1.0 - n_y[1] ** 2  # Производная tanh
    n_error[1] = n_w[2][2] * n_error[2] * derivative


# Функция обновления весов
def adjust_weights(x):
    global n_w
    n_w[0] += x * LEARNING_RATE * n_error[0]  # Обновление весов первого нейрона
    n_w[1] += x * LEARNING_RATE * n_error[1]  # Обновление весов второго нейрона

    n2_inputs = np.array([1.0, n_y[0], n_y[1]])  # Входы для выходного нейрона
    n_w[2] += (
        n2_inputs * LEARNING_RATE * n_error[2]
    )  # Обновление весов выходного нейрона


# Функция для отображения текущих весов каждого нейрона
def show_learning():
    print("Current weights:")
    for i, w in enumerate(n_w):
        print(
            "neuron ",
            i,
            ": w0 =",
            "%.2f" % w[0],
            ", w1 =",
            "%.2f" % w[1],
            ", w2 =",
            "%.2f" % w[2],
        )
    print("----------------")


# Основной цикл обучения
all_correct = False
while not all_correct:  # Повторять, пока сеть не обучится
    all_correct = True
    np.random.shuffle(index_list)  # Перемешивание обучающих данных
    for i in index_list:
        forward_pass(x_train[i])  # Прямой проход
        backward_pass(y_train[i])  # Обратный проход
        adjust_weights(x_train[i])  # Обновление весов
        show_learning()  # Печать текущих весов

        # Проверка правильности классификации
        if (y_train[i] < 0.5 and n_y[2] >= 0.5) or (y_train[i] >= 0.5 and n_y[2] < 0.5):
            all_correct = False
