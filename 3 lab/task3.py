import numpy as np

# Устанавливаем фиксированный seed для генератора случайных чисел для воспроизводимости результатов
np.random.seed(3)
LEARNING_RATE = 0.1
index_list = [0, 1, 2, 3]  # Список индексов для перемешивания обучающих примеров

# Обучающие данные для XOR, включающие значение смещения (первый элемент каждого входа равен 1.0)
x_train = np.array(
    [[1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0]]
)

y_train = np.array([0.0, 1.0, 1.0, 0.0])  # Ожидаемые выходы для XOR


# Функция для инициализации весов нейрона с заданным количеством входов
def neuron_w(input_count):
    weights = np.zeros(input_count + 1)  # Включаем вес для смещения
    for i in range(1, input_count + 1):
        weights[i] = np.random.uniform(-1.0, 1.0)  # Случайные веса от -1 до 1
    return weights


# Инициализация весов для трех нейронов (два скрытых и один выходной)
n_w = [neuron_w(2), neuron_w(2), neuron_w(2)]
n_y = [0.0, 0.0, 0.0]  # Список для хранения выходных значений нейронов
n_error = [0.0, 0.0, 0.0]  # Список для хранения значений ошибки для каждого нейрона


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


# Функция прямого прохода для вычисления выходов каждого нейрона
def forward_pass(x):
    global n_y
    # Вычисляем выходы первых двух нейронов с помощью tanh
    n_y[0] = np.tanh(np.dot(n_w[0], x))
    n_y[1] = np.tanh(np.dot(n_w[1], x))

    # Вычисляем входы для выходного нейрона (включая смещение)
    n2_inputs = np.array([1.0, n_y[0], n_y[1]])

    # Вычисляем выход выходного нейрона с помощью сигмоидальной функции
    z2 = np.dot(n_w[2], n2_inputs)
    n_y[2] = 1.0 / (1.0 + np.exp(-z2))


# Функция обратного прохода для вычисления ошибки на каждом слое
def backward_pass(y_truth):
    global n_error
    # Ошибка выходного нейрона
    error_prime = y_truth - n_y[2]
    derivative = n_y[2] * (1.0 - n_y[2])  # Производная сигмоиды
    n_error[2] = error_prime * derivative  # Ошибка выходного нейрона

    # Ошибка для скрытых нейронов, учитывая производную tanh
    derivative = 1.0 - n_y[0] ** 2
    n_error[0] = (
        n_w[2][1] * n_error[2] * derivative
    )  # Ошибка для первого скрытого нейрона
    derivative = 1.0 - n_y[1] ** 2
    n_error[1] = (
        n_w[2][2] * n_error[2] * derivative
    )  # Ошибка для второго скрытого нейрона


# Функция обновления весов на основе вычисленных ошибок
def adjust_weights(x):
    global n_w
    # Обновление весов для скрытых нейронов
    n_w[0] += x * LEARNING_RATE * n_error[0]
    n_w[1] += x * LEARNING_RATE * n_error[1]

    # Обновление весов выходного нейрона
    n2_inputs = np.array([1.0, n_y[0], n_y[1]])  # Входы для выходного нейрона
    n_w[2] += n2_inputs * LEARNING_RATE * n_error[2]


# Цикл обучения нейронной сети
all_correct = False
while not all_correct:  # Обучаем сеть до сходимости
    all_correct = True
    np.random.shuffle(index_list)  # Перемешиваем порядок обучающих примеров
    for i in index_list:  # Тренируемся на всех примерах
        forward_pass(x_train[i])  # Прямой проход
        backward_pass(y_train[i])  # Обратный проход
        adjust_weights(x_train[i])  # Корректировка весов
        show_learning()  # Отображаем обновленные веса

    # Проверка сходимости сети
    for i in range(len(x_train)):
        forward_pass(x_train[i])
        print(
            "x1 =",
            "%.1f" % x_train[i][1],
            ", x2 =",
            "%.1f" % x_train[i][2],
            ", y =",
            "%.4f" % n_y[2],
        )
        # Проверяем, правильно ли сеть классифицировала все примеры
        if ((y_train[i] < 0.5) and (n_y[2] >= 0.5)) or (
            (y_train[i] >= 0.5) and (n_y[2] < 0.5)
        ):
            all_correct = False
