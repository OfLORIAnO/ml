import numpy as np

# Инициализация параметров
np.random.seed(3)
LEARNING_RATE = 0.1
index_list = [0, 1, 2, 3]

# Обучающие данные
x_train = np.array(
    [
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ]
)
y_train = np.array([0.0, 1.0, 1.0, 0.0])


# Инициализация весов
def neuron_w(input_count):
    weights = np.random.uniform(-1.0, 1.0, input_count + 1)
    return weights


# Добавляем два скрытых слоя и выходной слой
n_w = [neuron_w(2), neuron_w(2), neuron_w(3), neuron_w(3)]
n_y = [0, 0, 0, 0]
n_error = [0, 0, 0, 0]


# Прямой проход
def forward_pass(x):
    global n_y
    n_y[0] = np.tanh(np.dot(n_w[0], x))
    n_y[1] = np.tanh(np.dot(n_w[1], x))

    hidden_inputs = np.array([1.0, n_y[0], n_y[1]])
    n_y[2] = np.tanh(np.dot(n_w[2], hidden_inputs))

    output_inputs = np.array([1.0, n_y[2], n_y[1]])
    z_out = np.dot(n_w[3], output_inputs)
    n_y[3] = 1.0 / (1.0 + np.exp(-z_out))


# Обратный проход
def backward_pass(y_truth):
    global n_error
    error_prime = y_truth - n_y[3]
    n_error[3] = error_prime * n_y[3] * (1.0 - n_y[3])

    derivative = 1.0 - n_y[2] ** 2
    n_error[2] = n_w[3][1] * n_error[3] * derivative

    derivative = 1.0 - n_y[0] ** 2
    n_error[0] = n_w[2][1] * n_error[2] * derivative

    derivative = 1.0 - n_y[1] ** 2
    n_error[1] = (n_w[2][2] * n_error[2] + n_w[3][2] * n_error[3]) * derivative


# Обновление весов
def adjust_weights(x):
    global n_w
    n_w[0] += x * LEARNING_RATE * n_error[0]
    n_w[1] += x * LEARNING_RATE * n_error[1]

    hidden_inputs = np.array([1.0, n_y[0], n_y[1]])
    n_w[2] += hidden_inputs * LEARNING_RATE * n_error[2]

    output_inputs = np.array([1.0, n_y[2], n_y[1]])
    n_w[3] += output_inputs * LEARNING_RATE * n_error[3]


# Основной цикл обучения
all_correct = False
while not all_correct:
    all_correct = True
    np.random.shuffle(index_list)
    for i in index_list:
        forward_pass(x_train[i])
        backward_pass(y_train[i])
        adjust_weights(x_train[i])

        # Проверка правильности классификации
        if (y_train[i] < 0.5 and n_y[3] >= 0.5) or (y_train[i] >= 0.5 and n_y[3] < 0.5):
            all_correct = False
