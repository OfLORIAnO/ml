import numpy as np

# Параметры нейронной сети
input_size = 2
hidden_layer1_size = 3
hidden_layer2_size = 3
output_size = 1
learning_rate = 0.1
epochs = 10001


# Функция активации и её производная
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


# Инициализация весов
def initialize_weights():
    np.random.seed(1)
    weights_input_hidden1 = np.random.uniform(-1, 1, (input_size, hidden_layer1_size))
    weights_hidden1_hidden2 = np.random.uniform(
        -1, 1, (hidden_layer1_size, hidden_layer2_size)
    )
    weights_hidden2_output = np.random.uniform(-1, 1, (hidden_layer2_size, output_size))
    return weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output


# Прямое распространение
def forward_propagation(
    x_train, weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output
):
    layer1_output = relu(np.dot(x_train, weights_input_hidden1))
    layer2_output = relu(np.dot(layer1_output, weights_hidden1_hidden2))
    final_output = sigmoid(np.dot(layer2_output, weights_hidden2_output))
    return layer1_output, layer2_output, final_output


# Вычисление ошибки
def compute_error(y_train, final_output):
    return y_train - final_output


# Обратное распространение
def backpropagation(
    error,
    final_output,
    layer2_output,
    layer1_output,
    weights_hidden2_output,
    weights_hidden1_hidden2,
):
    d_output = error * sigmoid_derivative(final_output)
    d_hidden2 = d_output.dot(weights_hidden2_output.T) * relu_derivative(layer2_output)
    d_hidden1 = d_hidden2.dot(weights_hidden1_hidden2.T) * relu_derivative(
        layer1_output
    )
    return d_output, d_hidden2, d_hidden1


# Обновление весов
def update_weights(
    weights_input_hidden1,
    weights_hidden1_hidden2,
    weights_hidden2_output,
    layer1_output,
    layer2_output,
    d_output,
    d_hidden2,
    d_hidden1,
    x_train,
):
    weights_hidden2_output += layer2_output.T.dot(d_output) * learning_rate
    weights_hidden1_hidden2 += layer1_output.T.dot(d_hidden2) * learning_rate
    weights_input_hidden1 += x_train.T.dot(d_hidden1) * learning_rate
    return weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output


# Основной цикл обучения
def train(x_train, y_train):
    # Инициализируем веса
    weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output = (
        initialize_weights()
    )

    for epoch in range(epochs):
        # Прямое распространение
        layer1_output, layer2_output, final_output = forward_propagation(
            x_train,
            weights_input_hidden1,
            weights_hidden1_hidden2,
            weights_hidden2_output,
        )

        # Печать промежуточных значений
        if epoch == 1000:
            print("Layer 1 Output:\n", layer1_output)
            print("Layer 2 Output:\n", layer2_output)
            print("Final Output:\n", final_output)
            print("---------------")

        # Вычисление ошибки
        error = compute_error(y_train, final_output)
        if epoch % 1000 == 0:
            print(f"Error at epoch {epoch}: {np.mean(np.abs(error))}")

        # Обратное распространение
        d_output, d_hidden2, d_hidden1 = backpropagation(
            error,
            final_output,
            layer2_output,
            layer1_output,
            weights_hidden2_output,
            weights_hidden1_hidden2,
        )

        # Обновление весов
        weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output = (
            update_weights(
                weights_input_hidden1,
                weights_hidden1_hidden2,
                weights_hidden2_output,
                layer1_output,
                layer2_output,
                d_output,
                d_hidden2,
                d_hidden1,
                x_train,
            )
        )

    return final_output


# Данные для обучения
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Запуск обучения
final_output = train(x_train, y_train)

# Проверка результатов
print("Final output after training:")
print(final_output)
