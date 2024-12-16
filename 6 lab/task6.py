import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import multiprocessing

# Установка начального зерна для воспроизводимости
np.random.seed(7)


# Загрузка датасета MNISTп
def read_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    images, labels = mnist["data"], mnist["target"].astype(int)

    # Разделение на обучающую и тестовую выборки
    train_images, train_labels = images[:60000], labels[:60000]
    test_images, test_labels = images[60000:], labels[60000:]

    return train_images, train_labels, test_images, test_labels


# Подготовка данных
train_images, train_labels, test_images, test_labels = read_mnist()
x_train = train_images.astype(np.float32)
x_test = test_images.astype(np.float32)
mean, stddev = np.mean(x_train), np.std(x_train)
x_train = (x_train - mean) / stddev
x_test = (x_test - mean) / stddev

# Создание one-hot представления меток
y_train = np.eye(10)[train_labels]
y_test = np.eye(10)[test_labels]


# Функция для сдвига изображения
def shift_image(image, direction):
    image_2d = image.reshape(28, 28)
    shifted_image = np.zeros_like(image_2d)

    if direction == "left":
        shifted_image[:, 1:] = image_2d[:, :-1]
    elif direction == "right":
        shifted_image[:, :-1] = image_2d[:, 1:]
    elif direction == "up":
        shifted_image[1:, :] = image_2d[:-1, :]
    elif direction == "down":
        shifted_image[:-1, :] = image_2d[1:, :]

    return shifted_image.reshape(784)


# Создание аугментированного датасета с предварительным выделением памяти
def create_augmented_data(x_train, y_train):
    directions = ["left", "right", "up", "down"]
    augmented_size = x_train.shape[0] * 5  # Оригинал + 4 направления
    augmented_x = np.zeros((augmented_size, 784), dtype=np.float32)
    augmented_y = np.zeros((augmented_size, 10), dtype=np.float32)

    # Копирование оригинальных данных
    augmented_x[::5] = x_train
    augmented_y[::5] = y_train

    for i, direction in enumerate(directions):
        augmented_x[i::5] = np.array([shift_image(img, direction) for img in x_train])
        augmented_y[i::5] = y_train

    return augmented_x, augmented_y


augmented_x_train, augmented_y_train = create_augmented_data(x_train, y_train)


# Определение класса модели
class Model:
    def __init__(
        self, hidden_neurons=25, epochs=20, learning_rate=0.01, model_name="Model"
    ):
        self.hidden_neurons = hidden_neurons
        self.EPOCHS = epochs
        self.LEARNING_RATE = learning_rate
        self.model_name = model_name  # Название модели
        # Инициализация весов с использованием случайных значений
        self.hidden_layer_w = np.random.uniform(-0.1, 0.1, (hidden_neurons, 785))
        self.output_layer_w = np.random.uniform(-0.1, 0.1, (10, hidden_neurons + 1))
        # Для хранения значений активации и ошибок
        self.hidden_layer_y = np.zeros(hidden_neurons)
        self.output_layer_y = np.zeros(10)
        self.hidden_layer_error = np.zeros(hidden_neurons)
        self.output_layer_error = np.zeros(10)
        # Для построения графика обучения
        self.chart_x = []
        self.chart_y_train = []
        self.chart_y_test = []

    def get_accuracy(self, x_test, y_test):
        correct = 0
        x_test_with_bias = np.hstack((np.ones((x_test.shape[0], 1)), x_test))
        for x, y in zip(x_test_with_bias, y_test):
            self.forward_pass(x)
            if np.argmax(self.output_layer_y) == np.argmax(y):
                correct += 1
        return correct / len(x_test)

    def forward_pass(self, x):
        # Вычисление активации скрытого слоя
        z_hidden = np.dot(self.hidden_layer_w, x)
        self.hidden_layer_y = np.tanh(z_hidden)
        # Добавление смещения для скрытого слоя
        hidden_with_bias = np.hstack(([1.0], self.hidden_layer_y))
        # Вычисление активации выходного слоя
        z_output = np.dot(self.output_layer_w, hidden_with_bias)
        self.output_layer_y = 1.0 / (1.0 + np.exp(-z_output))

    def backward_pass(self, x, y_truth):
        # Вычисление ошибки выходного слоя
        error_output = (
            -(y_truth - self.output_layer_y)
            * self.output_layer_y
            * (1 - self.output_layer_y)
        )
        self.output_layer_error = error_output

        # Вычисление ошибки скрытого слоя
        error_hidden = np.dot(self.output_layer_w[:, 1:].T, self.output_layer_error) * (
            1 - self.hidden_layer_y**2
        )
        self.hidden_layer_error = error_hidden

        # Обновление весов выходного слоя
        hidden_with_bias = np.hstack(([1.0], self.hidden_layer_y))
        self.output_layer_w -= self.LEARNING_RATE * np.outer(
            error_output, hidden_with_bias
        )

        # Обновление весов скрытого слоя
        self.hidden_layer_w -= self.LEARNING_RATE * np.outer(error_hidden, x)

    def show_learning(self, epoch_no, train_acc, test_acc):
        print(
            f"[{self.model_name}] Эпоха {epoch_no + 1}: Точность на обучении: {train_acc:.4f}, Точность на тесте: {test_acc:.4f}"
        )
        self.chart_x.append(epoch_no + 1)
        self.chart_y_train.append(1.0 - train_acc)
        self.chart_y_test.append(1.0 - test_acc)

    def plot_learning(self):
        plt.plot(self.chart_x, self.chart_y_train, "r-", label="Ошибка на обучении")
        plt.plot(self.chart_x, self.chart_y_test, "b-", label="Ошибка на тесте")
        plt.title(f"Обучение модели: {self.model_name}")
        plt.xlabel("Эпохи")
        plt.ylabel("Ошибка")
        plt.legend()
        plt.show()

    def train(self, x_train, y_train, x_test, y_test):
        # Добавление смещения к обучающим данным
        x_train_with_bias = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
        for epoch in range(self.EPOCHS):
            # Перемешивание данных для каждой эпохи
            indices = np.random.permutation(x_train.shape[0])
            correct_train = 0

            for idx in indices:
                x = x_train_with_bias[idx]
                y = y_train[idx]
                self.forward_pass(x)
                if np.argmax(self.output_layer_y) == np.argmax(y):
                    correct_train += 1
                self.backward_pass(x, y)

            # Вычисление точности на тестовых данных
            train_acc = correct_train / x_train.shape[0]
            test_acc = self.get_accuracy(x_test, y_test)

            self.show_learning(epoch, train_acc, test_acc)

        self.plot_learning()


# Функция для обучения модели
def train_model(model, x_train, y_train, x_test, y_test, model_name="Model"):
    model.train(x_train, y_train, x_test, y_test)
    accuracy = model.get_accuracy(x_test, y_test)
    print(f"Точность модели [{model_name}]: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    # Создание экземпляров моделей с различными параметрами
    base_model = Model(
        hidden_neurons=24, epochs=20, learning_rate=0.01, model_name="Base Model"
    )
    improved_model = Model(
        hidden_neurons=38, epochs=50, learning_rate=0.005, model_name="Improved Model"
    )
    augmented_model = Model(
        hidden_neurons=38, epochs=50, learning_rate=0.005, model_name="Augmented Model"
    )

    train_model(base_model, x_train, y_train, x_test, y_test, "Base Model")
    train_model(improved_model, x_train, y_train, x_test, y_test, "Improved Model")
    train_model(
        augmented_model,
        augmented_x_train,
        augmented_y_train,
        x_test,
        y_test,
        "Augmented Model",
    )

    # # Создание процессов для параллельного обучения моделей
    # processes = []
    # models = [
    #     (base_model, x_train, y_train, x_test, y_test, "Base Model"),
    #     (improved_model, x_train, y_train, x_test, y_test, "Improved Model"),
    #     (
    #         augmented_model,
    #         augmented_x_train,
    #         augmented_y_train,
    #         x_test,    #         y_test,
    #         "Augmented Model",
    #     ),
    # ]

    # for model_args in models:
    #     p = multiprocessing.Process(target=train_model, args=model_args)
    #     processes.append(p)
    #     p.start()

    # for p in processes:
    #     p.join()
