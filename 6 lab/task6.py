import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import multiprocessing

np.random.seed(7)


# Load MNIST dataset
def read_mnist():
    mnist = fetch_openml("mnist_784", version=1)
    images = mnist.data.to_numpy()
    labels = mnist.target.to_numpy()

    train_images = images[:60000]
    train_labels = labels[:60000].astype(int)
    test_images = images[60000:]
    test_labels = labels[60000:].astype(int)

    return train_images, train_labels, test_images, test_labels


# Prepare data
train_images, train_labels, test_images, test_labels = read_mnist()
x_train = train_images.reshape(60000, 784)
x_test = test_images.reshape(10000, 784)
mean, stddev = np.mean(x_train), np.std(x_train)
x_train = (x_train - mean) / stddev
x_test = (x_test - mean) / stddev

y_train = np.zeros((60000, 10))
y_test = np.zeros((10000, 10))
for i, y in enumerate(train_labels):
    y_train[i][y] = 1
for i, y in enumerate(test_labels):
    y_test[i][y] = 1


# Shift image function
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


# Create augmented dataset
augmented_x_train = []
augmented_y_train = []

for idx, image in enumerate(x_train):
    augmented_x_train.append(image)  # Original image
    augmented_y_train.append(y_train[idx])  # Original label

    # Add shifted images
    for direction in ["left", "right", "up", "down"]:
        augmented_x_train.append(shift_image(image, direction))
        augmented_y_train.append(y_train[idx])

augmented_x_train = np.array(augmented_x_train)
augmented_y_train = np.array(augmented_y_train)


# Model class definition
class Model:
    def __init__(self, hidden_neurons=25, epochs=20, learning_rate=0.01):
        self.hidden_neurons = hidden_neurons
        self.EPOCHS = epochs
        self.LEARNING_RATE = learning_rate
        self.hidden_layer_w = self.layer_w(25, 784)
        self.hidden_layer_y = np.zeros(25)
        self.hidden_layer_error = np.zeros(25)
        self.output_layer_w = self.layer_w(10, 25)
        self.output_layer_y = np.zeros(10)
        self.output_layer_error = np.zeros(10)
        self.chart_x = []
        self.chart_y_train = []
        self.chart_y_test = []

    def get_accuracy(self, x_test, y_test):
        correct_results = 0
        x_test_with_bias = np.concatenate(
            (np.ones((x_test.shape[0], 1)), x_test), axis=1
        )  # Bias добавляется один раз для всех данных
        for idx in range(len(x_test)):
            x = x_test_with_bias[idx]
            self.forward_pass(x)
            if self.output_layer_y.argmax() == y_test[idx].argmax():
                correct_results += 1
        accuracy = correct_results / len(x_test)
        return accuracy

    def layer_w(self, neuron_count, input_count):
        weights = np.zeros((neuron_count, input_count + 1))
        for i in range(neuron_count):
            for j in range(1, input_count + 1):
                weights[i][j] = np.random.uniform(-0.1, 0.1)
        return weights

    def forward_pass(self, x):
        for i, w in enumerate(self.hidden_layer_w):
            z = np.dot(w, x)
            self.hidden_layer_y[i] = np.tanh(z)

        hidden_output_array = np.concatenate((np.array([1.0]), self.hidden_layer_y))

        for i, w in enumerate(self.output_layer_w):
            z = np.dot(w, hidden_output_array)
            self.output_layer_y[i] = 1.0 / (1.0 + np.exp(-z))

    def backward_pass(self, y_truth):
        for i, y in enumerate(self.output_layer_y):
            error_prime = -(y_truth[i] - y)
            derivative = y * (1.0 - y)
            self.output_layer_error[i] = error_prime * derivative

        for i, y in enumerate(self.hidden_layer_y):
            error_weights = [w[i + 1] for w in self.output_layer_w]
            derivative = 1.0 - y**2
            weighted_error = np.dot(np.array(error_weights), self.output_layer_error)
            self.hidden_layer_error[i] = weighted_error * derivative

    def adjust_weights(self, x):
        for i, error in enumerate(self.hidden_layer_error):
            self.hidden_layer_w[i] -= x * self.LEARNING_RATE * error

        hidden_output_array = np.concatenate((np.array([1.0]), self.hidden_layer_y))

        for i, error in enumerate(self.output_layer_error):
            self.output_layer_w[i] -= hidden_output_array * self.LEARNING_RATE * error

    def show_learning(self, epoch_no, train_acc, test_acc):
        print(
            f"Epoch {epoch_no + 1}: Training Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}"
        )
        self.chart_x.append(epoch_no + 1)
        self.chart_y_train.append(1.0 - train_acc)
        self.chart_y_test.append(1.0 - test_acc)

    def plot_learning(self):
        plt.plot(self.chart_x, self.chart_y_train, "r-", label="Training Error")
        plt.plot(self.chart_x, self.chart_y_test, "b-", label="Test Error")
        plt.axis([0, len(self.chart_x), 0.0, 1.0])
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.legend()
        plt.show()

    def train(self, x_train, y_train, x_test, y_test):
        x_train_with_bias = np.concatenate(
            (np.ones((x_train.shape[0], 1)), x_train), axis=1
        )  # Bias добавляем один раз
        index_list = list(range(len(x_train)))

        for epoch in range(self.EPOCHS):
            np.random.shuffle(index_list)
            correct_training_results = 0

            for idx in index_list:
                x = x_train_with_bias[idx]  # Используем данные с bias
                self.forward_pass(x)
                if self.output_layer_y.argmax() == y_train[idx].argmax():
                    correct_training_results += 1
                self.backward_pass(y_train[idx])
                self.adjust_weights(x)

            correct_test_results = 0
            for idx in range(len(x_test)):
                x = np.concatenate((np.array([1.0]), x_test[idx]))  # bias для теста
                self.forward_pass(x)
                if self.output_layer_y.argmax() == y_test[idx].argmax():
                    correct_test_results += 1

            self.show_learning(
                epoch,
                correct_training_results / len(x_train),
                correct_test_results / len(x_test),
            )

        self.plot_learning()


# Функция для обучения модели
def train_model(model, x_train, y_train, x_test, y_test):
    model.train(x_train, y_train, x_test, y_test)
    accuracy = model.get_accuracy(x_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    # Создание экземпляров моделей
    base_model = Model(hidden_neurons=50, epochs=30, learning_rate=0.01)
    improved_model = Model(hidden_neurons=100, epochs=50, learning_rate=0.005)
    augmented_model = Model(hidden_neurons=100, epochs=50, learning_rate=0.005)

    # Используем multiprocessing для параллельного обучения
    processes = []

    # Создаем процессы для обучения каждой модели
    p1 = multiprocessing.Process(
        target=train_model, args=(base_model, x_train, y_train, x_test, y_test)
    )
    p2 = multiprocessing.Process(
        target=train_model, args=(improved_model, x_train, y_train, x_test, y_test)
    )
    p3 = multiprocessing.Process(
        target=train_model,
        args=(augmented_model, augmented_x_train, augmented_y_train, x_test, y_test),
    )

    # Запуск процессов
    processes.append(p1)
    processes.append(p2)
    processes.append(p3)

    for p in processes:
        p.start()  # Запуск процесса

    for p in processes:
        p.join()  # Ожидание завершения всех процессов
