import numpy as np


# Класс для одного персептрона
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        # Инициализируем случайные веса, включая bias
        self.weights = np.random.randn(input_size + 1)
        self.learning_rate = learning_rate

    def predict(self, inputs):
        # Добавляем bias к входам
        inputs = np.append(1, inputs)  # 1 для bias
        # Вычисляем взвешенную сумму
        summation = np.dot(inputs, self.weights)
        # Применяем пороговую активацию (функция активации)
        return 1 if summation >= 0 else 0

    def train(self, X, y, epochs=100):
        # Обучаем персептрон
        for epoch in range(epochs):
            for inputs, target in zip(X, y):
                prediction = self.predict(inputs)
                # Вычисляем ошибку
                error = target - prediction
                # Обновляем веса
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error  # bias


# Данные для обучения персептрона (таблица истинности для AND)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Входные значения
y = np.array([0, 0, 0, 1])  # Ожидаемые результаты (AND)

# Создаем персептрон
perceptron = Perceptron(input_size=2)

# Обучаем персептрон (увеличим количество эпох до 100)
perceptron.train(X, y, epochs=100)

# Проверяем работу персептрона на обучающих данных
print("Результаты предсказаний после обучения:")
for inputs in X:
    print(f"Input: {inputs} -> Prediction: {perceptron.predict(inputs)}")
