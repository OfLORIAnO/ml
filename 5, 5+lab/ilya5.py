import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. Загрузка и подготовка данных
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)
X = X / 255.0
y_binary = (y == 4).astype(np.float32)

# Разделяем данные на обучающую и тестовую выборки с сохранением пропорций классов
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# Преобразуем y_train и y_test в векторы-столбцы
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# 2. Определение архитектуры нейронной сети
input_size = X_train.shape[1]
hidden_size = 128
output_size = 1
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
b2 = np.zeros((1, output_size))


# 3. Реализация функций активации и потерь
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_loss(y_true, y_pred, W1, W2, lambda_reg):
    m = y_true.shape[0]
    epsilon = 1e-15
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    cross_entropy_loss = -(1 / m) * np.sum(
        y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon)
    )
    L2_loss = (lambda_reg / (2 * m)) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    loss = cross_entropy_loss + L2_loss
    return loss


# 4. Обучение нейронной сети с использованием мини-батчей и регуляризации
learning_rate = 0.01
num_epochs = 5000
batch_size = 128
m = X_train.shape[0]
num_batches = m // batch_size
lambda_reg = 0.001

for epoch in range(num_epochs):
    permutation = np.random.permutation(m)
    X_train_shuffled = X_train[permutation]
    y_train_shuffled = y_train[permutation]

    epoch_loss = 0

    for batch in range(num_batches):
        start = batch * batch_size
        end = start + batch_size
        X_batch = X_train_shuffled[start:end]
        y_batch = y_train_shuffled[start:end]

        # Прямое распространение
        Z1 = np.dot(X_batch, W1) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = sigmoid(Z2)

        # Вычисляем потерю
        loss = compute_loss(y_batch, A2, W1, W2, lambda_reg)
        epoch_loss += loss

        # Обратное распространение
        dZ2 = A2 - y_batch
        dW2 = (1 / batch_size) * np.dot(A1.T, dZ2) + (lambda_reg / m) * W2
        db2 = (1 / batch_size) * np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * (1 - np.tanh(Z1) ** 2)
        dW1 = (1 / batch_size) * np.dot(X_batch.T, dZ1) + (lambda_reg / m) * W1
        db1 = (1 / batch_size) * np.sum(dZ1, axis=0, keepdims=True)

        # Обновление параметров
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    epoch_loss /= num_batches
    print(f"Эпоха {epoch + 1}/{num_epochs}, Потеря: {epoch_loss:.4f}")


# 5. Тестирование нейронной сети
def predict(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    predictions = (A2 > 0.5).astype(int)
    return predictions


y_pred = predict(X_test, W1, b1, W2, b2)

# Оцениваем метрики
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Точность: {accuracy * 100:.2f}%")
print(f"Точность положительных предсказаний (Precision): {precision * 100:.2f}%")
print(f"Полнота (Recall): {recall * 100:.2f}%")

# 6. Вывод результатов
num_examples = 5
examples = X_test[:num_examples]
true_labels = y_test[:num_examples]
pred_labels = y_pred[:num_examples]
plt.figure(figsize=(10, 2))
for i in range(num_examples):
    plt.subplot(1, num_examples, i + 1)
    plt.imshow(examples[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
    plt.title(
        f"Истина: {int(true_labels[i][0])}\nПредсказание: {int(pred_labels[i][0])}"
    )
plt.show()
