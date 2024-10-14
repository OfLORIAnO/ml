import random
from show import visualize_training_process, show_learning
from typing import List

from init import x_train, y_train, seed_number, LEARNING_RATE, w


# Определяем переменные, необходимые для процесса обучения
random.seed(seed_number)  # Чтобы обеспечить повторяемость
# Определяем обучающие примеры

index_list = [i for i in range(len(x_train))]  # Чтобы сделать порядок случайным

# Печатаем начальные значения весов
show_learning(w)


# Значение первого элемента вектора x должно быть равно 1
# Для нейрона с n входами длины w and x должны быть равны n+1
def compute_output(w, x):
    z = 0.0
    for i in range(len(w)):
        z += x[i] * w[i]  # Вычисление суммы взвешенных входов
    if z < 0:  # Применение знаковой функции
        return -1
    else:
        return 1


learn_process: List[List[float]] = []
# Цикл обучения персептрона
all_correct = False
while not all_correct:
    all_correct = True
    random.shuffle(index_list)  # Сделать порядок случайным
    for i in index_list:
        x = x_train[i]
        y = y_train[i]
        p_out = compute_output(w, x)  # Функция персептрона

        if y != p_out:  # Обновить веса, когда неправильно
            for j in range(0, len(w)):
                w[j] += y * LEARNING_RATE * x[j]
            all_correct = False
            show_learning(w)  # Показать обновлённые веса
            learn_process.append(w.copy())


visualize_training_process(y_train, x_train, w, learn_process)
