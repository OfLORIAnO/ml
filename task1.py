import random
from show import show_chart, show_learning


# Определяем переменные, необходимые для процесса обучения
random.seed(7) # Чтобы обеспечить повторяемость
LEARNING_RATE = 0.1
# Определяем обучающие примеры 
x_train: list[tuple[int]] = [(1.0, 1.9 ,-1.7), (1.0, 0.5,-0.8), (1.0, -.6, .7), (1.0, -.8, .8), (1.0, -1.9, 1.7)] # Входы
y_train: list[int] = [1.0, 1.0, 1.0, -1.0] # Выход (истина)

index_list = [i for i in range(len(x_train))] # Чтобы сделать порядок случайным
# Определяем веса персептрона
w = [0.9, -0.6, -0.5] # Инициализируем «случайными» числами

# Печатаем начальные значения весов
show_learning(w)

# Значение первого элемента вектора x должно быть равно 1
# Для нейрона с n входами длины w and x должны быть равны n+1
def compute_output(w, x):
    z = 0.0
    for i in range(len(w)):
        z += x[i] * w[i] # Вычисление суммы взвешенных входов
    if z < 1.7: # Применение знаковой функции
        return -1
    else:
        return 1


# Цикл обучения персептрона
all_correct = False
while not all_correct:
    all_correct = True
    random.shuffle(index_list) # Сделать порядок случайным
    print("w", w)
    for i in index_list:
        x = x_train[i]
        y = y_train[i]
        p_out = compute_output(w, x) # Функция персептрона

        if y != p_out: # Обновить веса, когда неправильно
            for j in range(0, len(w)):
                w[j] += (y * LEARNING_RATE * x[j])
            all_correct = False
            show_learning(w) # Показать обновлённые веса

# x1 = [[x[1],x[2]] for x in x_train]
# show_chart(x1, w)