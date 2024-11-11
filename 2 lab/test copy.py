def y1(x1, x2, x3):
    return int(((not (x1)) and (x2 != x3)) or x2)


def y2(x1, x2, x3):
    return int((not (x1) and x3) or x2)


def formatData(x1, x2, x3, y):
    return f"Input: [{x1} {x2} {x3}] -> Output: {y}"


print("Таблица истинности для функции (¬X1∧(X2⊕X3))∨X2")
for x1 in range(2):
    for x2 in range(2):
        for x3 in range(2):
            y1_result = formatData(x1, x2, x3, y1(x1, x2, x3))

            print(y1_result)
print(" __________________________")
print(" ")
print("Таблица истинности для функции ¬X1∧ X3∨X2")
for x1 in range(2):
    for x2 in range(2):
        for x3 in range(2):
            y2_result = formatData(x1, x2, x3, y2(x1, x2, x3))

            print(y2_result)
