def y1(x1, x2, x3):
    return (not (x1) and ((not (x1) and x2) or (x1 and not (x2)))) or x3


def y2(x1, x2, x3):
    return (not (x1) and x2) or x3


for x1 in range(2):
    for x2 in range(2):
        for x3 in range(2):
            print(x1, x2, x3, "f1:", y1(x1, x2, x3), "f2:", y2(x1, x2, x3))
