def y1(x1, x2, x3):
    return int(((not (x1)) and (x2 != x3)) or x2)


def y2(x1, x2, x3):
    return int((not (x1)) and ((x2 or x3) and ((not (x2)) or (not (x3)))) or x2)


def y3(x1, x2, x3):
    return int(((not (x1)) and x2) or ((not (x1)) and x3) or x2)


def y4(x1, x2, x3):
    return int((not (x1) and x3) or x2)


def F(x1, x2, x3):
    return int(((not (x1)) and (x2 != x3)) or x2)


def formatData(x1, x2, x3, y):
    return f"Input: [{x1} {x2} {x3}] -> Output: {y}"


for x1 in range(2):
    for x2 in range(2):
        for x3 in range(2):
            y1_result = y1(x1, x2, x3)
            y2_result = y2(x1, x2, x3)
            y3_result = y3(x1, x2, x3)
            y4_result = y4(x1, x2, x3)
            F_result = F(x1, x2, x3)
            print(
                x1,
                x2,
                x3,
                "    f1:",
                y1_result,
                "f2:",
                y2_result,
                "f3:",
                y3_result,
                "f4:",
                y4_result,
                "f4:",
                F_result,
                y1_result == y2_result == y3_result == y4_result,
            )
