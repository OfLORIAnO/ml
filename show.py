
import matplotlib.pyplot as plt
import numpy as np

# ([x1,x2,...xn], [y1,y2,...,yn])
# plt.plot(x, y)

# data = {
    # 'x': [1,2],
    # 'y': [1,2]
    # }
# plt.scatter('x', 'y', data=data)


def show_chart(x: list[int], w: list[int]):
    print(w)
    w0, w1, w2 = w[0], w[1], w[2] 
    for x1,x2 in x:
        print("------")
        print('w',w0, w1, w2)
        print('x',x1,x2)
        k = -(w0/w2) - (w1*x1)/w2 # угловой коэф
        print("k", k)
        
    pass

def show_learning(w):
    print('w0 =', '%5.2f' % w[0], ', w1 =', '%5.2f' % w[1], ', w2 =', '%5.2f' % w[2])