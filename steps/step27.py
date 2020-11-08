if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dezero import Variable, Function
from dezero.utils import _dot_var, _dot_func
from dezero.utils import plot_dot_graph
import numpy as np
import math


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx


def sin(x):
    return Sin()(x)

def my_sin(x, threshold=1e-150):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


#x = Variable(np.array(np.pi/4))
#y = sin(x)
#y.backward()

#print(y.data)
#print(x.grad)

x = Variable(np.array(np.pi/4))
y = my_sin(x)
y.backward()
y.name = 'y'
x.name = 'x'
print(y.data)
print(x.grad)
plot_dot_graph(y, verbose=False, to_file='taylor_sin_th=1e-150.png')