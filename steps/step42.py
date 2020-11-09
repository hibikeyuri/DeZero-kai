if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F


# トイ・データセット
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
plt.scatter(x, y)

x, y = Variable(x), Variable(y) # 省略可能
W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

def predict(x):
    y = F.matmul(x, W) + b
    return y

def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)

lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)
    W.cleargrad()
    b.cleargrad()
    loss.backward()
    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b, loss)

print(x.shape)
y_final = predict(x)
print(y_final.shape)
#plt.scatter(x.data,y_final.data)
#plt.show()

