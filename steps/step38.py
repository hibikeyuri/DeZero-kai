if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import dezero.functions as F


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(x.grad)


#args feature test
#def res(*shape):
#    if len(shape)==1 :print(shape, shape[0])

#res((1, 0))
#res([1, 0])
#res(1, 0)

x1 = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
print(x1.grad)
y1 = F.transpose(x1)
y1.backward(retain_grad=True)

print(y1.grad)
print(x1.grad)

#x = Variable(np.random.rand(2, 3))
#print(x)
#x = x.transpose()
#print(x)
#x = x.T
#print(x)