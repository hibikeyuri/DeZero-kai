import numpy as np
import unittest


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # 1. 関数の取得
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            
            for x, gx in zip(f.inputs, gxs):
                x.grad = gx

                if x.creator is not None:
                    funcs.append(x.creator)


class Function:
    def __call__(self, *inputs):#1. アスタリスクを付ける
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)#2. asterik argument unpacking
        #this is unpacking. output is a tuple
        #if there is one input, then output a single value
        if not isinstance(ys, tuple):#3. タプルではない場合の追加対応
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]
        
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        
        #リストの要素が1つのときは最初の要素を返す
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def add(x0, x1):
    return Add()(x0, x1)


x = Variable(np.array(2.0))
y = Variable(np.array(3.0))
z = add(square(x), square(y))
z.backward()
print(z.data)
print(x.grad)
print(y.grad)