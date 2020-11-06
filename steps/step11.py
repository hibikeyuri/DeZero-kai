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
            x, y = f.input, f.output # 2. 関数の入出力を取得
            x.grad = f.backward(y.grad) # 3. backwardメソッドを呼ぶ

            if x.creator is not None:
                funcs.append(x.creator) #　1つ前の関数をリストに追加


class Function:
    def __call__(self, inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]
        
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs
    
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y, )


xs = [Variable(np.array(2)), Variable(np.array(3))] # リストとして準備
f = Add()
ys = f(xs) # ysはタプル
y = ys[0]
print(y.data)
