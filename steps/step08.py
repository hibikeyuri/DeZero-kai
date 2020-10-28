import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # 1. 関数の取得
            x, y = f.input, f.output # 2. 関数の入出力を取得
            x.grad = f.backward(y.grad) # 3. backwardメソッドを呼ぶ

            if x.creator is not None:
                funcs.append(x.creator) #　1つ前の関数をリストに追加

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self) # 出力変数に生みの親を覚えさせる
        self.input = input # 入力された変数を覚える
        self.output = output # 出力の覚える
        return output
    
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
y.backward()
print(x.grad)