import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        f = self.creator # 1. 関数の取得
        if f is not None:
            x = f.input # 2. 関数の入力を取得
            x.grad = f.backward(self.grad) # 3. 関数のメソッドを呼ぶ
            x.backward() # 自分より一つ前のbackwardメソッドを呼ぶ


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

assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x

"""
y.grad = np.array(1.0)

C = y.creator
b = C.input
b.grad = C.backward(y.grad)

B = b.creator
a = B.input
a.grad = B.backward(b.grad)

A = a.creator
x = A.input
x.grad = A.backward(a.grad)

print(x.grad)
"""

y.grad = np.array(1.0)
y.backward()

print(x.grad)
print(b.grad)
print(a.grad)
