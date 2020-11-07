import numpy as np
import weakref


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        see_set = set()

        def add_func(f):
            if f not in see_set:
                funcs.append(f)
                see_set.add(f)
                funcs.sort(key=lambda x: x.generation)
            
        add_func(self.creator)

        while funcs:
            f = funcs.pop() # 1. 関数の取得
            #gys = [output.grad for output in f.outputs]
            gys = [output().grad for output in outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)
        
    def cleargrad(self):
        self.grad = None


class Function:
    def __call__(self, *inputs):#1. アスタリスクを付ける
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)#2. asterik argument unpacking
        #this is unpacking. output is a tuple
        #if there is one input, then output a single value
        if not isinstance(ys, tuple):#3. タプルではない場合の追加対応
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        self.outputs = [weakref.ref(output) for output in outputs]#using weakref to reduce memory consumption
        
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


for i in range(10):
    x = Variable(np.random.randn(10000))
    y = square(square(square(x)))
    print("|iter times: {}|".format(i))