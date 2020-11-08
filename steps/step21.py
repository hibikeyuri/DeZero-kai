import numpy as np
import weakref
import contextlib


class Variable:
    __array_priority__ = 200

    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' '*9)
        return 'variable(' + p +')'
    
    def __mul__(self, other):
        return mul(self, other)
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def backward(self, retain_grad=False): # when a varaible do not retain the grad, set 0 to varible.grad
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
            gys = [output().grad for output in f.outputs]
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
            
            if not retain_grad:# when no retain grad, set 0 to variable grad
                for y in f.outputs:
                    y().grad = None

        
    def cleargrad(self):
        self.grad = None
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype


class Function:
    def __call__(self, *inputs):#1. アスタリスクを付ける
        inputs =  [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)#2. asterik argument unpacking
        #this is unpacking. output is a tuple
        #if there is one input, then output a single value
        if not isinstance(ys, tuple):#3. タプルではない場合の追加対応
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)

        self.inputs = inputs# 参照
        self.outputs = outputs
        self.outputs = [weakref.ref(output) for output in outputs]#using weakref to reduce memory consumption
        
        #リストの要素が1つのときは最初の要素を返す
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


class Config:
    enable_backprop = True


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


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
    x1 = as_array(x1)
    return Add()(x0, x1)

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

@contextlib.contextmanager
def config_test():
    print('start') # 前処理
    try:
        yield
    finally:
        print('done')# 後処理

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

Variable.__add__ = add
Variable.__radd__ = add
Variable.__mul__ = mul
Variable.__rmul__ = mul

x = Variable(np.array(2.0))
y = x + np.array(3.0)
print(y)
y = x + 2
print(y)
y = x + 2.0
print(y)
y = 2 + x
print(y)
y = 2.0 + x
print(y)
x = Variable(np.array(2.0))
y = 3.0 * x + 1.0
print(y)

x = Variable(np.array([1.0]))
y = np.array([2.0]) + x
print(y)