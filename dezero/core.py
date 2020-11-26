import weakref
import numpy as np
import contextlib
import dezero


# =============================================================================
# Config
# =============================================================================
class Config:
    enable_backprop = True


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


# =============================================================================
# Variable / Function
# =============================================================================
class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
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
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)

    def transpose(self):
        return dezero.functions.transpose(self)
    
    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def backward(self, retain_grad=False, create_graph=False): # when a varaible do not retain the grad, set 0 to varible.grad
        if self.grad is None:
            #self.grad = np.ones_like(self.data)
            self.grad = Variable(np.ones_like(self.data))

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
    
    @property
    def T(self):
        return dezero.functions.transpose(self)
        

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


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


# =============================================================================
# 四則演算 / 演算子のオーバーロード
# =============================================================================
class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        #this add function use the feature of the numpy add
        #it can automatically broadcast one array to the others
        y = x0 + x1
        return y
    
    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy
        gx1 = -gy
        if self.x0.shape != self.x1.shape: # for broadcast
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        #x0, x1 = self.inputs[0].data, self.inputs[1].data
        x0, x1 = self.inputs# gy and x1 are Variable instance
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape: # for broadcast
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)

        return gx0, gx1# when using * overload operator, it calls Function.__call__()


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    
    def backward(self, gy):
        #x0, x1 = self.inputs[0].data, self.inputs[1].data
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:  # for broadcast
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Pow(Function):
    def __init__(self, c):
        self.c = c
    
    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        #x = self.inputs[0].data
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


def neg(x):
    return Neg()(x)


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


def square(x):
    return Square()(x)


def pow(x, c):
    return Pow(c)(x)


class Parameter(Variable):
    pass


def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    #Variable.__div__ = div
    #Variable.__rdiv__ = rdiv
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__getitem__ = dezero.functions.get_item


#You got a dream, you gotta protect it. People can't do something themselves, they wanna
#tell you can't do it. If you want something, go get it. Period.

# I'M FUCKING TIRED

# I want to make a lot of moeny!
