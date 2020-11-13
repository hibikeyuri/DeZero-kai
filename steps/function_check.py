import numpy as np



def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """Reshape gradient appropriately for dezero.functions.sum's backward.
    Args:
        gy (dezero.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.
    Returns:
        dezero.Variable: Gradient variable which is reshaped appropriately
    """
    ndim = len(x_shape)
    print("ndim is : ", ndim)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)
    
    if not (ndim==0 or tupled_axis is None or keepdims):
        print('the output gradient is no scalar!')
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        print('actual axis: ', actual_axis)
        print('befor list gy shape ', gy.shape)
        shape = list(gy.shape)
        print('after list gy shape ', shape)
        for a in sorted(actual_axis):
            #let 1 number to add position a
            #ex: [1] -> (0, 1) -> [1, 1] 
            shape.insert(a, 1)
        
    else:
        print('HAH! ')
        shape = gy.shape
    
    print('final shape ', shape)
    gy = gy.reshape(shape)
    return gy

gy = np.array([5, 7, 9]) # shape is (2, )
gy = np.array([1])
print(gy.shape)
x_shape = (2, 3)
gy = reshape_sum_backward(gy, x_shape, 0, keepdims=False)
print(gy.shape)

test = [2, 3]
test.insert(2, 1)
print(test)