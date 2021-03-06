is_simple_core = False #True for core_simple

if is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import Function
    from dezero.core_simple import using_config
    from dezero.core_simple import no_grad
    from dezero.core_simple import as_array
    from dezero.core_simple import as_variable
    from dezero.core_simple import setup_variable

else:
    from dezero.core import Variable
    from dezero.core import Function
    from dezero.core import using_config
    from dezero.core import no_grad
    from dezero.core import as_array
    from dezero.core import as_variable
    from dezero.core import setup_variable
    from dezero.layers import Layer # Basic Layer to restore Parameters and multiple Layers
    from dezero.models import Model # inherit the Layers, it can contain the model which you want to make 


    import dezero.functions
    import dezero.layers
    import dezero.models
    import dezero.optimizers
    import dezero.datasets
    import dezero.utils

setup_variable()