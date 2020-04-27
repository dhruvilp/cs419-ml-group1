import torch.nn as nn
import torch.nn.functional as F

class ViewAdapter:
    """hack to allow use of view in forward function"""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    def __call__(self, x):
        return x.view(*self.args, **self.kwargs)

def identity(x):
    return x
    
# layers is an array of dicts [{'type': 'Conv2d', 'args': [3, 32, 3], 'kwargs': {}, 'pooling': 0, 'activation': 'relu'}]
# pools is also an array of dicts [{'type': 'MaxPool2D', 'args': [2, 2], 'kwargs': {}}]
# kwargs may not be needed in the end result
# pooling is not set if the layer doesn't use pooling
def make_model_class(layers, pools):
    for pool in pools:
        try:
            class_name = pool['type']
            args = pool['args']
            kwargs = pool.get('kwargs', {})
        except:
            raise ValueError('pool should specify type and args')
        try:
            pool_class = getattr(nn, class_name)
        except:
            raise ValueError(f'pool type {class_name} was not found')
        pool['instance'] = pool_class(*args, **kwargs)

    for layer in layers:
        # get params from layer dict
        try:
            class_name = layer['type']
            args = layer['args']
            activation = layer.get('activation')
            kwargs = layer.get('kwargs', {})
        except:
            raise ValueError('layer should specify type args, and activation')

        # view special case/hack
        if class_name == 'view':
            layer['instance'] = ViewAdapter('')
            layer['pool'] = identity
            layer['activation'] = identity
            continue

        # initialize instance, pool and activation
        try:
            layer_class = getattr(nn, class_name)
        except:
            raise ValueError(f'layer type {class_name} was not found')
        try:
            if not activation:
                # another hack for the last layer :/
                layer['activation'] = identity
            else:
                layer['activation'] = getattr(F, activation)
        except:
            raise ValueError(f'couldn\'t find activation function {activation}')
        pool = layer.get('pool')
        if pool:
            if pool > len(pools) or pool < 0:
                raise ValueError(f'invalid pool number {pool}')
            layer['pool'] = pools[int(pool)]['instance']
        else:
            # if no pool use identity for simplicity
            layer['pool'] = identity
        layer['instance'] = layer_class(*args, **kwargs)
        
    class ModelClass(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.layers = layers
            self.pools = pools
            
        def forward(self, x):
            for layer in self.layers:
                x = layer['pool'](layer['activation'](layer['instance'](x)))
            return x
    return ModelClass

def make_cfair10_example():
    return make_model_class([
        {'type': 'Conv2d', 'args': [3, 32, 3], 'activation': 'relu'},
        {'type': 'Conv2d', 'args': [32, 32, 3], 'activation': 'relu', 'pool': 0},
        {'type': 'Conv2d', 'args': [32, 128, 3], 'activation': 'relu'},
        {'type': 'Conv2d', 'args': [128, 128, 3], 'activation': 'relu', 'pool': 0},
        {'type': 'view', 'args': [-1, 128 * 5 * 5]},
        {'type': 'Linear', 'args': [128 * 5 * 5, 256], 'activation': 'relu'},
        {'type': 'Linear', 'args': [256, 256], 'activation': 'relu'},
        {'type': 'Linear', 'args': [256, 10]}
    ],[
        {'type': 'MaxPool2d', 'args': [2, 2]}
    ])
