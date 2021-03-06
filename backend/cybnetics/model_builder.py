import torch
import torch.nn as nn
import torch.nn.functional as F

class ViewAdapter:
    """ hack to allow use of view in forward function """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    def __call__(self, x):
        return x.view(*self.args, **self.kwargs)

class FlattenAdapter:
    """ hack to allow use of flatten in forward function """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    def __call__(self, x):
        return torch.flatten(x, *self.args, **self.kwargs)


class BadModelSpec(ValueError):
    pass

def identity(x):
    return x

# layers is an array of dicts [{'type': 'Conv2d', 'args': [3, 32, 3], 'kwargs': {}, 'pooling': 0, 'activation': 'relu', 'name': 'conv1'}]
# pools is also an array of dicts [{'type': 'MaxPool2D', 'args': [2, 2], 'kwargs': {}}]
# kwargs may not be needed in the end result
# pooling is not set if the layer doesn't use pooling
def make_model_class(layers, pools, dropouts):

    for pool in pools:
        try:
            class_name = pool['type']
            args = pool['args']
            kwargs = pool.get('kwargs', {})
        except:
            raise BadModelSpec('pool should specify type and args')
        try:
            pool_class = getattr(nn, class_name)
        except:
            raise BadModelSpec(f'pool type {class_name} was not found')
        pool['instance'] = pool_class(*args, **kwargs)

    for dropout in dropouts:
        try:
            class_name = dropout['type']
            args = dropout['args']
            kwargs = pool.get('kwargs', {})
        except:
            raise BadModelSpec('droupout should specify type and args')
        try:
            dropout_class = getattr(nn, class_name)
        except:
            raise BadModelSpec(f'pool type {class_name} was not found')
        dropout['instance'] = dropout_class(*args, **kwargs)

    for layer in layers:
        # get params from layer dict
        try:
            class_name = layer['type']
            args = layer['args']

            if (class_name != 'view') and (class_name != 'flatten'):
                name = layer['name']

            activation = layer.get('activation')
            kwargs = layer.get('kwargs', {})
        except:
            raise BadModelSpec('layer should specify type args, and activation')

        # view special case/hack
        if class_name == 'view':
            layer['instance'] = ViewAdapter(*args, **kwargs)
            layer['pool'] = identity
            layer['dropout'] = identity
            layer['activation'] = identity
            continue

        if class_name == 'flatten':
            layer['instance'] = FlattenAdapter(*args, **kwargs)
            layer['pool'] = identity
            layer['dropout'] = identity
            layer['activation'] = identity
            continue

        # initialize instance, pool and activation
        try:
            layer_class = getattr(nn, class_name)
        except:
            raise BadModelSpec(f'layer type {class_name} was not found')
        try:
            if not activation:
                # another hack for the last layer :/
                layer['activation'] = identity
            else:
                layer['activation'] = getattr(F, activation)
        except:
            raise BadModelSpec(f'couldn\'t find activation function {activation}')
        pool = layer.get('pool')
        if pool is not None:
            if pool >= len(pools) or pool < 0:
                raise BadModelSpec(f'invalid pool number {pool}')
            layer['pool'] = pools[int(pool)]['instance']
        else:
            # if no pool use identity for simplicity
            layer['pool'] = identity
        layer['instance'] = layer_class(*args, **kwargs)

        dropout = layer.get('dropout')
        if dropout is not None:
            if dropout >= len(dropouts) or dropout < 0:
                raise BadModelSpec(f'invalid dropout number {dropout}')
            layer['dropout'] = dropouts[int(dropout)]['instance']
        else:
            # if no pool use identity for simplicity
            layer['dropout'] = identity
        layer['instance'] = layer_class(*args, **kwargs)

    class ModelClass(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.layers = layers
            self.pools = pools
            self.droupouts = dropouts
            # set layers as attrs so loading the tate dict works
            for layer in self.layers:
                if layer['type'] == 'view':
                    continue
                if layer['type'] == 'flatten':
                    continue
                if getattr(self, layer['name'], None):
                    raise BadModelSpec(f'invalid layer name {layer["name"]}')
                setattr(self, layer['name'], layer['instance'])

        def forward(self, x):
            for layer in self.layers:
                x = layer['pool'](layer['activation'](layer['instance'](x)))
                x = layer['dropout'](x)
            return x
    return ModelClass


##### Examples #####
# applies to: strong_cifar.pt
def make_cfair10_example():
    return make_model_class([
        {'type': 'Conv2d', 'args': [3, 32, 3], 'activation': 'relu', 'name': 'conv1'},
        {'type': 'Conv2d', 'args': [32, 32, 3], 'activation': 'relu', 'pool': 0, 'name': 'conv2'},
        {'type': 'Conv2d', 'args': [32, 128, 3], 'activation': 'relu', 'name': 'conv3'},
        {'type': 'Conv2d', 'args': [128, 128, 3], 'activation': 'relu', 'pool': 0, 'name': 'conv4'},
        {'type': 'view', 'args': [-1, 128 * 5 * 5]},
        {'type': 'Linear', 'args': [128 * 5 * 5, 256], 'activation': 'relu', 'name': 'fc1'},
        {'type': 'Linear', 'args': [256, 256], 'activation': 'relu', 'name': 'fc2'},
        {'type': 'Linear', 'args': [256, 10], 'name': 'fc3'}
    ],[
        {'type': 'MaxPool2d', 'args': [2, 2]}
    ], [

    ])

# applies to: mnist.pt, mnist_tvm_no_good.pt , mnist_tvm_better.pt, mnist_tvm_good.pt
def make_mnist_example():
    return make_model_class([
        {'type': 'Conv2d', 'args': [1, 32, 3], 'activation': 'relu', 'name': 'conv1'},
        {'type': 'Conv2d', 'args': [32, 64, 3], 'activation': 'relu', 'pool': 0, 'dropout': 0, 'name': 'conv2'},
        {'type': 'flatten', 'args': [1]},
        {'type': 'Linear', 'args': [9216, 128], 'activation': 'relu', 'dropout': 1, 'name': 'fc1'},
        {'type': 'Linear', 'args': [128, 10], 'activation': 'log_softmax', 'name': 'fc2'},
    ],[
        {'type': 'MaxPool2d', 'args': [2, 2]}
    ], [
        {'type': 'Dropout2d', 'args': [0.25]},
        {'type': 'Dropout2d', 'args': [0.5]}
    ])

# applies to: drop_cifar.pt
def make_cfair10_dropout_example():
    return make_model_class([
        {'type': 'Conv2d', 'args': [3, 32, 3], 'activation': 'relu', 'name': 'conv1'},
        {'type': 'Conv2d', 'args': [32, 32, 3], 'activation': 'relu', 'pool': 0, 'name': 'conv2'},
        {'type': 'Conv2d', 'args': [32, 128, 3], 'activation': 'relu', 'name': 'conv3'},
        {'type': 'Conv2d', 'args': [128, 128, 3], 'activation': 'relu', 'pool': 0, 'name': 'conv4'},
        {'type': 'view', 'args': [-1, 128 * 5 * 5]},
        {'type': 'Linear', 'args': [128 * 5 * 5, 256], 'activation': 'relu', 'dropout': 0, 'name': 'fc1'},
        {'type': 'Linear', 'args': [256, 256], 'activation': 'relu', 'name': 'fc2'},
        {'type': 'Linear', 'args': [256, 10], 'name': 'fc3'}
    ],[
        {'type': 'MaxPool2d', 'args': [2, 2]}
    ], [
        {'type': 'Dropout2d', 'args': [0.3]}
    ])
