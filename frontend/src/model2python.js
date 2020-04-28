export const model2python = ({layers, pools, color}) => {
    let result = "";
    let indent = 0;
    const line = (string) => {
        for(let i = indent; i > 0; i--) result += '    ';
        result += string + '\n'
    };
    // imports
    line('import cv2')
    line('import torch');
    line('import torch.nn as n');
    line('import torch.nn.functional as F');

    line('');
    
    // class definition
    line('class CybneticsModel(nn.Module):');
    indent++;
    
    // init function
    line('def __init__(self):')
    line('super().__init__()')
    indent++;
    
    // pools
    pools.forEach(({type, args}, i) => {
        line(`self.pool${i} = nn.${type}(${args.join(", ")})`)
    })
    line('');
    
    // layers
    layers.forEach(({name, type, args}) => {
        if (type !== 'view') {
            line(`self.${name} = nn.${type}(${args.join(", ")})`)
        }
    })
    line('');
    indent--;
    
    // forward function
    line('def forward(self, x):');
    indent++;
    layers.forEach(({name, type, args, pool, activation}) => {
        if (type === 'view') {
            line(`x = x.view(x, ${args.join(", ")})`)
        } else {
            let temp = `self.${name}(x)`;
            if (activation !== undefined) {
                temp = `F.${activation}(${temp})`
            }
            if (pool !== undefined) {
                temp = `self.pool${pool}(${temp})`
            }
            line(`x = ${temp}`);
        }
    })
    line('return x');
    line('');
    indent--;
    indent--;
    
    // convert_tensor
    line('def convert_tensor(filename):')
    indent++;
    if (color) {
        line('img = cv2.imread(filename, cv2.IMREAD_COLOR)');
        line('init_reshape = img.transpose((2, 0, 1))');
        line('img_tensor = torch.from_numpy(init_reshape)');
        line('return img_tensor.float()');
        
    } else {
        line('img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)');
        line('init_reshape = img.transpose((2, 0, 1))');
        line('img_tensor = torch.from_numpy(init_reshape).reshape([1,3, img.shape[0], img.shape[1]])');
        line('return img_tensor.float()');
    }
    return result;
}

export const example_call = () => {
    return model2python({
        layers: [
            {'type': 'Conv2d', 'args': [3, 32, 3], 'activation': 'relu', 'name': 'conv1'},
            {'type': 'Conv2d', 'args': [32, 32, 3], 'activation': 'relu', 'pool': 0, 'name': 'conv2'},
            {'type': 'Conv2d', 'args': [32, 128, 3], 'activation': 'relu', 'name': 'conv3'},
            {'type': 'Conv2d', 'args': [128, 128, 3], 'activation': 'relu', 'pool': 0, 'name': 'conv4'},
            {'type': 'view', 'args': [-1, 128 * 5 * 5]},
            {'type': 'Linear', 'args': [128 * 5 * 5, 256], 'activation': 'relu', 'name': 'fc1'},
            {'type': 'Linear', 'args': [256, 256], 'activation': 'relu', 'name': 'fc2'},
            {'type': 'Linear', 'args': [256, 10], 'name': 'fc3'}
        ],
        pools: [
            {'type': 'MaxPool2d', 'args': [2, 2]}
        ],
        color: true
    })
}
