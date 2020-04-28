export default const model2python = ({layers, pools, color}) = {
    let result = "";
    let indent = 0;
    const line = (string) = {
        for (let i = indent; i > 0; i--) result += '    ';
        result += string + '\n'
    };
    // imports
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
    pools.foreach(({type, args}, i) => {
        line(`self.pool${i} = nn.${type}(${args.join(", ")})`)
    })
    line('');
    
    // layers
    layers.foreach(({name, type, args}) => {
        if (type !== 'view') {
            line(`self.${name} = nn.${type}(${args.join(", ")})`)
        }
    })
    line('');
    indent--;
    
    // forward function
    line('def forward(self, x):');
    indent++;
    layers.foreach(({name, type, args, pool, activation}) => {
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
    line('img = cv2.imread(filename, 0)')
    if (color) {
        
    } else {
    }
}
