import torch.nn as nn
import numpy as np
import torch

def initWeights(m, init_bias=0.0):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.normal_(init_bias, 0.01)

# class MLP(nn.Module):
#     def __init__(self, input_size, output_size, shape, activation, normalization=None):
#         super(MLP, self).__init__()
#         self.activation_fn = activation
#         assert normalization in [None, 'batch_norm', 'layer_norm']
#         if normalization == 'batch_norm':
#             modules = [nn.Linear(input_size, shape[0]), nn.BatchNorm1d(shape[0]), self.activation_fn()]
#             for idx in range(len(shape)-1):
#                 modules.append(nn.Linear(shape[idx], shape[idx+1]))
#                 modules.append(nn.BatchNorm1d(shape[idx+1]))
#                 modules.append(self.activation_fn())
#         elif normalization == 'layer_norm':
#             modules = [nn.Linear(input_size, shape[0]), nn.LayerNorm(shape[0]), self.activation_fn()]
#             for idx in range(len(shape)-1):
#                 modules.append(nn.Linear(shape[idx], shape[idx+1]))
#                 modules.append(nn.LayerNorm(shape[idx+1]))
#                 modules.append(self.activation_fn())
#         else:
#             modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
#             for idx in range(len(shape)-1):
#                 modules.append(nn.Linear(shape[idx], shape[idx+1]))
#                 modules.append(self.activation_fn())
#         modules.append(nn.Linear(shape[-1], output_size))
#         self.architecture = nn.Sequential(*modules)
#         self.input_shape = [input_size]
#         self.output_shape = [output_size]

#     def forward(self, input):
#         return self.architecture(input)

class MLP(nn.Module):
    def __init__(self, input_size, output_size, shape, activation, layer_norm=False, crelu=False):
        super(MLP, self).__init__()
        self.shape = shape
        self.activation_fn = activation
        self.layer_norm = layer_norm
        self.crelu = crelu
        self.modules = []
        if self.layer_norm:
            self.modules += [nn.Linear(input_size, shape[0]), nn.LayerNorm(shape[0]), self.activation_fn()]
        else:
            self.modules += [nn.Linear(input_size, shape[0]), self.activation_fn()]
        for idx in range(len(shape)-1):
            if self.crelu:
                self.modules.append(nn.Linear(shape[idx]*2, shape[idx+1]))
            else:
                self.modules.append(nn.Linear(shape[idx], shape[idx+1]))
            if self.layer_norm:
                self.modules.append(nn.LayerNorm(shape[idx+1]))
            self.modules.append(self.activation_fn())
        if self.crelu:
            self.modules.append(nn.Linear(shape[-1]*2, output_size))
        else:
            self.modules.append(nn.Linear(shape[-1], output_size))
        self.module_list = nn.ModuleList(self.modules)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    def forward(self, inputs):
        x = inputs
        if self.layer_norm:
            for idx in range(len(self.shape)):
                x = self.modules[3*idx](x) # linear
                x = self.modules[3*idx + 1](x) # layer norm
                if self.crelu: # activation
                    x1 = self.modules[3*idx + 2](x)
                    x2 = self.modules[3*idx + 2](-x)
                    x = torch.cat((x1, x2), dim=-1)
                else:
                    x = self.modules[3*idx + 2](x)
            x = self.modules[-1](x)
        else:
            for idx in range(len(self.shape)):
                x = self.modules[2*idx](x) # linear
                if self.crelu: # activation
                    x1 = self.modules[2*idx + 1](x)
                    x2 = self.modules[2*idx + 1](-x)
                    x = torch.cat((x1, x2), dim=-1)
                else:
                    x = self.modules[2*idx + 1](x)
            x = self.modules[-1](x)
        return x
