import torch
import torch.nn as nn

class MultiOmicsMLP(nn.Module):
    def __init__(self, input_dims_dict, output_dim, hidden_dims=[256, 128, 64]):
        super(MultiOmicsMLP, self).__init__()

        self.input_keys = sorted(input_dims_dict.keys())
        total_input_dim = sum(input_dims_dict[k] for k in self.input_keys)

        layers = [nn.Linear(total_input_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims) - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x_dict):
        x = torch.cat([x_dict[k] for k in self.input_keys], dim=-1)
        return self.model(x)