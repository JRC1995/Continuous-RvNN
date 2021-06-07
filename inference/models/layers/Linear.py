import torch.nn as nn

from models.utils import glorot_uniform_init


class Linear(nn.Module):
    def __init__(self, fan_in, fan_out,
                 true_fan_in=None, true_fan_out=None,
                 bias=True):
        super(Linear, self).__init__()

        if true_fan_in is None:
            true_fan_in = fan_in

        if true_fan_out is None:
            true_fan_out = fan_out

        self.linear = nn.Linear(fan_in, fan_out, bias=bias)

        glorot_uniform_init(self.linear.weight,
                            fan_in=true_fan_in, fan_out=true_fan_out)

        if bias is True:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)
