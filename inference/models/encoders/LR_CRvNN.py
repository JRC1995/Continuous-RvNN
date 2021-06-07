import torch as T
import torch.nn as nn
import torch.nn.functional as F
from models.layers import Linear
from models.utils import gelu
import math


class LR_CRvNN(nn.Module):
    def __init__(self, config):
        super(LR_CRvNN, self).__init__()

        self.config = config
        self.hidden_size = config["hidden_size"]
        self.cell_hidden_size = config["cell_hidden_size"]
        self.in_dropout = config["in_dropout"]
        self.hidden_dropout = config["hidden_dropout"]

        self.START = nn.Parameter(T.zeros(self.hidden_size))

        self.initial_transform_layer = Linear(self.hidden_size, self.hidden_size)
        self.wcell1 = Linear(2 * self.hidden_size, self.cell_hidden_size)
        self.wcell2 = Linear(self.cell_hidden_size, 4 * self.hidden_size)
        self.LN = nn.LayerNorm(self.hidden_size)

        self.eps = 1e-8

    # %%
    def initial_transform(self, sequence):
        sequence = self.LN(self.initial_transform_layer(sequence))
        return sequence


    # %%
    def composer(self, child1, child2):
        N, D = child1.size()

        concated = T.cat([child1, child2], dim=-1)
        assert concated.size() == (N, 2 * D)

        # concated = F.dropout(concated, p=self.hidden_dropout, training=self.training)
        if self.config["hidden_activation"].lower() == "relu":
            intermediate = F.relu(self.wcell1(concated))
        else:
            intermediate = gelu(self.wcell1(concated))
        intermediate = F.dropout(intermediate, p=self.hidden_dropout, training=self.training)
        contents = self.wcell2(intermediate)

        contents = contents.view(N, 4, D)
        gates = T.sigmoid(contents[:, 0:3, :])
        parent = contents[:, 3, :]
        f1 = gates[..., 0, :]
        f2 = gates[..., 1, :]
        i = gates[..., 2, :]

        transition = self.LN(f1 * child1 + f2 * child2 + i * parent)

        return transition


    # %%
    def encoder_block(self, sequence, input_mask):

        N, S, D = sequence.size()
        """
        Initial Transform
        """
        sequence = self.initial_transform(sequence)
        sequence = sequence * input_mask
        """
        Start Recursion
        """
        sequence_list = []
        ht = self.START.view(1, self.hidden_size).repeat(N, 1)
        global_state = ht.clone()
        for t in range(S):
            ht = self.composer(child1=ht, child2=sequence[:, t, :])
            global_state = input_mask[:, t, :] * ht + (1 - input_mask[:, t, :]) * global_state
            sequence_list.append(ht.clone())

        sequence = T.stack(sequence_list, dim=1)
        assert sequence.size() == (N, S, self.hidden_size)

        penalty = None

        return sequence, global_state, penalty

    # %%
    def forward(self, sequence, input_mask, **kwargs):

        if "temperature" in kwargs:
            self.temperature = kwargs["temperature"]
        else:
            self.temperature = 1.0

        self.temperature = 1.0 if self.temperature is None else self.temperature

        input_mask = input_mask.unsqueeze(-1)
        sequence = sequence * input_mask

        sequence, global_state, penalty = self.encoder_block(sequence, input_mask)
        sequence = sequence * input_mask
        return {"sequence": sequence, "penalty": penalty, "global_state": global_state}
