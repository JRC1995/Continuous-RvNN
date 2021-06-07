import torch as T
import torch.nn as nn
import torch.nn.functional as F
from models.layers import Linear
from models.utils import gelu
import math


class CYK_CELL(nn.Module):
    def __init__(self, config):
        super(CYK_CELL, self).__init__()

        self.hidden_size = config["hidden_size"]
        self.in_dropout = config["in_dropout"]
        self.hidden_dropout = config["hidden_dropout"]

        self.scorer = Linear(self.hidden_size, 1)

        self.initial_transform_layer = Linear(self.hidden_size, self.hidden_size)
        self.wcell1 = Linear(2 * self.hidden_size, self.cell_hidden_size)
        self.wcell2 = Linear(self.cell_hidden_size, 4 * self.hidden_size)
        self.LN = nn.LayerNorm(self.hidden_size)

        self.eps = 1e-8

    # %%
    def sum_normalize(self, logits, dim=-1):
        return logits / T.sum(logits + self.eps, keepdim=True, dim=dim)

    # %%
    def initial_transform(self, sequence):
        sequence = self.LN(self.initial_transform_layer(sequence))
        return sequence

    # %%
    def composer(self, child1, child2):
        N, S, D = child1.size()

        concated = T.cat([child1, child2], dim=-1)
        assert concated.size() == (N, S, 2 * D)

        # concated = F.dropout(concated, p=self.hidden_dropout, training=self.training)
        if self.config["hidden_activation"].lower() == "relu":
            intermediate = F.relu(self.wcell1(concated))
        else:
            intermediate = F.gelu(self.wcell1(concated))
        intermediate = F.dropout(intermediate, p=self.hidden_dropout, training=self.training)
        contents = self.wcell2(intermediate)

        contents = contents.view(N, S, 4, D)
        gates = T.sigmoid(contents[:, :, 0:3, :])
        parent = contents[:, :, 3, :]
        f1 = gates[..., 0, :]
        f2 = gates[..., 1, :]
        i = gates[..., 2, :]

        transition = self.LN(f1 * child1 + f2 * child2 + i * parent)

        return transition

    # %%
    def compute_entropy_penalty(self, active_probs, last_token_mask):
        N, S = active_probs.size()
        active_prob_dist = self.sum_normalize(active_probs, dim=-1)
        nll_loss = - T.log(T.sum(last_token_mask * active_prob_dist, dim=1) + self.eps)
        nll_loss = nll_loss.view(N)
        return nll_loss

    # %%
    def compute_speed_penalty(self, steps, input_mask):
        steps = T.max(steps, dim=1)[0]
        speed_penalty = steps.squeeze(-1) / (T.sum(input_mask.squeeze(-1), dim=1) - 2.0)
        return speed_penalty

    # %%
    def encoder_block(self, sequence, input_mask):
        N, S, D = sequence.size()
        """
        Initial Transform
        """
        sequence = self.initial_transform(sequence)
        sequence = sequence * input_mask

        chart = [sequence]

        for row in range(1, S):
            left_stack = []
            right_stack = []
            for j in range(row):
                left = chart[j][:, 0:-row, :]
                right = chart[row - j - 1][:, j + 1:, :]

                assert left.size() == (N, row - S, self.hidden_size)
                assert right.size() == (N, row - S, self.hidden_size)

                left_stack.append(left)
                right_stack.append(right)

            left_stack = T.stack(left_stack, dim=1)
            right_stack = T.stack(right_stack, dim=1)
            assert left_stack.size() == (N, row, S - row, self.hidden_size)
            assert right_stack.size() == (N, row, S - row, self.hidden_size)

            left_stack = left_stack.view(N * row, S - row, self.hidden_size)
            right_stack = right_stack.view(N * row, S - row, self.hidden_size)

            combined_stack = self.composer(left_stack, right_stack)

            combined_stack = combined_stack.view(N, row, S - row, self.hidden_size)
            combined_stack = combined_stack / (T.norm(combined_stack, keepdim=True, dim=-1) + self.eps)
            combined_scores = F.softmax(self.scorer(combined_stack), dim=1)

            new_row = T.sum(combined_scores * combined_stack, dim=1)
            assert new_row.size() == (N, S-row, self.hidden_size)

            chart.append(new_row)

        global_state = chart[-1]
        assert global_state.size() == (N, 1, self.hidden_size)
        global_state = global_state.squeeze(1)

        penalty = None

        return sequence, global_state, penalty

    # %%
    def forward(self, sequence, input_mask):
        input_mask = input_mask.unsqueeze(-1)
        sequence = sequence * input_mask

        sequence, global_state, penalty = self.encoder_block(sequence, input_mask)
        sequence = sequence * input_mask
        return {"sequence": sequence, "penalty": penalty, "global_state": global_state}
