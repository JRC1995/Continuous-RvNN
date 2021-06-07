import torch as T
import torch.nn as nn
import torch.nn.functional as F
from models.layers import Linear
from models.utils import gelu
import math


class CRvNN_balanced(nn.Module):
    def __init__(self, config):
        super(CRvNN_balanced, self).__init__()

        self.config = config
        self.hidden_size = config["hidden_size"]
        self.cell_hidden_size = config["cell_hidden_size"]
        self.window_size = config["window_size"]
        self.stop_threshold = config["stop_threshold"]
        # self.switch_threshold = config["switch_threshold"]
        self.entropy_gamma = config["halt_gamma"]
        self.structure_gamma = 0.01  # config["structure_gamma"]
        self.speed_gamma = 0.0  # config["speed_gamma"]
        self.in_dropout = config["in_dropout"]
        self.hidden_dropout = config["hidden_dropout"]
        self.recurrent_momentum = config["transition_features"]
        self.small_d = config["small_d"]

        self.START = nn.Parameter(T.randn(self.hidden_size))
        self.END = nn.Parameter(T.randn(self.hidden_size))

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

    def augment_sequence(self, sequence, input_mask):
        N, S, D = sequence.size()
        assert input_mask.size() == (N, S, 1)

        """
        AUGMENT SEQUENCE WITH START AND END TOKENS
        """
        # ADD START TOKEN
        START = self.START.view(1, 1, D).repeat(N, 1, 1)
        sequence = T.cat([START, sequence], dim=1)
        assert sequence.size() == (N, S + 1, D)
        input_mask = T.cat([T.ones(N, 1, 1).float().to(input_mask.device), input_mask], dim=1)
        assert input_mask.size() == (N, S + 1, 1)

        # ADD END TOKEN
        input_mask_no_end = T.cat([input_mask.clone(), T.zeros(N, 1, 1).float().to(input_mask.device)], dim=1)
        input_mask_yes_end = T.cat([T.ones(N, 1, 1).float().to(input_mask.device), input_mask.clone()], dim=1)
        END_mask = input_mask_yes_end - input_mask_no_end
        assert END_mask.size() == (N, S + 2, 1)

        END = self.END.view(1, 1, D).repeat(N, S + 2, 1)
        sequence = T.cat([sequence, T.zeros(N, 1, D).float().to(sequence.device)], dim=1)
        sequence = END_mask * END + (1 - END_mask) * sequence

        input_mask = input_mask_yes_end
        input_mask_no_start = T.cat([T.zeros(N, 1, 1).float().to(input_mask.device),
                                     input_mask[:, 1:, :]], dim=1)

        return sequence, input_mask, END_mask, input_mask_no_start, input_mask_no_end

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

    def recursive_f(self, sequence, input_mask):
        if sequence.size(1) == 1:
            return sequence, input_mask
        elif sequence.size(1) == 2:
            left_sequence = sequence[:, 0, :].unsqueeze(1)
            right_sequence = sequence[:, 1, :].unsqueeze(1)
            alpha = input_mask[:, 1, :].unsqueeze(1)
            sequence_ = self.composer(child1=left_sequence, child2=right_sequence)
            sequence = alpha * sequence_ + (1 - alpha) * left_sequence
            return sequence, input_mask[:, 0, :].unsqueeze(1)
        else:
            S = sequence.size(1)
            S1 = math.ceil(S / 2)
            left_child, left_mask = self.recursive_f(sequence[:, 0:S1, :], input_mask[:, 0:S1, :])
            right_child, right_mask = self.recursive_f(sequence[:, S1:, :], input_mask[:, S1:, :])
            alpha = right_mask
            sequence_ = self.composer(child1=left_child, child2=right_child)
            sequence = alpha * sequence_ + (1 - alpha) * left_child
            return sequence, left_mask

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
        sequence, input_mask = self.recursive_f(sequence, input_mask)
        global_state = sequence[:, 0, :]
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
        # sequence = sequence * input_mask
        return {"sequence": sequence, "penalty": penalty, "global_state": global_state}
