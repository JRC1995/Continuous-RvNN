import torch as T
import torch.nn as nn
import torch.nn.functional as F

from models.layers import Linear


class LSTM(nn.Module):
    def __init__(self, config):

        super(LSTM, self).__init__()

        self.hidden_size = config["hidden_size"]
        self.in_dropout = config["in_dropout"]
        self.encode_layers = 1
        self.config = config

        self.h0 = nn.Parameter(T.zeros(self.hidden_size).float())
        self.c0 = nn.Parameter(T.zeros(self.hidden_size).float())
        self.compress = Linear(2*self.hidden_size, self.hidden_size)

        self.rnn = nn.LSTM(input_size=self.hidden_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.encode_layers,
                           batch_first=True,
                           bidirectional=True)

    # %%
    def forward(self, sequence, input_mask, **kwargs):
        """
        N = Batch Size
        S = Sequence Size
        """

        N, S, _ = sequence.size()
        input_mask = input_mask.view(N, S, 1)
        lengths = T.sum(input_mask, dim=1).long().view(N).cpu()

        #sequence = F.dropout(sequence, p=self.in_dropout, training=self.training)

        packed_sequence = nn.utils.rnn.pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.rnn(packed_sequence)

        last_token_mask = input_mask - T.cat([input_mask[:, 1:, :],
                                              T.zeros(N,1,1).float().to(input_mask.device)],dim=1)

        sequence, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        assert sequence.size() == (N, S, 2*self.hidden_size)

        forward = T.sum(last_token_mask * sequence[:, :, 0:self.hidden_size], dim=1)
        backward = sequence[:, 0, self.hidden_size:]
        sequence = self.compress(sequence)
        sequence = sequence * input_mask

        global_state = self.compress(T.cat([forward, backward], dim=-1))

        assert global_state.size() == (N, self.hidden_size)

        return {"sequence": sequence, "global_state": global_state}