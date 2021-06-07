import torch as T
import torch.nn as nn
import torch.nn.functional as F

from controllers.encoder_controller import encoder
from models.layers import Linear
from models.utils import gelu
from models.utils import glorot_uniform_init

class NLI_model(nn.Module):
    def __init__(self, attributes, config):

        super(NLI_model, self).__init__()

        self.config = config
        self.classes_num = attributes["classes_num"]
        embedding_data = attributes["embedding_data"]
        pad_id = attributes["PAD_id"]
        self.pad_id = pad_id
        ATT_PAD = -999999
        self.ATT_PAD = T.tensor(ATT_PAD).float()
        self.zeros = T.tensor(0.0)
        self.out_dropout = config["out_dropout"]
        self.in_dropout = config["in_dropout"]
        self.hidden_size = config["hidden_size"]
        self.UNK_id = attributes["UNK_id"]
        self.unk_embed = None

        if embedding_data is not None:
            embedding_data = T.tensor(embedding_data)
            self.unk_embed = nn.Parameter(T.randn(embedding_data.size(-1)))
            self.word_embedding = nn.Embedding.from_pretrained(embedding_data,
                                                               freeze=config["word_embd_freeze"],
                                                               padding_idx=pad_id)
        else:
            vocab_len = attributes["vocab_len"]
            self.word_embedding = nn.Embedding(vocab_len, config["embd_dim"],
                                               padding_idx=pad_id)

        self.embd_dim = self.word_embedding.weight.size(-1)

        assert self.embd_dim == config["hidden_size"]

        if embedding_data is None:
            initrange = 0.1
            self.word_embedding.weight.data.uniform_(-initrange, initrange)

        if not config["global_state_return"]:
            self.attn_linear1 = nn.Linear(config["hidden_size"], config["hidden_size"])
            self.attn_linear2 = nn.Linear(config["hidden_size"], 1)

        self.encoder = encoder(config)

        self.prediction1 = nn.Linear(4 * config["hidden_size"], config["hidden_size"])
        self.prediction2 = nn.Linear(config["hidden_size"], self.classes_num)


    # %%

    def embed(self, sequence_idx, input_mask):

        N, S = sequence_idx.size()

        sequence = self.word_embedding(sequence_idx)

        if self.UNK_id is not None and self.unk_embed is not None:
            sequence = T.where(sequence_idx.unsqueeze(-1) == self.UNK_id,
                               self.unk_embed.view(1, 1, -1).repeat(N, S, 1),
                               sequence)


        sequence = sequence * input_mask.view(N, S, 1)

        return sequence, input_mask


    def extract_features(self, sequence, mask):
        N, S, D = sequence.size()

        mask = mask.view(N, S, 1)

        attention_mask = T.where(mask == 0,
                                 self.ATT_PAD.to(mask.device),
                                 self.zeros.to(mask.device))

        assert attention_mask.size() == (N, S, 1)

        energy = self.attn_linear2(gelu(self.attn_linear1(sequence)))

        assert energy.size() == (N, S, 1)

        attention = F.softmax(energy + attention_mask, dim=1)

        assert attention.size() == (N, S, 1)

        z = T.sum(attention * sequence, dim=1)

        assert z.size() == (N, D)

        return z

    # %%
    def forward(self, batch):

        sequence1 = batch["sequences1_vec"]
        sequence2 = batch["sequences2_vec"]
        input_mask1 = batch["input_masks1"]
        input_mask2 = batch["input_masks2"]
        temperature = batch["temperature"]

        N = sequence1.size(0)

        # EMBEDDING BLOCK
        sequence1, input_mask1 = self.embed(sequence1, input_mask1)
        sequence2, input_mask2 = self.embed(sequence2, input_mask2)

        sequence1 = F.dropout(sequence1, p=self.in_dropout, training=self.training)
        sequence2 = F.dropout(sequence2, p=self.in_dropout, training=self.training)

        if "batch_pair" in self.config and self.config["batch_pair"]:
            pad = T.zeros(N, 1, self.hidden_size).float().to(sequence1.device)
            zero = T.zeros(N, 1).float().to(sequence1.device)

            max_s = max(sequence1.size(1), sequence2.size(1))

            if self.config["left_padded"]:
                while sequence1.size(1) < max_s:
                    sequence1 = T.cat([pad.clone(), sequence1], dim=1)
                    input_mask1 = T.cat([zero.clone(), input_mask1], dim=1)
                while sequence2.size(1) < max_s:
                    sequence2 = T.cat([pad.clone(), sequence2], dim=1)
                    input_mask2 = T.cat([zero.clone(), input_mask2], dim=1)
            else:
                while sequence1.size(1) < max_s:
                    sequence1 = T.cat([sequence1, pad.clone()], dim=1)
                    input_mask1 = T.cat([input_mask1, zero.clone()], dim=1)
                while sequence2.size(1) < max_s:
                    sequence2 = T.cat([sequence2, pad.clone()], dim=1)
                    input_mask2 = T.cat([input_mask2, zero.clone()], dim=1)

            sequence = T.cat([sequence1, sequence2], dim=0)
            input_mask = T.cat([input_mask1, input_mask2], dim=0)
            sequence_dict = self.encoder(sequence, input_mask, temperature=temperature)

            sequence1_dict = {}
            sequence2_dict = {}
            for key in sequence_dict:
                sequence1_dict[key] = None if sequence_dict[key] is None else sequence_dict[key][0:N]
                sequence2_dict[key] = None if sequence_dict[key] is None else sequence_dict[key][N:]

        else:
            # ENCODER BLOCK
            sequence1_dict = self.encoder(sequence1, input_mask1, temperature=temperature)
            sequence2_dict = self.encoder(sequence2, input_mask2, temperature=temperature)

        sequence1 = sequence1_dict["sequence"]
        sequence2 = sequence2_dict["sequence"]

        penalty = None
        if "penalty" in sequence1_dict:
            penalty1 = sequence1_dict["penalty"]
            penalty2 = sequence2_dict["penalty"]
            if penalty1 is not None and penalty2 is not None:
                penalty = (penalty1 + penalty2) / 2

        if self.config["global_state_return"]:
            feats1 = sequence1_dict["global_state"]
            feats2 = sequence2_dict["global_state"]
        else:
            feats1 = self.extract_features(sequence1, input_mask1)
            feats2 = self.extract_features(sequence2, input_mask2)

        feats = T.cat([feats1, feats2,
                       feats1 * feats2,
                       T.abs(feats1 - feats2)], dim=-1)

        feats = F.dropout(feats, p=self.out_dropout, training=self.training)
        #intermediate = gelu(self.prediction1(feats))
        intermediate = F.elu(self.prediction1(feats))
        intermediate = F.dropout(intermediate, p=self.out_dropout, training=self.training)
        logits = self.prediction2(intermediate)

        assert logits.size() == (N, self.classes_num)

        return {"logits": logits, "penalty": penalty}
