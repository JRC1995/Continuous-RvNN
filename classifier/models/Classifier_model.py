import torch as T
import torch.nn as nn
import torch.nn.functional as F

from controllers.encoder_controller import encoder
from models.layers import Linear
from models.utils import gelu
from models.utils import glorot_uniform_init

class Classifier_model(nn.Module):
    def __init__(self, attributes, config):

        super(Classifier_model, self).__init__()

        self.config = config
        self.out_dropout = config["out_dropout"]
        self.classes_num = attributes["classes_num"]
        self.in_dropout = config["in_dropout"]
        embedding_data = attributes["embedding_data"]
        pad_id = attributes["PAD_id"]


        ATT_PAD = -999999
        self.ATT_PAD = T.tensor(ATT_PAD).float()
        self.zeros = T.tensor(0.0)

        if embedding_data is not None:
            embedding_data = T.tensor(embedding_data)
            self.word_embedding = nn.Embedding.from_pretrained(embedding_data,
                                                               freeze=config["word_embd_freeze"],
                                                               padding_idx=pad_id)
        else:
            vocab_len = attributes["vocab_len"]
            self.word_embedding = nn.Embedding(vocab_len, config["embd_dim"],
                                               padding_idx=pad_id)

        self.embd_dim = self.word_embedding.weight.size(-1)
        self.transform_word_dim = Linear(self.embd_dim, config["hidden_size"])

        if not config["global_state_return"]:
            self.attn_linear1 = Linear(config["hidden_size"], config["hidden_size"])
            self.attn_linear2 = Linear(config["hidden_size"], config["hidden_size"])

        self.encoder = encoder(config)

        if config["classifier_layer_num"] == 2:
            self.prediction1 = Linear(config["hidden_size"], config["hidden_size"])
            self.prediction2 = Linear(config["hidden_size"], self.classes_num)
        else:
            self.prediction2 = Linear(config["hidden_size"], self.classes_num)


    # %%

    def embed(self, sequence, input_mask):

        N, S = sequence.size()

        sequence = self.word_embedding(sequence)
        sequence = self.transform_word_dim(sequence)

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

        assert energy.size() == (N, S, D)

        attention = F.softmax(energy + attention_mask, dim=1)

        assert attention.size() == (N, S, D)

        z = T.sum(attention * sequence, dim=1)

        assert z.size() == (N, D)

        return z

    # %%
    def forward(self, batch):

        sequence = batch["sequences_vec"]
        input_mask = batch["input_masks"]

        N = sequence.size(0)

        # EMBEDDING BLOCK
        sequence, input_mask = self.embed(sequence, input_mask)
        sequence = F.dropout(sequence, p=self.in_dropout, training=self.training)

        # ENCODER BLOCK
        sequence_dict = self.encoder(sequence, input_mask)
        sequence = sequence_dict["sequence"]

        penalty = None
        if "penalty" in sequence_dict:
            penalty = sequence_dict["penalty"]

        if self.config["global_state_return"]:
            feats = sequence_dict["global_state"]
        else:
            feats = self.extract_features(sequence, input_mask)

        if self.config["classifier_layer_num"] == 2:
            feats = F.dropout(feats, p=self.out_dropout, training=self.training)
            feats = gelu(self.prediction1(feats))
        feats = F.dropout(feats, p=self.out_dropout, training=self.training)
        logits = self.prediction2(feats)

        assert logits.size() == (N, self.classes_num)

        return {"logits": logits, "penalty": penalty}
