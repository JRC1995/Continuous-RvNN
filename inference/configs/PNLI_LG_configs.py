
class optimizer_config:
    def __init__(self):
        # optimizer config
        self.max_grad_norm = 1.0
        self.batch_size = 128
        self.train_batch_size = 128
        self.dev_batch_size = 128
        self.bucket_size_factor = 10
        self.DataParallel = True
        self.weight_decay = 1e-2
        self.lr = 1e-3
        self.epochs = 100
        self.early_stop_patience = 4
        self.scheduler_patience = 2
        self.optimizer = "ranger"
        self.save_by = "accuracy"
        self.metric_direction = 1


class base_config(optimizer_config):
    def __init__(self):
        super().__init__()
        # word embedding
        self.word_embd_freeze = False
        self.left_padded = False
        # hidden size
        self.embd_dim = 200
        self.hidden_size = 200
        self.small_d = 64
        self.cell_hidden_size = 4 * self.hidden_size
        self.window_size = 5
        self.global_state_return = True
        self.transition_features = True
        self.no_modulation = False

        self.in_dropout = 0.1
        self.hidden_dropout = 0.1
        self.out_dropout = 0.3

        self.penalty_gamma = 1.0
        self.speed_gamma = 0.0
        self.halt_gamma = 0.01
        self.sparsity_gamma = 0.001
        self.stop_threshold = 0.01
        self.hidden_activation = "gelu"
        self.early_stopping = True

        self.encoder = "CRvNN"


class CRvNN_config(base_config):
    def __init__(self):
        super().__init__()
        self.encoder = "CRvNN"
        self.model_name = "(CRvNN)"

class CRvNN_no_transition_config(base_config):
    def __init__(self):
        super().__init__()
        self.encoder = "CRvNN"
        self.transition_features = False
        self.model_name = "(CRvNN no transition)"

class CRvNN_no_modulation_config(base_config):
    def __init__(self):
        super().__init__()
        self.encoder = "CRvNN"
        self.no_modulation = True
        self.model_name = "(CRvNN NO MODULATION)"

class CRvNN_no_gelu_config(base_config):
    def __init__(self):
        super().__init__()
        self.encoder = "CRvNN"
        self.hidden_activation = "relu"
        self.model_name = "(CRvNN NO GELU)"


class CRvNN_no_entropy_config(base_config):
    def __init__(self):
        super().__init__()
        self.encoder = "CRvNN"
        self.entropy_gamma = 0.0
        self.model_name = "(CRvNN NO ENTROPY)"

class CRvNN_LSTM_config(base_config):
    def __init__(self):
        super().__init__()
        self.encoder = "CRvNN_LSTM"
        self.model_name = "(CRvNN_LSTM)"

class LR_CRvNN_config(base_config):
    def __init__(self):
        super().__init__()
        self.encoder = "LR_CRvNN"
        self.model_name = "(Left to Right CRvNN)"

class CRvNN2_config(base_config):
    def __init__(self):
        super().__init__()
        self.encoder = "CRvNN2"
        self.model_name = "(CRvNN2)"


class LSTM_config(base_config):
    def __init__(self):
        super().__init__()
        self.global_state_return = False
        self.batch_pair = True
        self.in_dropout = 0.2
        self.out_dropout = 0.1
        self.encoder = "LSTM"
        self.model_name = "(LSTM)"

class ordered_memory_config(base_config):
    def __init__(self):
        super().__init__()
        self.batch_pair = True
        self.bucket_size_factor = 10
        self.dropout = 0.2
        self.output_last = False
        self.left_padded = False
        self.memory_dropout = 0.2
        self.in_dropout = 0.1
        self.out_dropout = 0.3
        self.memory_slots = 12
        self.double_slots_during_val = True
        self.hidden_size = 200
        self.bidirection = False
        self.encoder = "ordered_memory"
        self.model_name = "(ordered_memory)"
        self.optimizer = "adam_"
        self.weight_decay = 1.2e-6
        self.max_grad_norm = 1
