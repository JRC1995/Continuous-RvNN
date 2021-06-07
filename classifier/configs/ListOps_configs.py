class optimizer_config:
    def __init__(self):
        # optimizer config
        self.max_grad_norm = 1
        self.batch_size = 128
        self.train_batch_size = 128
        self.dev_batch_size = 128
        self.bucket_size_factor = 10
        self.DataParallel = True
        self.weight_decay = 1e-2
        self.lr = 1e-3
        self.epochs = 100
        self.early_stop_patience = 10
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
        self.embd_dim = 128
        self.hidden_size = 128
        self.cell_hidden_size = 4 * 128
        self.small_d = 64
        self.window_size = 5
        self.global_state_return = True
        self.recurrent_momentum = True
        self.no_modulation = False

        self.in_dropout = 0.3
        self.hidden_dropout = 0.1
        self.out_dropout = 0.2

        self.penalty_gamma = 1.0
        self.speed_gamma = 0.0
        self.entropy_gamma = 0.01
        self.stop_threshold = 0.01
        self.hidden_activation = "gelu"
        self.classifier_layer_num = 1
        self.early_stopping = True

        self.encoder = "FOCN"

class FOCN_config(base_config):
    def __init__(self):
        super().__init__()
        self.encoder = "FOCN"
        self.model_name = "(FOCN)"

class FOCN_LSTM_config(base_config):
    def __init__(self):
        super().__init__()
        self.in_dropout = 0.3
        self.hidden_dropout = 0.1
        self.out_dropout = 0.2
        self.encoder = "FOCN_LSTM"
        self.model_name = "(FOCN_LSTM)"

class LR_FOCN_config(base_config):
    def __init__(self):
        super().__init__()
        self.encoder = "LR_FOCN"
        self.model_name = "(Left to Right FOCN)"

class FOCN_no_recurrency_bias_config(base_config):
    def __init__(self):
        super().__init__()
        self.encoder = "FOCN"
        self.recurrent_momentum = False
        self.model_name = "(FOCN NO RECURRENCY BIAS)"

class FOCN_no_modulation_config(base_config):
    def __init__(self):
        super().__init__()
        self.encoder = "FOCN"
        self.no_modulation = True
        self.model_name = "(FOCN NO MODULATION)"

class FOCN_no_gelu_config(base_config):
    def __init__(self):
        super().__init__()
        self.encoder = "FOCN"
        self.hidden_activation = "relu"
        self.model_name = "(FOCN NO GELU)"


class FOCN_no_entropy_config(base_config):
    def __init__(self):
        super().__init__()
        self.encoder = "FOCN"
        self.entropy_gamma = 0.0
        self.model_name = "(FOCN NO ENTROPY)"

class ordered_memory_config(base_config):
    def __init__(self):
        super().__init__()
        self.dropout = 0.1
        self.output_last = True
        self.left_padded = True
        self.memory_dropout = 0.3
        self.left_padded = True
        self.memory_slots = 21
        self.hidden_size = 128
        self.bidirection = False
        self.encoder = "ordered_memory"
        self.model_name = "(ordered_memory)"
        self.optimizer = "adam_"
        self.weight_decay = 1.2e-6
