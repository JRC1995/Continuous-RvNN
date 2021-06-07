class CRvNN_hyperconfig:
    def __init__(self):
        super().__init__()
        self.in_dropout = [0.1, 0.2, 0.3, 0.4]
        self.out_dropout = [0.1, 0.2, 0.3, 0.4]
        self.hidden_dropout = [0.1, 0.2, 0.3, 0.4]
        self.epochs = 5
        self.limit = 100000
        self.max_trials = 20
        self.allow_repeat = False
        self.hyperalgo = "hyperopt.tpe.suggest"

    def process_config(self, config):
        return config


class ordered_memory_hyperconfig:
    def __init__(self):
        super().__init__()
        self.in_dropout = [0.1, 0.2, 0.3, 0.4]
        self.out_dropout = [0.1, 0.2, 0.3, 0.4]
        self.memory_dropout = [0.1, 0.2, 0.3, 0.4]
        self.dropout = [0.1, 0.2, 0.3, 0.4]
        self.epochs = 5
        self.limit = 100000
        self.max_trials = 20
        self.allow_repeat = False
        self.hyperalgo = "hyperopt.tpe.suggest"

    def process_config(self, config):
        return config