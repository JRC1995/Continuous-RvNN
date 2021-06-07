
class CRvNN_hyperconfig:
    def __init__(self):
        super().__init__()
        self.in_dropout = [0.1, 0.2, 0.3]
        self.out_dropout = [0.1, 0.2, 0.3]
        self.hidden_dropout = [0.1, 0.2, 0.3]
        self.max_trials = 20
        self.allow_repeat = False
        self.hyperalgo = "hyperopt.tpe.suggest"
        self.epochs = 7
        self.limit = -1

    def process_config(self, config):
        return config


class LSTM_hyperconfig:
    def __init__(self):
        super().__init__()
        self.in_dropout = [0.1, 0.2, 0.3]
        self.out_dropout = [0.1, 0.2, 0.3]
        self.max_trials = 20
        self.allow_repeat = False
        self.hyperalgo = "hyperopt.tpe.suggest"
        self.epochs = 7
        self.limit = -1

    def process_config(self, config):
        return config

