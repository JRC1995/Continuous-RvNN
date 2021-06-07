class FOCN_hyperconfig:
    def __init__(self):
        super().__init__()
        self.in_dropout = [0.1, 0.2, 0.3, 0.4]
        self.out_dropout = [0.1, 0.2, 0.3, 0.4]
        self.hidden_dropout = [0.1, 0.2, 0.3, 0.4]

        # epochs = 10
        # limit = 50000


    def process_config(self, config):
        return config


class FOCN_LSTM_hyperconfig:
    def __init__(self):
        super().__init__()
        self.in_dropout = [0.1, 0.2, 0.3, 0.4]
        self.out_dropout = [0.1, 0.2, 0.3, 0.4]
        self.hidden_dropout = [0.1, 0.2, 0.3, 0.4]
        # epochs = 10
        # limit = 50000


    def process_config(self, config):
        return config


