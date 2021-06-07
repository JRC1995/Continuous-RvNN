class FOCN_hyperconfig:
    def __init__(self):
        super().__init__()
        self.in_dropout = [0.2, 0.3, 0.4]
        self.out_dropout = [0.2, 0.3, 0.4]
        self.hidden_dropout = [0.2, 0.3, 0.4]
        self.bucket_size_factor = [10]
        self.train_batch_size = [64]

        # epochs = 3
        # limit = -1

    def process_config(self, config):
        return config

