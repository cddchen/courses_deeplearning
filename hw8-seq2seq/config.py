class configurations(object):
    def __init__(self):
        self.batch_size = 60
        self.emb_dim = 256
        self.hid_dim = 512
        self.n_layers = 3
        self.dropout = 0.5
        self.learning_rate = 5e-4
        self.max_output_len = 50
        self.num_steps = 120
        self.store_steps = 30
        self.summary_steps = 30
        self.load_model = False
        self.store_model_path = './ckpt'
        self.load_model_path = None #e.g. "./ckpt/model_{step}"
        self.data_path = './cmn-eng'
        self.attention = False
