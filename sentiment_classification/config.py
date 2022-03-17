class Config:
    def __init__(self, args):
        self.args = args
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.train_epochs = args.train_epochs
        self.eval_epochs = args.eval_epochs
        self.log_steps = args.log_steps
        self.max_len = args.max_len
        self.corpus_tag_or_dir = args.corpus_tag_or_dir
        self.pretrain_model_path = args.pretrain_model_path
        self.num_labels = args.num_labels
        self.model_save_path = args.model_save_path
        self.log_save_path = args.log_save_path
