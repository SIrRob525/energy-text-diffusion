import torch

class EDLMConfig:
    def __init__(self):
        self.batch_size = 100
        self.sequence_length = 128
        self.hidden_size = 768
        self.num_layers = 12
        self.num_heads = 12
        self.dropout = 0.1
        self.num_timesteps = 1024
        self.importance_sampling_size = 16
        self.importance_sampling_window = 0.4
        self.temperature = 1.0
        
        self.learning_rate = 5e-5
        self.weight_decay = 0.001
        self.warmup_steps = 0
        self.num_epochs = 1
        self.save_interval = 500
        self.output_dir = "checkpoints"
        self.init_dir = "checkpoints"
        
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
