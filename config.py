import torch

class EDLMConfig:
    def __init__(self):
        self.batch_size = 100
        self.sequence_length = 128
        self.hidden_size = 768
        self.num_layers = 12
        self.num_heads = 12
        self.dropout = 0.1
        self.num_timesteps = 100
        self.importance_sampling_size = 16
        self.importance_sampling_window = 1.0
        self.temperature = 1.0
        
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.warmup_steps = 4000
        self.num_epochs = 1
        self.save_interval = 500
        self.output_dir = "checkpoints"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
