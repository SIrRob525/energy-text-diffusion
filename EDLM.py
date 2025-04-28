import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from dataset import get_dataloader, decode_tokens
from model import MaskedDiffusionModel
from hf_token import TOKEN

class EnergyDiffusionModel(MaskedDiffusionModel):
    def __init__(
        self,
        vocab_size,
        max_seq_length=1024,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        mask_token_id=None,
        dropout=0.1,
        num_timesteps=1000,
        ar_model_name="gpt2",
        importance_sampling_size=16,
        importance_sampling_window=1.0,
        temperature=1.0
    ):
        super().__init__(
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mask_token_id=mask_token_id,
            dropout=dropout,
            num_timesteps=num_timesteps
        )
        
        self.ar_model = GPT2LMHeadModel.from_pretrained(ar_model_name, token=TOKEN)
        self.ar_model.eval()
        
        self.importance_sampling_size = importance_sampling_size
        self.importance_sampling_window = importance_sampling_window
        self.temperature = temperature
    
    def compute_energy(self, x_0_samples, x_t, t):
        with torch.no_grad():
            batch_size, num_samples, seq_len = x_0_samples.shape
            
            diffusion_logits = self.backward(x_t, t)
            diffusion_log_probs = F.log_softmax(diffusion_logits, dim=-1)
            
            ar_log_probs = torch.zeros(batch_size, num_samples, device=x_0_samples.device)
            
            for i in range(num_samples):
                sample_x_0 = x_0_samples[:, i, :]
                ar_outputs = self.ar_model(sample_x_0, labels=sample_x_0)
                ar_log_probs[:, i] = -ar_outputs.loss
            
            diffusion_log_probs_expanded = diffusion_log_probs.unsqueeze(1).expand(
                batch_size, num_samples, seq_len, diffusion_log_probs.size(-1)
            )
            
            indices = x_0_samples.unsqueeze(-1)
            diffusion_log_probs_gathered = torch.gather(
                diffusion_log_probs_expanded, 
                dim=3, 
                index=indices
            ).squeeze(-1)
            
            diffusion_log_probs_sum = diffusion_log_probs_gathered.sum(dim=2)
        
            energy = -ar_log_probs + diffusion_log_probs_sum
            
        return energy
    
    def sample(self, logits, x_t, t=None, temperature=1.0):
        batch_size, seq_length, _ = logits.shape
        
        t_normalized = t.float() / (self.num_timesteps - 1)
        use_importance_sampling = t_normalized >= (1.0 - self.importance_sampling_window)
        
        if not use_importance_sampling:
            return super().sample(logits, x_t, t, temperature)
        
        mask = (x_t == self.mask_token_id)
        probs = F.softmax(logits / temperature, dim=-1)
        
        flat_probs = probs.reshape(-1, probs.size(-1))
        flat_mask = mask.reshape(-1)
        masked_indices = torch.nonzero(flat_mask).squeeze(1)
        
        if masked_indices.numel() == 0:
            return x_t.clone()
        
        samples = []
        for _ in range(self.importance_sampling_size):
            x_0_pred = x_t.clone()
            flat_x_0_pred = x_0_pred.reshape(-1)
            
            sampled_tokens = torch.multinomial(flat_probs[masked_indices], 1).squeeze(1)
            flat_x_0_pred[masked_indices] = sampled_tokens
            
            samples.append(x_0_pred)
        
        samples = torch.stack(samples, dim=1)
        
        energies = self.compute_energy(samples, x_t, t)
        weights = F.softmax(-energies, dim=1)
        
        selected_indices = torch.multinomial(weights, 1).squeeze(1)
        
        selected_samples = torch.zeros_like(x_t)
        for b in range(batch_size):
            selected_samples[b] = samples[b, selected_indices[b]]
        
        return selected_samples
    

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataloader, tokenizer, mask_token_id = get_dataloader(batch_size=1, sequence_length=64)
    
    model = EnergyDiffusionModel(
        vocab_size=len(tokenizer),
        max_seq_length=64,
        hidden_size=128,
        num_layers=2,
        num_heads=2,
        mask_token_id=mask_token_id,
        num_timesteps=10,
        ar_model_name="gpt2",
        importance_sampling_size=4,
        importance_sampling_window=0.5,
        temperature=1.0
    ).to(device)
    
    for batch in dataloader:
        batch = batch.to(device)
        
        print(f"Original text: {decode_tokens(batch[0], tokenizer)}")
        
        print("\n--- Forward process ---\n")
        for t in [1, 5, 9]:
            t_tensor = torch.tensor([t], device=device)
            x_t, mask = model.forward(batch, t_tensor)
            noised_text = decode_tokens(x_t[0], tokenizer)
            mask_percentage = mask.float().mean().item() * 100
            
            print(f"Step {t}: {noised_text}")
        
        print("\n--- Backward process ---\n")
        t_tensor = torch.tensor([t], device=device)
        x_t, mask = model.forward(batch, t_tensor)
        noised_text = decode_tokens(x_t[0], tokenizer)
        logits = model.backward(x_t, t_tensor)
        print(f"Logits shape: {logits.shape}")
        
        print("\n--- Energy calculation ---\n")
        x_0_pred = model.sample(logits, x_t, t_tensor)
        
        samples = []
        for i in range(3):
            x_0_sample = model.sample(logits, x_t, t_tensor)
            samples.append(x_0_sample)
        
        samples_tensor = torch.stack(samples, dim=1)
        print(f"Samples tensor shape: {samples_tensor.shape}")
        energies = model.compute_energy(samples_tensor, x_t)
        print(f"Energy shape: {energies.shape}")
        
        print("\n--- Importance sampling ---\n")
        for i in range(3):
            print(f"Sample {i+1} (energy = {energies[0, i].item():.2f}): {decode_tokens(samples_tensor[0, i], tokenizer)}")
            print()

        print("\n--- Generation process ---\n")
        
        generated = model.generate(batch_size=1, temperature=1.0, device=device)
        
        print(f"Final generated text: {decode_tokens(generated[0], tokenizer)}")
        
        break
