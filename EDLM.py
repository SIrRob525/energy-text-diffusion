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
    
    def compute_energy(self, x_0_samples, x_t):
        with torch.no_grad():
            batch_size, num_samples, seq_len = x_0_samples.shape
            
            diffusion_logits = self.backward(x_t, torch.zeros(batch_size, device=x_0_samples.device))
            diffusion_log_probs = F.log_softmax(diffusion_logits, dim=-1)
            
            mask = (x_t == self.mask_token_id)
            
            energies = []
            for i in range(num_samples):
                sample_x_0 = x_0_samples[:, i, :]
                
                ar_outputs = self.ar_model(sample_x_0.unsqueeze(0), labels=sample_x_0.unsqueeze(0))
                ar_log_probs = -ar_outputs.loss
                
                diffusion_log_probs_gathered = torch.gather(
                    diffusion_log_probs,
                    dim=2,
                    index=sample_x_0.unsqueeze(-1)
                ).squeeze(-1)
                
                diffusion_log_probs_masked = diffusion_log_probs_gathered * mask.float()
                diffusion_log_probs_sum = diffusion_log_probs_masked.sum(dim=1)
                
                sample_energy = -ar_log_probs + diffusion_log_probs_sum
                energies.append(sample_energy)
            
            energy = torch.stack(energies, dim=1)
            
        return energy
    
    def sample(self, logits, x_t, t=None, temperature=1.0):
        batch_size, seq_length, _ = logits.shape
        
        t_normalized = t.float() / (self.num_timesteps - 1)
        use_importance_sampling = t_normalized >= (1.0 - self.importance_sampling_window)
        
        if not use_importance_sampling:
            return super().sample(logits, x_t, t, temperature)
        
        mask = (x_t == self.mask_token_id)
        probs = F.softmax(logits / temperature, dim=-1)
        
        samples = []
        for _ in range(self.importance_sampling_size):
            x_0_pred = x_t.clone()
            
            for b in range(batch_size):
                for j in range(seq_length):
                    if mask[b, j]:
                        token_probs = probs[b, j]
                        x_0_pred[b, j] = torch.multinomial(token_probs, 1).item()
            
            samples.append(x_0_pred)
        
        samples = torch.stack(samples, dim=1)
        
        energies = self.compute_energy(samples, x_t)
        weights = F.softmax(-energies, dim=1)
        
        selected_indices = torch.multinomial(weights, 1).squeeze(1)
        
        selected_samples = torch.zeros_like(x_t)
        for b in range(batch_size):
            selected_samples[b] = samples[b, selected_indices[b]]
        
        return selected_samples
    

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataloader, tokenizer, mask_token_id = get_dataloader(batch_size=1, sequence_length=64, split="validation")
    
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
        
        samples_for_energy = []
        for i in range(3):
            x_0_sample = model.sample(logits, x_t, t_tensor)
            samples_for_energy.append(x_0_sample)
        
        samples_tensor = torch.stack(samples_for_energy, dim=1)
        print(f"Samples tensor shape: {samples_tensor.shape}")
        energies = model.compute_energy(samples_tensor, x_t)
        print(f"Energy shape: {energies.shape}")
        
        print("\n--- Importance sampling ---\n")
        for i in range(3):
            print(f"Sample {i+1} (energy = {energies[0, i].item():.2f}): {decode_tokens(samples_tensor[0, i], tokenizer)}")
            print()

        print("\n--- Generation process ---\n")
        
        _, intermediates = model.generate(batch_size=1, device=device, return_intermediates=True)
        
        steps_to_show = [9, 5, 1, 0]
        for step in steps_to_show:
            step_text = decode_tokens(intermediates[step][0], tokenizer)
            mask_count = (intermediates[step][0] == mask_token_id).sum().item()
            mask_percentage = (mask_count / intermediates[step][0].size(0)) * 100
            
            print(f"Step {step}: {step_text}")
        
        break
