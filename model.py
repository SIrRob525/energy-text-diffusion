import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
from dataset import get_dataloader, decode_tokens

class MaskedDiffusionModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_length=1024,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        mask_token_id=None,
        dropout=0.1,
        num_timesteps=1000
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.mask_token_id = mask_token_id
        self.num_timesteps = num_timesteps
        
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=max_seq_length,
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=num_heads,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            use_cache=False
        )
        
        self.transformer = GPT2Model(config)
        self.token_embedding = self.transformer.wte
        self.position_embedding = self.transformer.wpe
        
        self.time_embedding = nn.Embedding(max_seq_length, hidden_size)
        
        self.output_projection = nn.Linear(hidden_size, vocab_size)
    
    def get_alpha_schedule(self, t):
        return 1.0 - (t / (self.num_timesteps - 1))
    
    def forward(self, x, t=None, mask_ratio=None):
        batch_size, seq_length = x.shape
        device = x.device
        
        if t is None:
            t = torch.randint(0, self.num_timesteps + 1, (batch_size,), device=device)
        
        if mask_ratio is None:
            alpha_t = self.get_alpha_schedule(t)
            mask_ratio = 1.0 - alpha_t
        
        mask_ratio = mask_ratio.view(-1, 1)
        
        mask = torch.rand(batch_size, seq_length, device=device) < mask_ratio
        
        x_t = x.clone()
        x_t[mask] = self.mask_token_id
        
        return x_t, mask
    
    def backward(self, x_t, t):
        batch_size, seq_length = x_t.shape
        device = x_t.device
        
        positions = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        
        token_embeddings = self.token_embedding(x_t)
        position_embeddings = self.position_embedding(positions)
        
        t_normalized = (t.float() / self.num_timesteps).clamp(0, 1)
        t_embedding_idx = (t_normalized * (self.max_seq_length - 1)).long()
        t_expanded = t_embedding_idx.view(-1, 1).expand(-1, seq_length)
        time_embeddings = self.time_embedding(t_expanded)
        
        embeddings = token_embeddings + position_embeddings + time_embeddings
        
        transformer_outputs = self.transformer(inputs_embeds=embeddings)
        hidden_states = transformer_outputs.last_hidden_state
        
        logits = self.output_projection(hidden_states)
        
        return logits
    
    def sample(self, logits, x_t, temperature=1.0):
        mask = (x_t == self.mask_token_id)
        probs = F.softmax(logits / temperature, dim=-1)
        
        x_0_pred = x_t.clone()
        
        batch_size, seq_length = x_t.shape
        for b in range(batch_size):
            for j in range(seq_length):
                if mask[b, j]:
                    token_probs = probs[b, j]
                    x_0_pred[b, j] = torch.multinomial(token_probs, 1).item()
        
        return x_0_pred
    
    def generate(self, batch_size=1, temperature=1.0, device="cuda", return_intermediates=False):
        seq_length = self.max_seq_length
        
        x_t = torch.full((batch_size, seq_length), self.mask_token_id, dtype=torch.long, device=device)
        
        intermediates = [None] * self.num_timesteps if return_intermediates else None
        if return_intermediates:
            intermediates[self.num_timesteps-1] = x_t.clone()
        
        for i in range(self.num_timesteps-1, -1, -1):
            t_current = torch.full((batch_size,), i, device=device)
            
            with torch.no_grad():
                logits = self.backward(x_t, t_current)
                x_0_pred = self.sample(logits, x_t, temperature)
                
                if i > 0:
                    t_next = torch.full((batch_size,), i-1, device=device)
                    x_t, _ = self.forward(x_0_pred, t_next)
                else:
                    x_t = x_0_pred
            
            if return_intermediates and i > 0:
                intermediates[i-1] = x_t.clone()
        
        if return_intermediates:
            return x_t, intermediates
        else:
            return x_t

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataloader, tokenizer, mask_token_id = get_dataloader(batch_size=1, sequence_length=64, split="validation")
    
    model = MaskedDiffusionModel(
        vocab_size=len(tokenizer),
        max_seq_length=64,
        hidden_size=128,
        num_layers=2,
        num_heads=2,
        mask_token_id=mask_token_id,
        num_timesteps=10
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

        print("\n--- Generation process ---\n")
        
        _, intermediates = model.generate(batch_size=1, temperature=1.0, device=device, return_intermediates=True)
        
        steps_to_show = [9, 5, 1, 0]
        for step in steps_to_show:
            step_text = decode_tokens(intermediates[step][0], tokenizer)
            mask_count = (intermediates[step][0] == mask_token_id).sum().item()
            mask_percentage = (mask_count / intermediates[step][0].size(0)) * 100
            
            print(f"Step {step}: {step_text}")
        
        break
