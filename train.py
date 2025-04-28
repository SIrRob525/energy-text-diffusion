import os
import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from dataset import get_dataloader
from EDLM import EnergyDiffusionModel
from config import EDLMConfig
from tqdm import tqdm

def train(config):
    device = config.device
    
    train_dataloader, tokenizer, mask_token_id = get_dataloader(
        batch_size=config.batch_size,
        sequence_length=config.sequence_length
    )
    
    model = EnergyDiffusionModel(
        vocab_size=len(tokenizer),
        max_seq_length=config.sequence_length,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        mask_token_id=mask_token_id,
        dropout=config.dropout,
        num_timesteps=config.num_timesteps,
        importance_sampling_size=config.importance_sampling_size,
        importance_sampling_window=config.importance_sampling_window,
        temperature=config.temperature
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_dataloader) * config.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    model.train()
    iteration = 0
    best_loss = float("inf")
    
    for epoch in range(config.num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for _, batch in enumerate(progress_bar):
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            timesteps = torch.randint(
                0, config.num_timesteps, 
                (batch.size(0),), 
                device=device
            )
            
            noised_batch, _ = model.forward(batch, timesteps)
            
            logits = model.backward(noised_batch, timesteps)
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch.view(-1)
            )
            
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            iteration += 1
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")


            if iteration % config.save_interval == 0:
                save_path = os.path.join(config.output_dir, f"last_model.pt")
                torch.save(model.state_dict(), save_path)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(model.state_dict(), os.path.join(config.output_dir, f"best_model.pt"))

def main():
    config = EDLMConfig()
    
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    
    train(config)

if __name__ == "__main__":
    main()
