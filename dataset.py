import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import datasets  
from hf_token import TOKEN
from tqdm import tqdm

def load_dataset(token=TOKEN):
    return datasets.load_dataset("Skylion007/openwebtext", split="train", token=token, trust_remote_code=True)

def add_mask_token(tokenizer):
    mask_token = "[MASK]"
    
    if mask_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})
    
    mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
    
    return tokenizer, mask_token_id

class OpenWebTextDataset(Dataset):
    def __init__(self, sequence_length=1024, token=TOKEN):
        self.sequence_length = sequence_length
        self.raw_data = load_dataset(token=token)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", token=token)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tokenizer, self.mask_token_id = add_mask_token(self.tokenizer)
    
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):

        passage = self.raw_data[idx]["text"]
        sequence = self.tokenizer.encode(passage)[:self.sequence_length]
        padding = [self.tokenizer.pad_token_id] * (self.sequence_length - len(sequence))
        sequence = sequence + padding
        
        return torch.tensor(sequence, dtype=torch.long)

def get_dataloader(batch_size=32, sequence_length=1024, num_workers=4, shuffle=True):
    dataset = OpenWebTextDataset(sequence_length=sequence_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, dataset.tokenizer, dataset.mask_token_id

def decode_tokens(tokens, tokenizer):
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    
    return tokenizer.decode(tokens)

if __name__ == "__main__":
    dataloader, tokenizer, mask_token_id = get_dataloader(batch_size=1, sequence_length=128)
    
    print(f"Mask token ID: {mask_token_id}")
    print(f"Mask token: {tokenizer.convert_ids_to_tokens(mask_token_id)}\n")

    for batch in dataloader:
        print(f"Batch shape: {batch.shape}\n")
        
        print(f"Batch: {batch}\n")
        
        sample_text = decode_tokens(batch[0], tokenizer)
        print(f"Sample text: {sample_text}\n")
        
        masked_sequence = batch[0].clone()
        mask_indices = torch.randint(0, masked_sequence.size(0), (masked_sequence.size(0) // 4,))
        masked_sequence[mask_indices] = mask_token_id
        
        masked_text = decode_tokens(masked_sequence, tokenizer)
        print(f"Masked text: {masked_text}")

        break
