import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import datasets  
from hf_token import TOKEN

def load_dataset(split="train", token=TOKEN):
    return datasets.load_dataset("wikitext", "wikitext-103-v1", split=split, token=token)

class WikiTextDataset(Dataset):
    def __init__(self, split="train", sequence_length=1024, token=TOKEN):
        self.sequence_length = sequence_length
        self.raw_data = load_dataset(split=split, token=token)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", token=token)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        
        self.passages = []
        for item in self.raw_data["text"]:
            if item.strip():
                encoded = self.tokenizer.encode(item)
                if len(encoded) > 0:  
                    self.passages.append(encoded)
        
        
        min_passage_length = 10 
        self.passages = [p for p in self.passages if len(p) >= min_passage_length]
    
    def __len__(self):
        return len(self.passages)
    
    def __getitem__(self, idx):
        
        passage = self.passages[idx]
        
        
        if len(passage) > self.sequence_length:
            
            sequence = passage[:self.sequence_length]
        else:
            
            sequence = passage
            padding = [self.tokenizer.pad_token_id] * (self.sequence_length - len(passage))
            sequence = sequence + padding
        
        return torch.tensor(sequence, dtype=torch.long)

def get_dataloader(batch_size=32, sequence_length=1024, split="validation", num_workers=4, shuffle=True):
    dataset = WikiTextDataset(split=split, sequence_length=sequence_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, dataset.tokenizer

def decode_tokens(tokens, tokenizer):
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    
    return tokenizer.decode(tokens)

if __name__ == "__main__":

    dataloader, tokenizer = get_dataloader(batch_size=1, sequence_length=64, split="validation")
    

    for batch in dataloader:

        print(f"Batch shape: {batch.shape}")
        
        print(f"Batch: {batch}")
        
        sample_text = decode_tokens(batch[0], tokenizer)
        print(f"Sample text: {sample_text}")

        break
