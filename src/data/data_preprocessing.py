from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
import torch
from typing import Dict, Tuple

class BoolQDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512, split: str = "train"):
        self.dataset = load_dataset("google/boolq")[split]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        text = f"Question: {item['question']} Context: {item['passage']}"
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(1 if item["answer"] else 0)
        return encoding

def create_dataloaders(tokenizer: PreTrainedTokenizer, batch_size: int, max_length: int = 512, num_workers: int = 16) -> Tuple[DataLoader, DataLoader]:
    print("INFO: Loading BoolQ dataset")
    train_dataset = BoolQDataset(tokenizer, max_length, "train")
    val_dataset = BoolQDataset(tokenizer, max_length, "validation")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print("INFO: BoolQ dataset loaded")
    return train_loader, val_loader
