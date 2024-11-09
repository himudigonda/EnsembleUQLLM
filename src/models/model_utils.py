import torch
from transformers import LlamaForSequenceClassification, LlamaTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass
import yaml
import os
from typing import List, Optional, Tuple
from transformers import AutoTokenizer, LlamaForSequenceClassification


@dataclass
class ModelConfig:
    model_name: str
    num_labels: int
    max_length: int
    batch_size: int
    learning_rate: float
    num_epochs: int
    gradient_accumulation_steps: int
    warmup_steps: int
    weight_decay: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    seeds: List[int]
    num_ensemble_models: int  # Add this field
    output_dir: str
    cache_dir: str
    use_bf16: bool
    num_gpus: int

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
def setup_model_and_tokenizer(config: ModelConfig, seed: Optional[int] = None):
    if seed is not None:
        torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=False)

    # Explicitly set a padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = LlamaForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        torch_dtype=torch.bfloat16 if config.use_bf16 else torch.float32,
        device_map="auto"
    )

    # Resize token embeddings to accommodate the new special tokens
    model.resize_token_embeddings(len(tokenizer))

    # Add LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "v_proj"]
    )

    model = get_peft_model(model, peft_config)

    # Ensure model recognizes the padding token
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

def save_model(model, output_dir: str):
    """Save the PEFT model to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

def load_model(config: ModelConfig, model_path: str) -> LlamaForSequenceClassification:
    """Load a trained model from the specified directory."""
    model, _ = setup_model_and_tokenizer(config)
    model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))
    return model
