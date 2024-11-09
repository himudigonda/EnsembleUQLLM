import torch
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
import os
import logging

from ..models.model_utils import save_model, ModelConfig

logger = logging.getLogger(__name__)

class LlamaTrainer:
    def __init__(self, model, train_loader, val_loader, config: ModelConfig, seed: int):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.seed = seed

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        num_training_steps = len(train_loader) * config.num_epochs
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps
        )
        self.scaler = torch.cuda.amp.GradScaler() if config.use_bf16 else None

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader, desc="Training"):
            batch = {k: v.to(self.model.device) for k, v in batch.items()}

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16 if self.config.use_bf16 else torch.float16):
                outputs = self.model(**batch)
                loss = outputs.loss

            self.optimizer.zero_grad()

            if self.config.use_bf16:
                # Skip scaler when using bf16
                loss.backward()
                self.optimizer.step()
            else:
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

            self.scheduler.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(self.val_loader, desc="Evaluating"):
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

        accuracy = correct / total
        return {"loss": total_loss / len(self.val_loader), "accuracy": accuracy}

    def train(self):
        logger.info(f"Training with seed {self.seed}")
        metrics_history = []
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch()
            eval_metrics = self.evaluate()

            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                **eval_metrics
            }
            metrics_history.append(metrics)
            logger.info(f"Epoch {epoch} metrics: {metrics}")

        save_model(self.model, os.path.join(self.config.output_dir, f"seed_{self.seed}"))
        return metrics_history
