import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple

from ..models.model_utils import ModelConfig, load_model

class EnsembleInference:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = [load_model(config, f"{config.output_dir}/seed_{seed}") for seed in config.seeds]

    @torch.no_grad()
    def predict_proba(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        all_preds = []
        labels = []

        for model in self.models:
            model.eval()
            model_preds = []

            for batch in tqdm(dataloader, desc="Predicting"):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                probs = torch.softmax(outputs.logits, dim=-1)
                model_preds.extend(probs.cpu().numpy())
                if not labels:
                    labels.extend(batch["labels"].cpu().numpy())

            all_preds.append(model_preds)

        return np.array(all_preds), np.array(labels)

    def get_ensemble_predictions(self, dataloader) -> Dict[str, np.ndarray]:
        preds, labels = self.predict_proba(dataloader)
        mean_preds = preds.mean(axis=0)
        variance = preds.var(axis=0)
        entropy = -np.sum(mean_preds * np.log(mean_preds + 1e-10), axis=-1)
        mutual_info = entropy - np.mean(-np.sum(preds * np.log(preds + 1e-10), axis=-1), axis=0)

        return {
            "predictions": preds,
            "mean_predictions": mean_preds,
            "variance": variance,
            "entropy": entropy,
            "mutual_information": mutual_info,
            "labels": labels
        }
