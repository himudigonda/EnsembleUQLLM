import torch
import numpy as np
from tqdm import tqdm
from typing import Dict
from ..models.model_utils import ModelConfig, load_model

class EnsembleInference:
    def __init__(self, config: ModelConfig):
        self.config = config
        # Load models using the seeds specified in the configuration
        self.models = [load_model(config, f"{config.output_dir}/seed_{seed}") for seed in config.seeds]
        print(f"INFO: Loaded models based on seeds: {config.seeds}")

    @torch.no_grad()
    def predict_proba(self, dataloader):
        model_preds = []
        labels = []

        for batch in tqdm(dataloader, desc="Predicting with Ensemble"):
            # Extract input_ids, attention_mask, and labels from the batch dictionary
            inputs = batch['input_ids']
            attention_mask = batch['attention_mask']
            label = batch['labels']

            # print("INFO: Batch inputs: ", inputs)
            # print("INFO: Batch attention_mask: ", attention_mask)
            # print("INFO: Batch labels: ", label)

            # Collect predictions from each model in the ensemble
            batch_preds = []

            for model in self.models:
                model.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    # Move inputs to the device (e.g., GPU) where the model is located
                    inputs = inputs.to(model.device)
                    attention_mask = attention_mask.to(model.device)

                    # Perform forward pass through the model
                    outputs = model(input_ids=inputs, attention_mask=attention_mask)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    # print(f"INFO: Model {model.device} outputs: ", outputs.logits)
                    # print(f"INFO: Model {model.device} probabilities: ", probs)
                    batch_preds.append(probs)

            # Stack the predictions from all models for uncertainty calculation
            stacked_preds = torch.stack(batch_preds)
            # print("INFO: Stacked model predictions: ", stacked_preds)

            # Get the mean prediction across the models for each sample
            avg_probs = torch.mean(stacked_preds, dim=0)
            model_preds.extend(avg_probs.to(torch.float32).cpu().numpy())  # Convert to Float32 before moving to CPU and NumPy
            labels.extend(label.numpy())

        # print("INFO: Final model predictions (all models): ", model_preds)
        # print("INFO: Final labels: ", labels)
        return model_preds, labels

    def get_ensemble_predictions(self, dataloader):
        # Get predictions and labels from predict_proba()
        preds, labels = self.predict_proba(dataloader)

        # Convert preds to a numpy array
        preds = np.array(preds)  # Shape: (num_samples, num_classes)
        # print("INFO: Predictions array: ", preds)

        # Calculate the mean and variance across ensemble models for uncertainty
        mean_preds = preds.mean(axis=1)  # Averaging over the output probabilities for each sample
        variance = preds.var(axis=1)  # Variance over the output probabilities for each sample

        print("INFO: Mean predictions: ", mean_preds)
        print("INFO: Variance: ", variance)

        # Calculate entropy and mutual information as measures of uncertainty
        entropy = -np.sum(mean_preds * np.log(mean_preds + 1e-10), axis=-1)
        mutual_info = entropy - np.mean(-np.sum(preds * np.log(preds + 1e-10), axis=-1), axis=0)

        print("INFO: Entropy: ", entropy)
        print("INFO: Mutual information: ", mutual_info)

        return {
            "predictions": preds,
            "mean_predictions": mean_preds,
            "variance": variance,
            "entropy": entropy,
            "mutual_information": mutual_info,
            "labels": labels
        }
