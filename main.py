import argparse
from src.models.model_utils import ModelConfig, setup_model_and_tokenizer
from src.training.trainer import LlamaTrainer
from src.data.data_preprocessing import create_dataloaders
from src.inference.ensemble_inference import EnsembleInference
from src.inference.uncertainty_metrics import calculate_uncertainty_metrics
import os

def train(config_path, seed):
    print(f"Training with seed: {seed}")
    config = ModelConfig.from_yaml(config_path)
    print("INFO: Config loaded")
    model, tokenizer = setup_model_and_tokenizer(config, seed)
    print("INFO: Model and tokenizer loaded")
    train_loader, val_loader = create_dataloaders(tokenizer, config.batch_size, config.max_length)
    print("INFO: Training data loaded")
    trainer = LlamaTrainer(model, train_loader, val_loader, config, seed)
    print("INFO: Trainer loaded")
    trainer.train()
    print("INFO: Training finished")

def inference(config_path):
    print("Running inference")
    config = ModelConfig.from_yaml(config_path)
    print("INFO: Config loaded")

    # Check if all models are trained
    for seed in config.seeds:
        model_path = os.path.join(config.output_dir, f"seed_{seed}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model for seed {seed} not found. Please train the model before running inference.")

    # Create tokenizer
    _, tokenizer = setup_model_and_tokenizer(config)

    # Create data loaders
    _, val_loader = create_dataloaders(tokenizer, config.batch_size, config.max_length)
    print("INFO: Validation data loaded")

    # Load models based on seeds specified in the configuration
    ensemble = EnsembleInference(config)
    print("INFO: Ensemble with models loaded based on seeds: ", config.seeds)

    # Get ensemble predictions
    results = ensemble.get_ensemble_predictions(val_loader)
    print("INFO: Ensemble predictions finished")

    # Calculate uncertainty metrics
    metrics = calculate_uncertainty_metrics(results)
    print("INFO: Uncertainty metrics finished")
    print(metrics)
    print("INFO: Inference finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "inference"], help="Mode: train or inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--seed", type=int, help="Seed for training")
    print("INFO: Arguments parsed")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.config, args.seed)
    elif args.mode == "inference":
        inference(args.config)
