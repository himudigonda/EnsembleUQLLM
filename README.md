# EnsembleUQLLM

EnsembleUQLLM is a robust and scalable framework designed for training and evaluating ensemble models for uncertainty quantification, leveraging the LLaMA architecture. It uses Low-Rank Adaptation (LoRA) to achieve efficient and lightweight fine-tuning.

## Features

- **Ensemble Modeling**: Train multiple models with different random seeds to quantify uncertainty effectively.
- **Low-Rank Adaptation (LoRA)**: Achieve efficient fine-tuning with minimal computational overhead.
- **Uncertainty Metrics**: Compute essential uncertainty metrics such as variance, entropy, and mutual information.
- **BoolQ Dataset Integration**: Pre-configured for binary classification tasks using the BoolQ dataset.
- **Configurable Pipeline**: Easily customize training and model parameters through YAML configuration files.

## Requirements

- Python 3.8+
- CUDA-compatible GPU
- Dependencies listed in `requirements.txt`

To install dependencies, run:

```sh
pip install -r requirements.txt
```

## Directory Structure

The project is organized as follows:

```
.
├── configs/
│   └── llama_config.yaml   # YAML configuration for model and training
├── src/
│   ├── data/               # Data loading and preprocessing scripts
│   ├── inference/          # Inference and uncertainty metrics modules
│   ├── models/             # Model setup utilities and configurations
│   ├── training/           # Training logic and scripts
│   └── utils/              # General utilities used throughout the project
├── scripts/
│   ├── train_models.sh     # Script to train ensemble models
│   └── run_inference.sh    # Script to run inference on the validation set
├── main.py                 # Entry point for training and inference
├── train.sh                # Shell script for running ensemble training
├── infer.py                # Shell script for running inference
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

## Configuration

All configurations are managed through `configs/llama_config.yaml`. This YAML file contains settings for the model, training, and uncertainty quantification. Key parameters include:

```yaml
# Model configuration
model_name: "meta-llama/Llama-3.2-1B"
num_labels: 2
max_length: 512

# Training configuration
batch_size: 8
learning_rate: 2e-4
num_epochs: 3
gradient_accumulation_steps: 4
warmup_steps: 100
weight_decay: 0.01

# LoRA configuration
lora_r: 8
lora_alpha: 32
lora_dropout: 0.1

# Uncertainty configuration
seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
num_ensemble_models: 10

# Paths
output_dir: "model_outputs"
cache_dir: "cache"

# Hardware configuration
use_bf16: true
num_gpus: 4
```

### Key Configuration Details:
- **Model Settings**: Defines model name, number of labels, and input sequence length.
- **Training Parameters**: Configures batch size, learning rate, and number of epochs.
- **LoRA Parameters**: Defines settings for efficient adaptation using LoRA.
- **Ensemble Settings**: Specifies the number of models and random seeds used for uncertainty quantification.

## Training

To train an ensemble of models using different seeds, run the following script:

```sh
sh train.sh
```

This script iterates over the seeds defined in the configuration file, training models sequentially.

## Inference

To run inference with the trained ensemble models and calculate uncertainty metrics:

```sh
sh infer.py
```

## Uncertainty Metrics

The following metrics are used to quantify uncertainty:

- **Variance**: Measures the spread of predictions across ensemble models.
- **Entropy**: Represents the confidence of predictions.
- **Mutual Information**: Indicates epistemic uncertainty by comparing prediction entropy to the average entropy across models.

<!-- ### Sample Output:
```json
{
  "AUC": 0.92,
  "Average Precision": 0.88,
  "Average Entropy": 0.34
}
``` -->

## Contributing

Contributions are welcome! Please feel free to submit a pull request or raise an issue for bug reports or feature requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- **Meta's LLaMA**: For the foundational LLaMA architecture.
- **Hugging Face Transformers**: For providing essential model utilities.
- **LoRA: Low-Rank Adaptation**: Enabling efficient model adaptation for diverse tasks.
