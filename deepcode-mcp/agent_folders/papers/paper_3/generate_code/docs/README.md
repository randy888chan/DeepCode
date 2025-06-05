# RecDiff: Social Recommendation with Diffusion-based Graph Denoising

## Overview

RecDiff is a novel social recommendation model that leverages hidden-space diffusion processes for robust graph denoising. The model addresses the noise and sparsity challenges in social recommendation by applying diffusion models to the latent embedding space, enabling more effective learning of user preferences and social relationships.

## Key Features

- **Graph Convolutional Encoders**: Separate GCN encoders for user-item interactions and user-user social networks
- **Latent-Space Diffusion**: Forward and reverse diffusion processes applied to social embeddings
- **Denoising Network**: Neural network for learning the reverse denoising process with timestep conditioning
- **Multi-Objective Training**: Combined ranking loss (BPR), denoising loss, and regularization terms
- **Robust Performance**: Superior performance on datasets with noisy social connections

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd recdiff

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.models.recdiff import RecDiff
from src.utils.data import DataLoader
from src.configs.default import get_config

# Load configuration
config = get_config()

# Load data
data_loader = DataLoader(
    user_item_path="data/user_item.csv",
    user_user_path="data/user_user.csv"
)
interaction_matrix, social_matrix = data_loader.load_data()

# Initialize model
model = RecDiff(
    n_users=data_loader.n_users,
    n_items=data_loader.n_items,
    embed_dim=config.embed_dim,
    n_layers=config.n_layers,
    n_timesteps=config.n_timesteps
)

# Train model (see experiments/run_experiment.py for full training loop)
```

### Running Experiments

```bash
# Run with default configuration
python experiments/run_experiment.py

# Run with custom parameters
python experiments/run_experiment.py \
    --embed_dim 128 \
    --learning_rate 0.001 \
    --n_timesteps 100 \
    --dataset_name "amazon-book"
```

## Project Structure

```
recdiff/
├── src/
│   ├── core/                   # Core algorithm implementations
│   │   ├── gcn.py             # Graph Convolutional Network encoder
│   │   ├── diffusion.py       # Forward/reverse diffusion processes
│   │   ├── denoiser.py        # Denoising neural network
│   │   └── fusion.py          # Embedding fusion module
│   ├── models/                # Model wrapper classes
│   │   └── recdiff.py         # Main RecDiff model
│   ├── utils/                 # Utility functions
│   │   ├── data.py            # Data loading and preprocessing
│   │   ├── loss.py            # Loss function implementations
│   │   ├── metrics.py         # Evaluation metrics
│   │   ├── predictor.py       # Prediction utilities
│   │   └── sched.py           # Diffusion scheduling
│   └── configs/               # Configuration files
│       └── default.yaml       # Default hyperparameters
├── tests/                     # Unit and integration tests
├── docs/                      # Documentation
├── experiments/               # Experiment scripts and notebooks
└── requirements.txt           # Python dependencies
```

## Algorithm Overview

RecDiff combines graph convolutional networks with diffusion models:

1. **Graph Encoding**: Separate GCN encoders process user-item interactions and social networks
2. **Forward Diffusion**: Gaussian noise is progressively added to social embeddings
3. **Reverse Denoising**: A neural network learns to remove noise step-by-step
4. **Embedding Fusion**: Clean collaborative and denoised social embeddings are combined
5. **Prediction**: Fused embeddings generate item recommendations

## Performance

RecDiff achieves state-of-the-art performance on benchmark datasets:

| Dataset | NDCG@10 | Recall@10 | NDCG@20 | Recall@20 |
|---------|---------|-----------|---------|-----------|
| Ciao    | 0.0842  | 0.1367    | 0.1028  | 0.1891    |
| Epinions| 0.0751  | 0.1243    | 0.0923  | 0.1734    |
| Yelp    | 0.0693  | 0.1156    | 0.0847  | 0.1625    |

## Citation

If you use RecDiff in your research, please cite:

```bibtex
@article{recdiff2024,
  title={RecDiff: Social Recommendation with Diffusion-based Graph Denoising},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## Support

For questions and support, please:
1. Check the [documentation](docs/)
2. Review [frequently asked questions](docs/FAQ.md)
3. Open an issue on GitHub