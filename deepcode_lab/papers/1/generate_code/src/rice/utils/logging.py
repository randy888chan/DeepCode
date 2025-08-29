"""
Logging utilities for the RICE system, providing training progress tracking and visualization.
"""

import os
import json
import time
from typing import Dict, Any, Optional, List
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

class Logger:
    """Logger class for tracking training metrics and visualizing results."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to store logs
            experiment_name: Name of the current experiment
        """
        self.log_dir = os.path.join(log_dir, experiment_name)
        self.experiment_name = experiment_name
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Initialize metric storage
        self.metrics = defaultdict(list)
        self.episode_metrics = {}
        self.eval_metrics = {}
        
        # Metadata
        self.start_time = time.time()
        self.metadata = {
            "experiment_name": experiment_name,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
            "log_dir": self.log_dir
        }
        
        # Save initial metadata
        self._save_metadata()

    def log_episode(self, episode: int, metrics: Dict[str, Any]) -> None:
        """
        Log metrics for a training episode.
        
        Args:
            episode: Current episode number
            metrics: Dictionary of metrics to log
        """
        for key, value in metrics.items():
            self.metrics[key].append(value)
            self.writer.add_scalar(f"train/{key}", value, episode)
        
        self.episode_metrics[episode] = metrics
        
        # Log to console
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Episode {episode} | {metrics_str}")

    def log_evaluation(self, episode: int, metrics: Dict[str, Any]) -> None:
        """
        Log evaluation metrics.
        
        Args:
            episode: Current episode number
            metrics: Dictionary of evaluation metrics
        """
        for key, value in metrics.items():
            self.writer.add_scalar(f"eval/{key}", value, episode)
        
        self.eval_metrics[episode] = metrics
        
        # Log to console
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Evaluation at episode {episode} | {metrics_str}")

    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """
        Log hyperparameters for the experiment.
        
        Args:
            hparams: Dictionary of hyperparameters
        """
        self.metadata["hyperparameters"] = hparams
        self.writer.add_hparams(hparams, {})
        self._save_metadata()

    def log_model_summary(self, model: torch.nn.Module) -> None:
        """
        Log model architecture summary.
        
        Args:
            model: PyTorch model to summarize
        """
        self.metadata["model_summary"] = str(model)
        self._save_metadata()

    def plot_metrics(self, save: bool = True) -> None:
        """
        Plot training metrics.
        
        Args:
            save: Whether to save the plots to disk
        """
        for metric_name, values in self.metrics.items():
            plt.figure(figsize=(10, 6))
            plt.plot(values)
            plt.title(f"{metric_name} over Episodes")
            plt.xlabel("Episode")
            plt.ylabel(metric_name)
            plt.grid(True)
            
            if save:
                plt.savefig(os.path.join(self.log_dir, f"{metric_name}.png"))
                plt.close()
            else:
                plt.show()

    def save_results(self) -> None:
        """Save all metrics and results to disk."""
        results = {
            "metadata": self.metadata,
            "metrics": dict(self.metrics),
            "episode_metrics": self.episode_metrics,
            "eval_metrics": self.eval_metrics,
            "total_time": time.time() - self.start_time
        }
        
        # Save to JSON
        results_path = os.path.join(self.log_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Generate and save plots
        self.plot_metrics(save=True)
        
        print(f"Results saved to {self.log_dir}")

    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        metadata_path = os.path.join(self.log_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def close(self) -> None:
        """Close the logger and cleanup."""
        self.writer.close()
        self.save_results()

    def get_summary_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all tracked metrics.
        
        Returns:
            Dictionary of metric names to their summary statistics
        """
        stats = {}
        for metric_name, values in self.metrics.items():
            values_array = np.array(values)
            stats[metric_name] = {
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "median": float(np.median(values_array))
            }
        return stats