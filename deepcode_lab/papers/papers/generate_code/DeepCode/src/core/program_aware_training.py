from typing import Tuple, List
import torch
import torch.nn as nn

class ProgramAwareTrainer:
    def __init__(self, model: nn.Module, lambda_param: float = 0.1):
        self.model = model
        self.lambda_param = lambda_param
        self.optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    def train_with_program_signals(self, data_loader: torch.utils.data.DataLoader):
        self.model.train()
        total_loss = 0.0
        
        for batch in data_loader:
            inputs, targets, program_properties = batch
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self._calculate_code_loss(outputs, targets)
            program_loss = self._calculate_program_loss(outputs, program_properties)
            total_loss = loss + self.lambda_param * program_loss
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            total_loss += total_loss.item()
        
        return total_loss / len(data_loader)

    def _calculate_code_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Implement code loss calculation (e.g., cross-entropy)
        return nn.functional.cross_entropy(outputs, targets)

    def _calculate_program_loss(self, outputs: torch.Tensor, program_properties: torch.Tensor) -> torch.Tensor:
        # Implement program property loss calculation
        # This would depend on the specific program properties being predicted
        return nn.functional.mse_loss(outputs, program_properties)