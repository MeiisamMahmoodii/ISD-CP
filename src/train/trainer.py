import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

class Trainer:
    """
    Handles the training and validation loop for the CausalTransformer.
    
    This class manages:
    - Iterating over the dataset.
    - Handling the nested batch structure (files -> chunks -> mini-batches).
    - Computing gradients and updating weights.
    - Logging progress.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: optim.Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        scheduler = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = nn.MSELoss()
        
    def train_epoch(self):
        """
        Runs one full epoch of training.
        
        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Handle nested list structure from DataLoader
            # The Dataset returns a LIST of dicts (one chunk of data).
            # DataLoader with batch_size=1 wraps this in another list.
            file_content = batch
            if isinstance(batch, list) and len(batch) == 1 and isinstance(batch[0], list):
                 file_content = batch[0]
            
            # Debug prints (optional, commented out)
            # print(f"Batch type: {type(batch)}")
            # print(f"File content type: {type(file_content)}")
            
            # Iterate over the mini-batches contained in the loaded file/chunk
            for mini_batch in file_content:
                # DataLoader collate adds a batch dim (size 1) because we have batch_size=1
                # and the "sample" is a list of dicts.
                # So each tensor in the dict gets an extra dim.
                
                x = mini_batch['x'].to(self.device)
                mask = mini_batch['mask'].to(self.device)
                value = mini_batch['value'].to(self.device)
                target = mini_batch['target'].to(self.device)
                
                # Squeeze the extra batch dimension added by DataLoader
                if x.ndim == 3 and x.shape[0] == 1:
                    x = x.squeeze(0)
                    mask = mask.squeeze(0)
                    value = value.squeeze(0)
                    target = target.squeeze(0)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                pred = self.model(x, mask, value)
                
                # Compute Loss (MSE)
                loss = self.criterion(pred, target)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        if self.scheduler:
            self.scheduler.step()
            
        return total_loss / len(self.train_loader)

    def validate(self):
        """
        Runs validation on the validation set.
        
        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                file_content = batch
                if isinstance(batch, list) and len(batch) == 1 and isinstance(batch[0], list):
                     file_content = batch[0]
                
                for mini_batch in file_content:
                    x = mini_batch['x'].to(self.device)
                    mask = mini_batch['mask'].to(self.device)
                    value = mini_batch['value'].to(self.device)
                    target = mini_batch['target'].to(self.device)
                    
                    if x.ndim == 3 and x.shape[0] == 1:
                        x = x.squeeze(0)
                        mask = mask.squeeze(0)
                        value = value.squeeze(0)
                        target = target.squeeze(0)
                    
                    pred = self.model(x, mask, value)
                    loss = self.criterion(pred, target)
                    total_loss += loss.item()
                    
        return total_loss / len(self.val_loader)
