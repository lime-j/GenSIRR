import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class LRWarmupCallback(Callback):
    """
    Implements learning rate warmup during the initial training epochs.
    
    Learning rate warmup helps stabilize training in the early stages by gradually
    increasing the learning rate from a small initial value to the target value
    over a specified number of epochs.
    """
    
    def __init__(self, warmup_epochs=5, initial_lr_factor=0.1, 
                 apply_to_all_params=True, verbose=False):
        """
        Args:
            warmup_epochs (int): Number of epochs for the warmup phase.
            initial_lr_factor (float): Initial learning rate factor (0.1 means starting at 10% of target lr).
            apply_to_all_params (bool): Whether to apply warmup to all parameter groups or only the first.
            verbose (bool): Whether to print learning rate updates during warmup.
        """
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lr_factor = initial_lr_factor
        self.apply_to_all_params = apply_to_all_params
        self.verbose = verbose
        self._original_lrs = None
        
    def on_fit_start(self, trainer, pl_module):
        """Store original learning rates when starting training."""
        if self.warmup_epochs <= 0:
            return
            
        # Get optimizer
        optimizer = trainer.optimizers[0]
        
        # Store original learning rates
        self._original_lrs = []
        for param_group in optimizer.param_groups:
            self._original_lrs.append(param_group['lr'])
            
        # Set initial learning rates
        self._update_lr(optimizer, trainer.current_epoch, pl_module)
        
    def on_train_epoch_start(self, trainer, pl_module):
        """Update learning rate at the start of each epoch during warmup."""
        if self.warmup_epochs <= 0 or self._original_lrs is None:
            return
            
        # Only apply during warmup period
        if trainer.current_epoch >= self.warmup_epochs:
            return
            
        # Get optimizer
        optimizer = trainer.optimizers[0]
        
        # Update learning rates
        self._update_lr(optimizer, trainer.current_epoch, pl_module)
        
    def _update_lr(self, optimizer, current_epoch, pl_module):
        """Update learning rates based on current epoch."""
        # Calculate current factor
        if self.warmup_epochs <= 1:
            factor = 1.0
        else:
            progress = min(1.0, current_epoch / (self.warmup_epochs - 1))
            factor = self.initial_lr_factor + progress * (1.0 - self.initial_lr_factor)
        
        # Apply factor to learning rates
        for i, param_group in enumerate(optimizer.param_groups):
            if i == 0 or self.apply_to_all_params:
                param_group['lr'] = self._original_lrs[i] * factor
                if self.verbose and (current_epoch == 0 or pl_module.local_rank == 0):
                    rank_zero_info(f"LR warmup: group {i}, epoch {current_epoch},"
                                 f" factor {factor:.4f}, lr {param_group['lr']:.6f}")
        
    def on_train_epoch_end(self, trainer, pl_module):
        """Restore original learning rates at the end of warmup."""
        if self.warmup_epochs <= 0 or self._original_lrs is None:
            return
            
        # Check if this is the last warmup epoch
        if trainer.current_epoch == self.warmup_epochs - 1:
            # Get optimizer
            optimizer = trainer.optimizers[0]
            
            # Restore original learning rates
            for i, param_group in enumerate(optimizer.param_groups):
                if i == 0 or self.apply_to_all_params:
                    param_group['lr'] = self._original_lrs[i]
                    if self.verbose and pl_module.local_rank == 0:
                        rank_zero_info(f"LR warmup completed: group {i}, restoring lr to"
                                     f" {param_group['lr']:.6f}")
                        
    def state_dict(self):
        """Return state dict for checkpointing."""
        return {
            "original_lrs": self._original_lrs,
            "warmup_epochs": self.warmup_epochs,
            "initial_lr_factor": self.initial_lr_factor
        }
    
    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        if "original_lrs" in state_dict:
            self._original_lrs = state_dict["original_lrs"]
        if "warmup_epochs" in state_dict:
            self.warmup_epochs = state_dict["warmup_epochs"]
        if "initial_lr_factor" in state_dict:
            self.initial_lr_factor = state_dict["initial_lr_factor"] 