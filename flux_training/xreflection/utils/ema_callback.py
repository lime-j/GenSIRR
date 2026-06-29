import copy
import lightning as L
from lightning.pytorch.utilities import rank_zero_only
import torch
from xreflection import build_network
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn

class EMACallback(L.Callback):
    """
    Exponential Moving Average (EMA) callback for Lightning.
    
    Maintains an exponential moving average of model parameters during training,
    and optionally uses these averaged parameters during validation and testing.
    
    Args:
        decay (float): The decay rate for EMA. Higher values (closer to 1.0)
                      give more weight to past parameters. Default: 0.9999
        update_interval (int): Frequency of updates in iterations. Default: 1
        use_for_val (bool): Whether to use EMA model for validation. Default: True
        use_for_test (bool): Whether to use EMA model for testing. Default: True
    """
    
    def __init__(self, decay=0.9999, update_interval=1, use_for_val=True, use_for_test=True):
        super().__init__()
        self.decay = decay
        self.update_interval = update_interval
        self.use_for_val = use_for_val
        self.use_for_test = use_for_test
        
        # EMA model and state tracking
        self.model_ema = None
        self.collected_params = None
        self.is_swapped = False
        self.ema_state_dict = None
        
        # Training step counter
        self.step_counter = 0
    
    def on_fit_start(self, trainer, pl_module):
        """Initialize EMA model at the start of training."""
        # Create a copy of the model
        self.model_ema = copy.deepcopy(pl_module.net_g)
        
        # Move to the same device as the original model
        device = next(pl_module.net_g.parameters()).device
        self.model_ema.to(device)
        
        # Set EMA model to evaluation mode
        self.model_ema.eval()
        
        # Don't track gradients for EMA model
        for param in self.model_ema.parameters():
            param.requires_grad_(False)
        
        # Restore EMA state if available from checkpoint
        if hasattr(self, 'ema_state_dict') and self.ema_state_dict is not None:
            self.model_ema.load_state_dict(self.ema_state_dict)
            self.ema_state_dict = None  # Clean up after loading
        
        # Notify module about EMA usage
        pl_module.use_ema = True
        
        # Print EMA configuration
        rank_zero_info(f"EMA initialized with decay: {self.decay}, update interval: {self.update_interval}")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update EMA model after each training batch."""
        # Update counter
        self.step_counter += 1
        
        # Skip if not at update interval
        if self.step_counter % self.update_interval != 0:
            return
        
        # Update EMA model parameters
        self.update_parameters(pl_module.net_g)
    
    def update_parameters(self, model):
        """Update EMA model parameters based on current model."""
        with torch.no_grad():
            for param_ema, param_model in zip(self.model_ema.parameters(), model.parameters()):
                # Apply exponential moving average update
                param_ema.data.mul_(self.decay).add_(param_model.data, alpha=1 - self.decay)
            
            # Handle batch normalization statistics
            for buf_ema, buf_model in zip(self.model_ema.buffers(), model.buffers()):
                buf_ema.data.copy_(buf_model.data)
    
    def on_validation_start(self, trainer, pl_module):
        """Swap original model with EMA model before validation if enabled."""
        if not self.use_for_val:
            return
            
        # Save original model and swap with EMA model
        self.swap_model(pl_module)
    
    def on_validation_end(self, trainer, pl_module):
        """Restore original model after validation if swapped."""
        if not self.use_for_val:
            return
            
        # Restore original model
        self.restore_model(pl_module)
    
    def on_test_start(self, trainer, pl_module):
        """Swap original model with EMA model before testing if enabled."""
        if not self.use_for_test:
            return
            
        # Save original model and swap with EMA model
        self.swap_model(pl_module)
    
    def on_test_end(self, trainer, pl_module):
        """Restore original model after testing if swapped."""
        if not self.use_for_test:
            return
            
        # Restore original model
        self.restore_model(pl_module)
    
    def swap_model(self, pl_module):
        """Swap original model with EMA model."""
        if self.is_swapped:
            rank_zero_warn("EMA model already swapped, skipping")
            return
            
        # Save reference to original model and replace with EMA
        self.collected_params = copy.deepcopy(pl_module.net_g.state_dict())
        pl_module.net_g.load_state_dict(self.model_ema.state_dict())
        self.is_swapped = True
        
        rank_zero_info("Swapped original model with EMA model for evaluation")
    
    def restore_model(self, pl_module):
        """Restore original model."""
        if not self.is_swapped:
            rank_zero_warn("No original model to restore, skipping")
            return
            
        # Restore original model
        pl_module.net_g.load_state_dict(self.collected_params)
        self.is_swapped = False
        
        rank_zero_info("Restored original model after evaluation")
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Save EMA model state in checkpoint."""
        if self.model_ema is not None:
            checkpoint['model_ema'] = self.model_ema.state_dict()
            checkpoint['ema_decay'] = self.decay
            checkpoint['ema_update_interval'] = self.update_interval
    
    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """Load EMA model state from checkpoint."""
        if 'model_ema' in checkpoint:
            # Store the state dict to be applied in on_fit_start
            self.ema_state_dict = checkpoint['model_ema']
            
            # Restore EMA settings if saved
            if 'ema_decay' in checkpoint:
                self.decay = checkpoint['ema_decay']
            if 'ema_update_interval' in checkpoint:
                self.update_interval = checkpoint['ema_update_interval'] 