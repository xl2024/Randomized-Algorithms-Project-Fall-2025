import torch
import math

class RandomSampler:
    def __init__(self, masker, model, total_steps):
        self.masker = masker
        self.model = model
        self.total_steps = total_steps
        
    def step(self, current_step):
        """
        Naive Random Resampling:
        1. Prune: Remove weights with smallest magnitude.
        2. Grow: Randomly Grow new connections (Ignorant of gradient).
        """
        start_fraction = 0.3 
        current_fraction = start_fraction * 0.5 * (1 + math.cos(math.pi * current_step / self.total_steps))
        
        params = dict(self.model.named_parameters())

        for name, mask in self.masker.masks.items():
            param = params[name]
            grad = param.grad
            
            if grad is None:
                continue

            active_connections = mask.sum().item()
            num_to_swap = int(active_connections * current_fraction)
            
            if num_to_swap == 0:
                continue

            # --- 1. PRUNE (Smallest Magnitude) ---
            weight_abs = torch.abs(param.data * mask)
            weight_abs[mask == 0] = float('inf')
            _, prune_indices = torch.topk(weight_abs.view(-1), k=num_to_swap, largest=False)  # k smallest weights
            mask_flat = mask.view(-1)
            mask_flat[prune_indices] = 0.0
            
            # --- 2. GROW Randomly ---
            empty_indices = (mask_flat == 0).nonzero().view(-1)
            actual_grow = min(num_to_swap, empty_indices.size(0))
            perm = torch.randperm(empty_indices.size(0))
            grow_indices = empty_indices[perm[:actual_grow]]
            mask_flat[grow_indices] = 1.0
            
            with torch.no_grad():
                param.view(-1)[grow_indices] = 0.0
            
        return current_fraction