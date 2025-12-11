import torch
import math
import random

class LLLRandom:
    def __init__(self, masker, model, total_steps, min_fan_in=1, max_fan_in=None, min_fan_out=1, max_fan_out=None):
        self.masker = masker
        self.model = model
        self.total_steps = total_steps
        
        self.min_fan_in = min_fan_in
        self.max_fan_in = max_fan_in
        self.min_fan_out = min_fan_out
        self.max_fan_out = max_fan_out

    def check_bad_events(self, name):
        bad_events = []
        mask = self.masker.masks[name]

        # Mask Shape: [Out_Features, In_Features]
        if mask.dim() == 2:
            fan_in = mask.sum(dim=1)
            if self.min_fan_in is not None:
                violators = (fan_in < self.min_fan_in).nonzero()
                if violators.numel() > 0:
                    bad_events.append({'type': 'neuron_in', 'layer': name, 'indices': violators})

            if self.max_fan_in is not None:
                violators = (fan_in > self.max_fan_in).nonzero()
                if violators.numel() > 0:
                    bad_events.append({'type': 'neuron_in', 'layer': name, 'indices': violators})

            fan_out = mask.sum(dim=0)
            if self.min_fan_out is not None:
                violators = (fan_out < self.min_fan_out).nonzero()
                if violators.numel() > 0:
                    bad_events.append({'type': 'neuron_out', 'layer': name, 'indices': violators})

            if self.max_fan_out is not None:
                violators = (fan_out > self.max_fan_out).nonzero()
                if violators.numel() > 0:
                    bad_events.append({'type': 'neuron_out', 'layer': name, 'indices': violators})
                        
        return bad_events

    def resample(self):
        resampled_count = 0    
        for i in range(1000):
            healthy = True
            for name, mask in self.masker.masks.items():
                bad_events = self.check_bad_events(name)
                if bad_events:
                    healthy = False
                    event = random.choice(bad_events)
                    indices = event['indices']
                    target_density = self.masker.density
                    
                    for idx in indices:
                        idx = idx.item()
                        resampled_count += 1
                        
                        if event['type'] == 'neuron_in':
                            row_len = mask.shape[1]
                            new_row = (torch.rand(row_len, device=mask.device) < target_density).float()
                            mask[idx, :] = new_row

                        elif event['type'] == 'neuron_out':
                            col_len = mask.shape[0]
                            new_col = (torch.rand(col_len, device=mask.device) < target_density).float()
                            mask[:, idx] = new_col

            if healthy:
                break

        if resampled_count > 0:
            self.masker.apply_mask_to_weights()

        return resampled_count

    def step(self, current_step):
        start_fraction = 0.3 
        current_fraction = start_fraction * 0.5 * (1 + math.cos(math.pi * current_step / self.total_steps))
        
        params = dict(self.model.named_parameters())
        active_connections = {}

        for name, mask in self.masker.masks.items():
            param = params[name]
            grad = param.grad
            
            if grad is None:
                continue

            active_connections[name] = mask.sum().item()
            num_to_swap = int(active_connections[name] * current_fraction)
            
            if num_to_swap == 0:
                continue

            # --- 1. PRUNE (Smallest Magnitude) ---
            weight_abs = torch.abs(param.data * mask)
            weight_abs[mask == 0] = float('inf')
            _, prune_indices = torch.topk(weight_abs.view(-1), k=num_to_swap, largest=False)  # k smallest weights
            mask_flat = mask.view(-1)
            mask_flat[prune_indices] = 0.0

        # MT resampling algorithm, try for 1000 times
        self.resample()
                
        for name, mask in self.masker.masks.items():
            param = params[name]
            mask_flat = mask.view(-1)

            num_to_grow = int(active_connections[name] - mask.sum().item())
            
            if num_to_grow <= 0:
                continue
            
            # --- 2. GROW Randomly ---
            empty_indices = (mask_flat == 0).nonzero().view(-1)
            actual_grow = min(num_to_grow, empty_indices.size(0))
            perm = torch.randperm(empty_indices.size(0))
            grow_indices = empty_indices[perm[:actual_grow]]
            mask_flat[grow_indices] = 1.0
            
            with torch.no_grad():
                param.view(-1)[grow_indices] = 0.0

        return current_fraction
