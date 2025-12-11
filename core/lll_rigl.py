import torch
import math

class LLLResampler:
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

    def resample(self, bad_events):
        if not bad_events:
            return 0

        resampled_count = 0

        for event in bad_events:
            layer_name = event['layer']
            indices = event['indices']
            mask = self.masker.masks[layer_name]
            target_density = self.masker.density
            
            for idx in indices:
                idx = idx.item()
                resampled_count += 1
                
                if event['type'] == 'neuron_in':
                    # Target: mask[idx, :]
                    row_len = mask.shape[1]
                    
                    new_row = (torch.rand(row_len, device=mask.device) < target_density).float()
                    
                    # if self.min_fan_in >= 1 and new_row.sum() == 0:
                    #     new_row[torch.randint(0, row_len, (1,))] = 1.0

                    mask[idx, :] = new_row

                elif event['type'] == 'neuron_out':
                    # Target: mask[:, idx]
                    col_len = mask.shape[0]
                    
                    new_col = (torch.rand(col_len, device=mask.device) < target_density).float()
                    
                    # if self.min_fan_out >= 1 and new_col.sum() == 0:
                    #     new_col[torch.randint(0, col_len, (1,))] = 1.0

                    mask[:, idx] = new_col

        return resampled_count

    def step(self, current_step):
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

            # MT resampling algorithm, try for 1000 times
            for i in range(1000):
                bad_events = self.check_bad_events(name)
                if bad_events:
                    self.resample(bad_events)
                    self.masker.apply_mask_to_weights()
                else:
                    break
            
            num_to_grow = int(active_connections - mask.sum().item())
            
            if num_to_grow <= 0:
                continue
            
            # --- 2. GROW (Largest Gradient) ---
            grad_abs = torch.abs(grad)
            grad_abs[mask == 1] = -float('inf')
            _, grow_indices = torch.topk(grad_abs.view(-1), k=num_to_grow, largest=True)  # k largest gradients
            mask_flat[grow_indices] = 1.0
            with torch.no_grad():
                param.view(-1)[grow_indices] = 0.0   # New weights start at 0 (Standard RigL)

        return current_fraction
