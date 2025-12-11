import torch

class Masker:
    def __init__(self, model, density=0.1):
        self.model = model
        self.density = density
        self.masks = {}
        self.init_masks()

    def init_masks(self):
        """Initializes a random (Erdos-Renyi style or Uniform) mask."""
        print(f"Initializing masks with density {self.density}...")
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() > 1:  # Skip 1D biases and BatchNorm
                mask_tensor = (torch.rand_like(param) < self.density).float()
                safe_name = name.replace('.', '_') + '_mask'  # Replace '.' with '_' to make it a valid single variable name
                self.model.register_buffer(safe_name, mask_tensor)
                self.masks[name] = mask_tensor
        print("Masks initialized.")

    def apply_mask_to_weights(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.masks:
                    param.data *= self.masks[name]

    def apply_mask_to_gradients(self):
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.grad *= self.masks[name]
    
    def get_sparsity_stats(self):
        total_params = 0
        active_params = 0
        for name, mask in self.masks.items():
            total_params += mask.numel()
            active_params += mask.sum().item()
        return active_params, total_params