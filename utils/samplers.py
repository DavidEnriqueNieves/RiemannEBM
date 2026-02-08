import torch
from torch import Tensor    

class SgldSampler:
    def __init__(self, step_size: float, noise_scale: float, num_steps: int):
        self.step_size = step_size
        self.noise_scale = noise_scale
        self.num_steps = num_steps


    def sample(self, model: torch.nn.Module, x_i: Tensor):
        # TODO: put an assert here for the shape?
        x_s = x_i.clone()
        #x_s = 5*torch.randn_like(x) + torch.tensor([0, 5]).unsqueeze(0).to(x.device)
        x_s.requires_grad_(True)
        for i in range(self.num_steps):
            e = model.forward(x_s)
            grad_x = torch.autograd.grad(
                outputs=e, 
                inputs=x_s,
                grad_outputs=torch.ones_like(e),  # same shape as f
                create_graph=True,
                retain_graph=True
                )[0]
            
            x_s.data = x_s.data - self.step_size * grad_x + self.noise_scale * torch.randn_like(x_s)
        final_samples = x_s.detach()
        return final_samples
