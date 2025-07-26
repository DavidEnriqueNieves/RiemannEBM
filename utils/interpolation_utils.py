import torch
import torch.nn as nn

def slerp_coeff(t, z0, zT, lt_size=(4, 16, 16), dot_threshold=0.9995):
    #if len(z0.size()) != 5:
    #    print('Carreful about the z size in the slerp function')
    t_size = t.size()
    z0 = z0.flatten(start_dim=-len(lt_size))
    zT = zT.flatten(start_dim=-len(lt_size))
    s0 = (1-t).clone().flatten()
    s1 = t.clone().flatten()
    dot = torch.sum((z0 * zT) / (z0.norm(dim=-1, keepdim=True).detach() * zT.norm(dim=-1, keepdim=True).detach()), dim=-1).flatten()
    #dot = torch.nn.functional.cosine_similarity(z0, zT, dim=-1).flatten()
    # print(dot.size())
    filter_dot = dot.abs() < dot_threshold
    if filter_dot.sum() > 0:
        theta_0 = torch.arccos(dot[filter_dot])
        sin_theta_0 = torch.sin(theta_0)
        # print(theta_0.size())
        # stop
        theta_t = theta_0 * t.flatten()[filter_dot]
        sin_theta_t = torch.sin(theta_t)
        s0[filter_dot] = torch.sin(theta_0 - theta_t) / (sin_theta_0)
        s1[filter_dot] = sin_theta_t / (sin_theta_0)
    return s0.view(t_size), s1.view(t_size)
"""
def sample_t(steps, sampling_type='linspace', size=(2,), device='cpu'):
    if sampling_type == 'linspace':
        t = torch.linspace(0, 1, steps)
        dt = torch.tensor(1.0 / (steps - 1))
    if sampling_type == 'uniform':
        t = torch.randperm(5000+1)/5000
        t = t[:steps]
        t, _ = torch.sort(t)
        t[0], t[-1] = 0, 1
        dt = t[1:] - t[:-1]
        if torch.any(dt==0):
            print(t)
        dt.unsqueeze(-1)
    t = t.unsqueeze(-1).repeat(1, *size)
    return t.to(device), dt.to(device)
"""

def sample_t(batch_size, steps, sampling_type='linespace', size = (2,)):
    if sampling_type == "linspace":
        t = torch.linspace(0, 1, steps)
    elif sampling_type == 'uniform':
        t = torch.randperm(5000 + 1) / 5000
        t = t[:steps]
        t, _ = torch.sort(t)
        t[0], t[-1] = 0, 1
    else:
        raise NotImplementedError()
    t = t.reshape(1, steps, *([1]*len(size)))
    t = t.repeat(batch_size, 1, *([1]*len(size)))
    dt = t[:, 1:] - t[:, :-1]
    #dt = dt.reshape(1, steps-1, *([1]*len(size)))
    #dt = dt.repeat(batch_size, 1, *([1] * len(size)))
    return t, dt



class ZeroNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x0, xT=None, t=None):
        return torch.zeros_like(x0)


def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.

    Parameters
    ----------
    x : Tensor, shape (bs, *dim)
        represents the source minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    t : Tensor, shape (bs, number of x dimensions)

    Example
    -------
    x: Tensor (bs, C, W, H)
    t: Vector (bs)
    pad_t_like_x(t, x): Tensor (bs, 1, 1, 1)
    """
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


if __name__ == "__main__":
    t = sample_t_2(128, 50, size=(4, 16, 16))