# %%
from hydra import initialize, compose
from datetime import datetime
from hydra.utils import instantiate
from model import MLP_ELU_convex
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from sklearn.cluster import KMeans
from torch import Tensor
from tqdm import tqdm
from typing import Tuple
from utils.toy_dataset import GaussianMixture
from utils.toy_dataset import GaussianMixture
import einops
import hydra
import ipdb
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import torch.nn as nn
sys.path.append("../")


class h_diag_RBF(nn.Module):
    def __init__(self, n_centers, data_size=2, data_to_fit_ambiant=None, data_to_fit_latent=None, kappa=1):
        super().__init__()
        self.K = n_centers
        self.data_size = data_size
        self.kappa = kappa
        self.W = torch.nn.Parameter(torch.rand(self.K, 1))
        #sigmas = np.ones((self.K, 1))
        sigmas = np.ones((self.K, data_size))
        if (data_to_fit_ambiant is not None) and (data_to_fit_latent is not None):
            data_to_fit_a = data_to_fit_ambiant.cpu().detach().numpy()
            data_to_fit_l = data_to_fit_latent.cpu().detach().numpy()
            print("fitting")
            clustering_model = KMeans(n_clusters=self.K)
            clustering_model.fit(data_to_fit_a)
            clusters = clustering_model.cluster_centers_
            self.register_buffer('C', torch.tensor(clustering_model.cluster_centers_, dtype=torch.float32))
            labels = clustering_model.labels_
            for k in range(self.K):
                points = data_to_fit_l[labels == k]
                variance = ((points - clusters[k]) ** 2).mean(axis=0)
                #print('variance', variance.shape)
                sigmas[k, :] = np.sqrt(variance)
        else:
            self.register_buffer('C', torch.zeros(self.K, self.data_size))
        lbda = torch.tensor(0.5 / (self.kappa * sigmas) ** 2, dtype=torch.float32)
        self.register_buffer('lamda', lbda)
    
    def h(self, x_t):
        if len(x_t.shape) > 2:
            x_t = x_t.reshape(x.shape[0], -1)
        dist2 = torch.cdist(x_t, self.C) ** 2
        phi_x = torch.exp(-0.5 * self.lamda[None, :, :] * dist2[:, :, None])
        h_x = phi_x.sum(dim=1)
        return h_x

class h_diag_Land(nn.Module):
    def __init__(self, reference_sample, gamma = 0.2):
        super().__init__()
        self.reference_sample = reference_sample
        self.gamma = gamma
    

    def weighting_function(self, x):
        pairwise_sq_diff = (x[:, None, :] - self.reference_sample[None, :, :]) ** 2
        pairwise_sq_dist = pairwise_sq_diff.sum(-1)
        weights = torch.exp(-pairwise_sq_dist / (2 * self.gamma**2))
        return weights
    
    def h(self, x_t):
        weights = self.weighting_function(x_t)  # Shape [B, N]
        differences = self.reference_sample[None, :, :] - x_t[:, None, :]  # Shape [B, N, D]
        squared_differences = differences**2  # Shape [B, N, D]

        # Compute the sum of weighted squared differences for each dimension
        M_dd_diag = torch.einsum("bn,bnd->bd", weights, squared_differences)
        return M_dd_diag

class ConformalRiemannianMetric(nn.Module):
    def __init__(self, h, euclidian_weight=0):
        super().__init__()
        self.h = h
        self.euclidian_weight = euclidian_weight
    def g(self, x_t):
        return self.h(x_t)
        
    def kinetic(self, x_t, x_t_dot):
        g = self.g(x_t)
        return (self.euclidian_weight + g)*(x_t_dot.pow(2).sum(dim=-1))


class DiagonalRiemannianMetric(nn.Module):
    def __init__(self, h, euclid_weight=0):
        super().__init__()
        self.h = h
        self.euclid_weight = euclid_weight
        
    def g(self, x_t):
        return self.h(x_t)
        
    def kinetic(self, x_t, x_t_dot):
        g = self.g(x_t)
        return torch.einsum('bi,bi->b', x_t_dot, (self.euclid_weight + g) * x_t_dot)

class Method2Metric(nn.Module):
    def __init__(self,
            ebm: nn.Module,
            a_num: float = 1.0,
            b_num: float = 1.0,
            eps: float = 1e-6,
            eta: float = 0.01,
            mu: float = 1.0,
            euclidian_weight = 0,
            alpha_fn_choice: str = 'linear'):

        super().__init__()
        self.ebm: nn.Module = ebm
        self.alpha_fn: callable = None
        self.euclidian_weight = euclidian_weight

        # different ways of obtaining alpha
        alpha_fn_1 = lambda E_th: a_num + b_num * E_th
        alpha_fn_2  = lambda E_th: 1/(a_num + b_num * E_th + eps)
        self.eta: float = eta
        self.mu: float = mu

        if alpha_fn_choice == 'linear':
            self.alpha_fn = alpha_fn_1
        elif alpha_fn_choice == 'inverse':
            self.alpha_fn = alpha_fn_2
        else:
            raise ValueError(f"Unknown alpha_fn_choice: {alpha_fn_choice}")
        
        self.last_tensors: dict = {}
    
    def get_score_n_nrg(self, x_t: Tensor) -> Tuple[Tensor, Tensor]:
        energy_out = self.ebm.forward(x_t)
        energy_out.requires_grad_(True)
        score: Tensor = torch.autograd.grad(
            outputs=energy_out,
            inputs=x_t,
            grad_outputs=torch.ones_like(energy_out),
            create_graph=True,
            retain_graph=True
            )[0]
        return  score, energy_out
    
    def kinetic(self, x_t: Tensor, x_t_dot: Tensor) -> Tensor:

        score_on_pos, energy_on_pos = self.get_score_n_nrg(x_t)  

        # print(f"{score_on_pos.shape=}")
        grad_outer_prod: Tensor = torch.einsum('bi,bj->bij', score_on_pos, score_on_pos)
        # print(f"{grad_outer_prod.shape=}")
        alpha_val: Tensor = self.alpha_fn(energy_on_pos).to(x_t.device)

        # give it a 'batch' dimension to add with grad_outer_prod
        I_mat: Tensor = einops.rearrange(torch.eye(2).to(x_t.device), 'i j -> 1 i j')
        # print(f"{I_mat.shape=}")
        # print(f"{grad_outer_prod.shape=}")
        # print(f"{alpha_val.shape=}")
        alpha_val = einops.rearrange(alpha_val, 'b 1 -> b 1 1')

        A_pre: Tensor = ( self.mu * I_mat - self.eta * grad_outer_prod)

        A_mat: Tensor =  alpha_val * A_pre

        A_mat: Tensor = A_mat + self.euclidian_weight * I_mat
        # print(f"{A_mat.shape=}")

        # A_mat_norms: Tensor = torch.linalg.norm(A_mat, dim=(1,2))
        # print(f"{A_mat_norms.shape=}")

        pre_out: Tensor = torch.einsum('bij,bj->bi', A_mat, x_t_dot)

        out: Tensor = torch.einsum('bi,bi->b', pre_out, x_t_dot)
        # print(f"{out.shape=}")
        # print(f"{pre_out.shape=}")

        self.last_tensors = {
            "score_on_pos": score_on_pos.clone(),
            "grad_outer_prod": grad_outer_prod.clone(),
            "alpha_val": alpha_val.clone(),
            "A_mat": A_mat.clone(),
        }

        return out


class Method3Metric(Method2Metric):

    def __init__(self,
            ebm: nn.Module,
            a_num: float = 1.0,
            b_num: float = 1.0,
            eps: float = 1e-6,
            euclidian_weight = 0,
            alpha_fn_choice: str = 'linear'):
        super().__init__(ebm, a_num, b_num, eps, euclidian_weight=euclidian_weight, alpha_fn_choice=alpha_fn_choice)
    