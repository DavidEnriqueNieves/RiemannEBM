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
from utils.h_utils import linear_normalization
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

class RiemannianMetric(nn.Module):
    def __init__(self, latex_name=""):
        super().__init__()
        self.latex_name = latex_name
    
    def get_name_latex(self):
        if self.latex_name:
            return self.latex_name
        else:
            return self.__class__.__name__
    
    def g(self, x_t):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def kinetic(self, x_t, x_t_dot):
        raise NotImplementedError("Subclasses should implement this method.")

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

class ConformalRiemannianMetric(RiemannianMetric):
    def __init__(self, h, euclidian_weight=0):
        super().__init__()
        self.h = h
        self.euclidian_weight = euclidian_weight
    def g(self, x_t):
        return self.h(x_t)
        
    def kinetic(self, x_t, x_t_dot):
        g = self.g(x_t)
        return (self.euclidian_weight + g)*(x_t_dot.pow(2).sum(dim=-1))


class DiagonalRiemannianMetric(RiemannianMetric):
    def __init__(self, h, euclid_weight=0):
        super().__init__()
        self.h = h
        self.euclid_weight = euclid_weight
        
    def g(self, x_t):
        return self.h(x_t)
        
    def kinetic(self, x_t, x_t_dot):
        g = self.g(x_t)
        return torch.einsum('bi,bi->b', x_t_dot, (self.euclid_weight + g) * x_t_dot)

class Method2Metric(RiemannianMetric):
    def __init__(self,
            ebm: nn.Module,
            eps: float = 1e-6,
            eta: float = 0.01,
            a_num: float = None,
            b_num: float = None,
            make_alpha_positive: bool = True,
            mu: float = 1.0,
            gamma: float = 1.0,
            alpha_a: float = 1.0,
            alpha_b: float = 0.0,

            alpha_ub: float = 2.0,
            alpha_lb: float = 1.0,

            euclidian_weight = 0,
            alpha_fn_choice: str = 'linear'):

        super().__init__()
        self.ebm: nn.Module = ebm
        self.alpha_fn: callable = None
        self.euclidian_weight = euclidian_weight
        self.eps: float = eps

        if a_num is None:
            a_num = 1.0
        
        if b_num is None:
            b_num = 0.0
        
        self.a_num: float = a_num
        self.b_num: float = b_num

        self._alpha_a: float = alpha_a
        self._alpha_b: float = alpha_b
        
        # sets the lower bound for alpha after it has been normalized over the
        # entire dataset 
        self._alpha_ub: float = alpha_ub
        self._alpha_lb: float = alpha_lb

        # different ways of obtaining alpha
        # the lower bound below is there to make sure that after "normalizing",
        # we don't get eigenvalues that are too close to zero, which would 
        alpha_fn_1 = lambda E_th: self._alpha_a * (E_th - self._alpha_b) + self._alpha_lb
        alpha_fn_2  = lambda E_th: 1/(self._alpha_a * torch.exp(-E_th) + self._alpha_b + self.eps)
        self.eta: float = eta
        self.mu: float = mu
        self.gamma: float = gamma

        if alpha_fn_choice == 'linear':
            self.alpha_fn = alpha_fn_1
        elif alpha_fn_choice == 'inverse':
            self.alpha_fn = alpha_fn_2
        else:
            raise ValueError(f"Unknown alpha_fn_choice: {alpha_fn_choice}")
        
        self.last_tensors: dict = {}

        # cached values for pre-checking the metric
        # usually calculated on the ENTIRE dataset
        self.A_tot_scaled: Tensor = None
        self.A_tot_scaled_inv: Tensor = None
        self.score_total: Tensor = None
        self.energy_total: Tensor = None

    
    def g(self, pos: Tensor, check: bool = False) -> Tensor:
        score_on_pos, energy_on_pos = self.get_score_n_nrg(pos)  

        # print(f"{score_on_pos.shape=}")
        grad_outer_prod: Tensor = torch.einsum('bi,bj->bij', score_on_pos, score_on_pos)
        # print(f"{grad_outer_prod.shape=}")
        alpha_val: Tensor = self.alpha_fn(energy_on_pos).to(pos.device)
        alpha_val = einops.rearrange(alpha_val, 'b 1 -> b 1 1')

        if check:
            self._check_dyad_eigs(energy_on_pos, grad_outer_prod)
        
        if check:
            # sets all the values for normalizing alpha
            self._set_alpha_norm(energy_on_pos)

            # recalculate alpha_val with the new normalization parameters
            alpha_val: Tensor = self.alpha_fn(energy_on_pos).to(pos.device)
            alpha_val = einops.rearrange(alpha_val, 'b 1 -> b 1 1')

        # give it a 'batch' dimension to add with grad_outer_prod
        I_mat: Tensor = einops.rearrange(torch.eye(2).to(pos.device), 'i j -> 1 i j')

        A_pre: Tensor = ( self.mu * I_mat - self.eta * grad_outer_prod)
        A_mat: Tensor =  alpha_val * A_pre 

        A_mat: Tensor = A_mat 

        if check:
            self._check_nmd_Atot(A_mat)

        return A_mat
    
    def _check_nmd_Atot(self, A_tot_scld):
        # print(f"{A_mat.shape=}")
        print(f"{A_tot_scld.shape=}")
        eigvals, eigvecs = torch.linalg.eig(A_tot_scld)
        print(f"{eigvecs.shape=}")

        assert torch.all(eigvecs.imag.abs() < 1e-6), "Complex eigenvalues found in A_tot_scld"
        assert torch.all(eigvals.real >= 0), f"Non-positive eigenvalues found in A_tot_scld with min {eigvals.real.min().item()}. {eigvals=}"

        print(f"Min eigenvalue of A_tot_scld: {eigvals.real.min().item()}")
        print(f"Max eigenvalue of A_tot_scld: {eigvals.real.max().item()}")

        A_inv: Tensor = torch.linalg.inv(A_tot_scld)

        eigvals_inv, eigvecs_inv = torch.linalg.eig(A_inv)
        assert torch.all(eigvecs_inv.imag.abs() < 1e-6), "Complex eigenvalues found in A_inv"
        assert torch.all(eigvals_inv.real >= 0), f"Non-positive eigenvalues found in A_inv with min {eigvals_inv.real.min().item()}. {eigvals_inv=}"


    def _set_alpha_norm(self, energy_on_pos: Tensor):
        """
        Used to make sure that alpha is entirely positive over the range of
        energies in the dataset, and that it does not lead to eigenvalues that
        are too close to zero, which would lead to instability. Should be called
        on the entire dataset before training.
        """
        min_en: float = energy_on_pos.real.min().item()
        print(f"{min_en=}")
        max_en: float = energy_on_pos.real.max().item()
        print(f"{max_en=}")
        mult_factor: float = 10.0
        alpha_a: float = mult_factor * (1 / (max_en - min_en + self.eps))
        alpha_b: float = min_en

        alpha_lb: float = 5.0

        self.alpha_a = alpha_a
        self.alpha_b = alpha_b
        self.alpha_lb = alpha_lb

    
    def _check_A_tot_inv(self, A_tot_scld) -> Tensor:
                # print(f"{A_mat_scld.shape=}")
        print(f"{A_tot_scld.shape=}")
        eigvals, eigvecs = torch.linalg.eig(A_tot_scld)
        print(f"{eigvecs.shape=}")

        assert torch.all(eigvecs.imag.abs() < 1e-6), "Complex eigenvalues found in A_tot_scld"
        assert torch.all(eigvals.real >= 0), f"Non-positive eigenvalues found in A_tot_scld with min {eigvals.real.min().item()}. {eigvals=}"

        print(f"Min eigenvalue of A_tot_scld: {eigvals.real.min().item()}")
        print(f"Max eigenvalue of A_tot_scld: {eigvals.real.max().item()}")

        A_tot_scld_inv: Tensor = torch.linalg.inv(A_tot_scld)

        eigvals_inv, eigvecs_inv = torch.linalg.eig(A_tot_scld_inv)
        assert torch.all(eigvecs_inv.imag.abs() < 1e-6), "Complex eigenvalues found in A_tot_scld_inv"
        assert torch.all(eigvals_inv.real >= 0), f"Non-positive eigenvalues found in A_tot_scld_inv with min {eigvals_inv.real.min().item()}. {eigvals_inv=}"
        return A_tot_scld_inv


    def _check_dyad_eigs(self, met_score, grad_dyad_tot):
        eigvals, eigvecs = torch.linalg.eig(grad_dyad_tot)
        print(f"{eigvals.shape=}")
        print(f"{met_score.shape=}")
        print(f"{torch.norm(met_score, dim=1, p=2)[:10]**2=}")
        print(f"{eigvals[:10]=}")
        assert torch.all(eigvals.imag.abs() < 1e-6), "Complex eigenvalues found in grad_outer_prod"
        assert torch.all(eigvals.real >= -1.0e-5), f"Non-positive eigenvalues found in grad_outer_prod with min {eigvals.real.min().item()}. {eigvals=}"


    @property
    def alpha_lb(self) -> float:
        return self._alpha_lb
    
    @alpha_lb.setter
    def alpha_lb(self, value: float):
        self._alpha_lb = value
    
    @property
    def alpha_ub(self) -> float:
        return self._alpha_ub
    
    @alpha_ub.setter
    def alpha_ub(self, value: float):
        self._alpha_ub = value
    
    @property
    def alpha_a(self) -> float:
        return self._alpha_a

    @alpha_a.setter
    def alpha_a(self, value: float):
        self._alpha_a = value
    
    @property
    def alpha_b(self) -> float:
        return self._alpha_b

    @alpha_b.setter
    def alpha_b(self, value: float):
        self._alpha_b = value
    
    @property
    def a(self) -> float:
        return self.a_num
    @property
    def b(self) -> float:
        return self.b_num

    @a.setter
    def a(self, value: float):
        assert value > 0, "a should be positive for stability reasons."
        self.a_num = value
    
    @b.setter
    def b(self, value: float):
        assert value >= 0, "b should be non-negative for stability reasons."
        self.b_num = value
    
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

    def kinetic(self, x_t: Tensor, x_t_dot: Tensor, a_num: float = None, b_num: float = None, check=False) -> Tensor:
        # WARNING! a_num and b_num should probably NOT be used

        I_mat: Tensor = einops.rearrange(torch.eye(2).to(x_t.device), 'i j -> 1 i j')

        if a_num is None:
            a_num = self.a_num
        if b_num is None:
            b_num = self.b_num

        A_mat: Tensor = a_num * self.g(x_t, check=check) + b_num * I_mat

        if check:
            Amat_max, A_mat_min = A_mat.max().item(), A_mat.min().item()
            a_num, b_num = linear_normalization(Amat_max, A_mat_min, 1e3, 0)
            print(f"Normalizing A_mat with a_num={a_num} and b_num={b_num} to ensure that its values are in a reasonable range for stability")
            self.a_num: float = a_num
            self.b_num: float = b_num

        A_mat: Tensor = a_num * self.g(x_t, check=check) + b_num * I_mat
        pre_inprods: Tensor = torch.einsum('bij,bj->bi', A_mat, x_t_dot)
        inner_prods: Tensor = torch.einsum('bi,bi->b', pre_inprods, x_t_dot)
        # print(f"{out.shape=}")
        # print(f"{pre_out.shape=}")

        return inner_prods


class Method3Metric(Method2Metric):

    def __init__(self,
            ebm: nn.Module,
            eps: float = 1e-6,
            eta: float = 0.01,
            a_num: float = None,
            b_num: float = None,
            make_alpha_positive: bool = True,
            mu: float = 1.0,
            gamma: float = 1.0,
            alpha_a: float = 1.0,
            alpha_b: float = 0.0,

            alpha_ub: float = 2.0,
            alpha_lb: float = 1.0,
            beta: float = 0.5,

            euclidian_weight = 0,
            alpha_fn_choice: str = 'linear'):

        super().__init__(
            ebm=ebm,
            eps=eps,
            eta=eta,
            a_num=a_num,
            b_num=b_num,
            make_alpha_positive=make_alpha_positive,
            mu=mu,
            gamma=gamma,
            alpha_a=alpha_a,
            alpha_b=alpha_b,
            alpha_ub=alpha_ub,
            alpha_lb=alpha_lb,
            euclidian_weight=euclidian_weight,
            alpha_fn_choice=alpha_fn_choice
        )

        self.beta: float = beta 

    def one_form(self, x_t: Tensor, A_mat: Tensor = None, check: bool = False) -> Tensor:
        score_on_pos, energy_on_pos = self.get_score_n_nrg(x_t)  

        score_inner_prods: Tensor = torch.einsum('bi,bi->b', score_on_pos, score_on_pos)
        score_inner_prods: Tensor = einops.rearrange(score_inner_prods, 'b -> b 1 1')
        
        if check:
            assert  not torch.isnan(score_inner_prods).any(), "NaN values found in score_inner_prods, which could lead to instability. Consider increasing eps or checking the energy landscape for very low values."
            assert torch.all(score_inner_prods > 0), "Negative values found in score_inner_prods, which could lead to instability. Consider increasing eps or checking the energy landscape for very low values."

        # Computing the inverse of A(x) cheaply using the Sherman-Morrison formula
        # inv_met: Tensor = (self.eta / self.mu) * I_mat - \
        #     (
        #         (grad_outer_prod) / \
        #         ((self.mu / self.eta) + inner_prods)
        #     )

        # yes, I know there is an easier way of doing this but I just want this to work 
        # yes, I know there is an easier way of doing this but I just want this to work 

        inv_met: Tensor = torch.linalg.inv(A_mat)

        I_mat: Tensor = einops.rearrange(torch.eye(2), 'i j -> 1 i j')
        I_mat = I_mat.to(x_t.device)

        if check:
            eigvecs, eigvals_inv_met = torch.linalg.eig(inv_met)

            print(f"{torch.isnan(inv_met).any()=}")
            assert not torch.isnan(inv_met).any(), "NaN values found in inv_met, which could lead to instability. Consider increasing eps or checking the energy landscape for very low values."
            assert torch.all(eigvecs.imag.abs() < 1e-6), "Complex eigenvalues found in A_inv"
            assert torch.all(eigvals_inv_met.real >= 0), f"Non-positive eigenvalues found in A_inv with min {eigvals_inv_met.real.min().item()}. {eigvals_inv_met=}"

        inv_score_inner_prods_pre: Tensor = torch.einsum('bij,bj->bi', inv_met, score_on_pos)
        inv_inner_prod: Tensor = torch.einsum('bi,bi->b', inv_score_inner_prods_pre, score_on_pos)

        if check:
            assert not torch.isnan(inv_inner_prod).any(), "NaN values found in inv_inner_prod, which could lead to instability. Consider increasing eps or checking the energy landscape for very low values."

            assert torch.all(inv_inner_prod > 0), "Negative values found in inv_inner_prod, which could lead to instability. Consider increasing eps or checking the energy landscape for very low values."

            print(f"{torch.isnan(inv_inner_prod).any()=}")
            print(f"{torch.linalg.norm(inv_inner_prod)=}")

        beta_comp: Tensor = torch.sqrt( inv_inner_prod / score_inner_prods)

        if check:
            self._check_beta(score_inner_prods, inv_inner_prod, beta_comp)

        one_ov_sq: Tensor = 1  / torch.sqrt(inv_inner_prod + self.eps)
        
        if check:
            assert not torch.any(torch.isnan(one_ov_sq)), "NaN values found in one_ov_sq, which could lead to instability. Consider increasing eps or checking the energy landscape for very low values."
            # print(f"{one_ov_sq.shape=}")

            # if beta is less than the above value, we should not run into the assert below...

            einops.parse_shape(inv_met, 'b i j')
            einops.parse_shape(score_on_pos, 'b i')
            einops.parse_shape(one_ov_sq, 'b')

        one_form: Tensor = self.beta * one_ov_sq[:, None] * score_on_pos

        if check:
            einops.parse_shape(one_form, 'b i')

        # calculating the norm of the one-form under A(x)

        pre_out: Tensor = torch.einsum('bij,bj->bi', A_mat , one_form)
        norm_one_form: Tensor = torch.einsum('bi,bi->b', pre_out, one_form)

        if check:
            assert torch.all(norm_one_form < 1.0), f"The norm of the one-form under A(x) should be less than 1 for stability reasons. Got an average of {norm_one_form.mean().item()} and a max of {norm_one_form.max().item()}"
            print(f"{norm_one_form.max().item()=}")

        return one_form

    def _check_beta(self, score_inner_prods, inv_inner_prod, beta_comp):
        print(f"beta should be less than {beta_comp.min().item()} to fulfill the condition for a Randers metric")
        if self.beta >= beta_comp.min().item():
            print("Warning! beta is greater than the minimum value required for a Randers metric, which could lead to instability. Consider reducing beta or checking the energy landscape for very low values.")

        print(f"{inv_inner_prod.max().item()=}")
        print(f"{score_inner_prods.max().item()=}")
        print(f"{beta_comp.max().item()=}")
    
    def kinetic(self, x_t: Tensor, x_t_dot: Tensor, a_num: float = None, b_num: float = None, check=False) -> Tensor:
        # WARNING! a_num and b_num should probably NOT be used

        I_mat: Tensor = einops.rearrange(torch.eye(2).to(x_t.device), 'i j -> 1 i j')

        if a_num is None:
            a_num = self.a_num
        if b_num is None:
            b_num = self.b_num

        A_mat: Tensor = a_num * self.g(x_t, check=check) + b_num * I_mat

        if check:
            Amat_max, A_mat_min = A_mat.max().item(), A_mat.min().item()
            a_num, b_num = linear_normalization(Amat_max, A_mat_min, 1e3, 0)
            print(f"Normalizing A_mat with a_num={a_num} and b_num={b_num} to ensure that its values are in a reasonable range for stability")
            self.a_num: float = a_num
            self.b_num: float = b_num

        A_mat: Tensor = a_num * self.g(x_t, check=check) + b_num * I_mat
        pre_inprods: Tensor = torch.einsum('bij,bj->bi', A_mat, x_t_dot)
        inner_prods: Tensor = torch.einsum('bi,bi->b', pre_inprods, x_t_dot)
        # print(f"{out.shape=}")
        # print(f"{pre_out.shape=}")

        # F(x, y)^2 = <y, A(x)y> + 2 * sq(<y, A(x)y>) * <one_form, y> + (<one_form, y>)^2
        one_form: Tensor = self.one_form(x_t, A_mat=A_mat)
        one_form_inner_prod: Tensor = torch.einsum('bi,bi->b', one_form, x_t_dot)

        # WARNING! Find out more about this form of normalization???
        out: Tensor = self.a_num * (inner_prods + 2 * torch.sqrt(inner_prods) * one_form_inner_prod + one_form_inner_prod.pow(2)) + self.b_num 

        return out

if __name__ == "__main__":

    import torch
    import torch
    from torch import Tensor
    import math
    import sys
    sys.path.append('../')  
    import numpy as np
    import plotly.graph_objects as go
    from dataclasses import dataclass
    import einops
    from plotly.subplots import make_subplots
    from model import MLP_ELU_convex
    import hydra
    from metrics import RiemannianMetric, DiagonalRiemannianMetric, Method2Metric
    from utils.toy_dataset import GaussianMixture
    from train_EBM_geodesic import get_metrics_dict

    # %%
    DEVICE: str = "cuda:1"

    NB_GAUSSIANS = 200
    RADIUS = 8
    mean_ = (torch.linspace(0, 180, NB_GAUSSIANS + 1)[0:-1] * math.pi / 180)
    MEAN = RADIUS * torch.stack([torch.cos(mean_), torch.sin(mean_)], dim=1)
    COVAR = torch.tensor([[1., 0], [0, 1.]]).unsqueeze(0).repeat(len(MEAN), 1, 1)

    x_p, y_p = torch.meshgrid(torch.linspace(-10, 10, 100), torch.linspace(-2.5, 10, 62), indexing='xy')
    x_t = torch.cat([x_p.flatten().unsqueeze(1), y_p.flatten().unsqueeze(1)], dim=1).to(DEVICE)

    ## Defining the mixture
    weight_1 = (torch.ones(NB_GAUSSIANS) / NB_GAUSSIANS)
    mixture_1 = GaussianMixture(center_data=MEAN, covar=COVAR, weight=weight_1).to(DEVICE)
    mixture_1 = mixture_1.to(DEVICE)

    ## compute the energy landscape
    energy_landscape_1: Tensor = mixture_1.energy(x_t)
    print(f"{energy_landscape_1.shape=}")


    num_samples: int = int(1000)
    sample_dataset = mixture_1.sample(num_samples).to(DEVICE)
    reference_samples = mixture_1.sample(num_samples)
    ## ebm-based metric
    loaded: dict = torch.load("./tutorial/EBM_mixture1.pth", weights_only=False)

    ebm = loaded['type']()
    ebm.load_state_dict(loaded['weight'])
    ebm.to(DEVICE)

    target_metric_cfg: dict = {
    "_target_": "train_EBM_geodesic.Method3Metric",
        "a_num": 20.0,
        "b_num": 0.01,
        "eps": 1e-6,
        "eta": 0.1,
        "mu": 1.0,
        "beta": 0.1,
        "alpha_fn_choice": "linear"
    }

    method3_metric: Method3Metric = hydra.utils.instantiate(target_metric_cfg, ebm=ebm)

    x_t.requires_grad_(True)
    one_form_all: Tensor = method3_metric.one_form(x_t)
    print(f"{one_form_all.shape=}")