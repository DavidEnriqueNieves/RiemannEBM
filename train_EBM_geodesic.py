# %%
from typing import Set
from cbs import plot_end_comparison
from dataclasses import dataclass   
from datetime import datetime
import pickle
import plotly
from hydra import initialize, compose
from hydra.utils import instantiate
from metrics import h_diag_RBF, Method3Metric, Method2Metric, h_diag_Land, DiagonalRiemannianMetric, ConformalRiemannianMetric
from model import MLP_ELU_convex
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from plotly import graph_objects as go
from sklearn.cluster import KMeans
from torch import Tensor
from tqdm import tqdm
from typing import Tuple
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
        
# # %%
# ## helper function
def linear_normalization(maxi, mini, target_max, target_min):
    alpha = (target_max - target_min)/(maxi - mini)
    beta = target_min - alpha*mini
    return alpha, beta

def normalize_diag(h, dataset_sample, mini=1e-4, maxi=2):
    all_h = h(dataset_sample)
    if len(all_h.shape)==2:
        print(f'normalizing diag metric')
        alpha, beta = [], []
        for i in range(all_h.shape[1]):
            alpha_, beta_ = linear_normalization(all_h[:,i].max(), all_h[:,i].min(), target_max=maxi,target_min=mini )
            alpha.append(alpha_)
            beta.append(beta_)
        alpha, beta = torch.stack(alpha), torch.stack(beta)
        return alpha.unsqueeze(0), beta.unsqueeze(0)

def get_metrics_dict(cfg: DictConfig, mixture_1: nn.Module, pos: Tensor, ebm: nn.Module, reference_samples: Tensor, DEVICE: str) -> dict:
    ## Metric based on 1/p
    true_p = lambda x: mixture_1.prob(x) ## works
    p_max, p_min = true_p(pos).max().item(), true_p(pos).min().item() 
    alpha_p, beta_p = linear_normalization(p_max, p_min, 1, 1e-3)
    true_p_n = lambda x: alpha_p*true_p(x) + beta_p
    inv_p = lambda x: 1/true_p_n(x)
    print(f"inv_p -- mini : {inv_p(pos).min():0.3f}, maxi : {inv_p(pos).max():0.3f}")

    ## Metric based on 1/exp(-E) (ebm)
    ebm_p = lambda x: torch.exp(-ebm(x))
    ebm_p_max, ebm_p_min = ebm_p(pos).max().item(), ebm_p(pos).min().item()
    alpha_ebm_p, beta_ebm_p = linear_normalization(ebm_p_max, ebm_p_min, 1, 1e-3)
    ebm_p_n = lambda x: alpha_ebm_p*ebm_p(x) + beta_ebm_p
    inv_ebm_p = lambda x: 1/ebm_p_n(x)
    print(f"inv_ebm_p -- mini : {inv_ebm_p(pos).min():0.3f}, maxi : {inv_ebm_p(pos).max():0.3f}")

    ## Metric based on logp
    true_en = lambda x: -torch.log(mixture_1.prob(x))
    en_max, en_min = true_en(pos).max().item(), true_en(pos).min().item() 
    alpha_en, beta_en = linear_normalization(en_max, en_min, 1e3, 0)
    true_en_n = lambda x: 1 + alpha_en*true_en(x) + beta_en
    print(f"true_en_n -- mini : {true_en_n(pos).min():0.3f}, maxi : {true_en_n(pos).max():0.3f}")

    ## Metric based on E (ebm)
    ebm_en = lambda x: ebm(x)
    ebm_en_max, ebm_en_min = ebm_en(pos).max().item(), ebm_en(pos).min().item()
    alpha_ebm_en, beta_ebm_en = linear_normalization(ebm_en_max, ebm_en_min, 1e3, 0)
    ebm_en_n = lambda x: 1 + alpha_ebm_en*ebm_en(x) + beta_ebm_en
    print(f"ebm_en_n -- mini : {ebm_en_n(pos).min():0.3f}, maxi : {ebm_en_n(pos).max():0.3f}")


    ### land metric
    h_land = h_diag_Land(reference_samples, gamma=1)
    alpha_land, beta_land = normalize_diag(h_land.h, pos, mini=1e-3, maxi=1)
    land_n = lambda x: 1/(alpha_land*h_land.h(x) + beta_land)
    print(f"g_land -- mini : {land_n(pos).min()}, maxi : {land_n(pos).max()}")

    ### rbf metric
    h_rbf = h_diag_RBF(n_centers = 30,  data_to_fit_ambiant=reference_samples, data_to_fit_latent=reference_samples).to(DEVICE)
    alpha_rbf, beta_rbf = normalize_diag(h_rbf.h, pos, mini=1e-3, maxi=1)
    rbf_n = lambda x: 1/(alpha_rbf*h_rbf.h(x) + beta_rbf)
    print(f"g_rbf -- mini : {rbf_n(pos).min()}, maxi : {rbf_n(pos).max()}")

    dico_metric_unif = {
        # "diag_rbf_invp" : DiagonalRiemannianMetric(rbf_n),
        # "conf_ebm_invp": ConformalRiemannianMetric(inv_ebm_p),
        # "conf_true_invp": ConformalRiemannianMetric(inv_p),
        # "conf_ebm_logp": ConformalRiemannianMetric(ebm_en_n),
        # "conf_true_logp": ConformalRiemannianMetric(true_en_n), 
        # "diag_land_invp": DiagonalRiemannianMetric(land_n),
    } 

    for metric_def in cfg.training.metrics:
        print("Instantiating metric under: ")
        print(OmegaConf.to_yaml(metric_def))

        metric_obj: nn.Module = instantiate(metric_def, ebm=ebm)
        assert isinstance(metric_obj, nn.Module), f"Instantiated metric is not a nn.Module, got {type(metric_obj)}"
        dico_metric_unif[metric_def._target_] = metric_obj 

    return dico_metric_unif


@hydra.main(version_base=None, config_path="configs", config_name="train_geodesic_intrp", )
def main(cfg: DictConfig):

    root_exps_dir: Path = Path("./exps")
    root_exps_dir.mkdir(exist_ok=True)

    exp_group: str = cfg.meta.experiment_group
    exp_name: str = cfg.meta.experiment_name
    
    timestr: str = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir: Path = root_exps_dir / exp_group / f"{exp_name}_{timestr}"

    exp_dir.mkdir(exist_ok=True, parents=True)

    print(f"Experiment directory: {exp_dir}")

    
    config_path: Path = exp_dir / "config.yaml"
    OmegaConf.save(config=cfg, f=config_path)
    print(f"Configuration saved to {config_path}")

    if cfg.debug:
        ipdb.set_trace()

    # %%
    DEVICE: str = cfg.device

    NB_GAUSSIANS = cfg.dataset.nb_gaussians
    RADIUS = cfg.dataset.shape_radius
    mean_ = (torch.linspace(0, 180, NB_GAUSSIANS + 1)[0:-1] * math.pi / 180)
    MEAN = RADIUS * torch.stack([torch.cos(mean_), torch.sin(mean_)], dim=1)
    COVAR = torch.tensor([[1., 0], [0, 1.]]).unsqueeze(0).repeat(len(MEAN), 1, 1)

    x_p, y_p = torch.meshgrid(torch.linspace(-10, 10, 100), torch.linspace(-2.5, 10, 62), indexing='xy')
    pos = torch.cat([x_p.flatten().unsqueeze(1), y_p.flatten().unsqueeze(1)], dim=1).to(DEVICE)

    ## Defining the mixture
    weight_1 = (torch.ones(NB_GAUSSIANS) / NB_GAUSSIANS)
    mixture_1 = GaussianMixture(center_data=MEAN, covar=COVAR, weight=weight_1).to(DEVICE)
    mixture_1 = mixture_1.to(cfg.device)

    ## compute the energy landscape
    energy_landscape_1: Tensor = mixture_1.energy(pos)
    print(f"{energy_landscape_1.shape=}")
    shape_info: dict = parse_shape(energy_landscape_1, 'B')

    T_STEPS: int = int(cfg.training.t_steps)
    dt: float = 1.0/(T_STEPS-1)

    # %%
    num_samples: int = int(cfg.training.num_samples)
    sample_dataset = mixture_1.sample(num_samples).to(DEVICE)
    reference_samples = mixture_1.sample(num_samples)
    ## ebm-based metric
    loaded: dict = torch.load("./tutorial/EBM_mixture1.pth", weights_only=False)

    ebm = loaded['type']()
    ebm.load_state_dict(loaded['weight'])
    ebm.to(DEVICE)
    
    metric_dict: dict = get_metrics_dict(
        cfg,
        mixture_1,
        pos,
        ebm,
        reference_samples,
        DEVICE)
   
    plot_init=True
    EPOCH: int = int(cfg.training.epochs)
    load = False
    MEAN: Tensor = mixture_1.means.to(DEVICE).detach()
    RADIUS: float = float(cfg.dataset.shape_radius)

    losses: dict[str, list] = {metric: [] for metric in metric_dict.keys()}

    #EPOCH = 1
    dico_traj = {}
    
    for metric in metric_dict.keys():
        print(f"\n\n {metric}")
        riemann_metric = metric_dict[metric]
        
        z0 = MEAN[10].unsqueeze(0).to(DEVICE).detach()
        z1 = MEAN[-10].unsqueeze(0).to(DEVICE).detach()
        
        t = torch.linspace(0, 1, T_STEPS).unsqueeze(1).to(DEVICE).detach()
        dt = 1.0/(T_STEPS-1)
        
        print("Initializing trajectory...")
        z_t: Tensor = (1-t)*z0 + t*z1
        
        # TODO: bring this back
        # if plot_init:
        #     print("Initial plot...")
        #     initial_plot_fn(DEVICE, T_STEPS, dt, RADIUS, riemann_metric, z_t)
    
        z_i: Tensor = z_t[1:-1].requires_grad_(True)
        optimizer = torch.optim.Adam([z_i], lr=1e-3)
        all_loss = []
        
        print(f"Starting training for metric {metric}...")
        for ep in tqdm(range(EPOCH)):
            optimizer.zero_grad()
            
            # TODO: fix this 
            # if metric not in [Method2Metric.__class__, Method3Metric]:
            z_t: Tensor = torch.cat([z0,z_i,z1])
            z_t_dot: Tensor = (z_t[1:] - z_t[:-1])/dt
            energy: Tensor = riemann_metric.kinetic(z_t[:-1], z_t_dot)

            z_t_inv: Tensor = torch.flip(z_t, dims=[0])
            z_t_dot_inv: Tensor = (z_t_inv[1:] - z_t_inv[:-1])/dt
            energy_inv: Tensor = riemann_metric.kinetic(z_t_inv[:-1], z_t_dot_inv)
        
            total_energy: Tensor = (energy + energy_inv)/2
            #total_energy = energy
            loss: Tensor = (total_energy*dt).sum()
            loss.backward()

            
            all_param = 0.0
            with torch.no_grad():
                all_param+=z_i.grad.norm()
            optimizer.step()
            all_loss.append(loss.item())
        
            with torch.no_grad():
                if ep % 10000 == 0:
                    z_t = plot_intermittent(x_p, y_p, energy_landscape_1, EPOCH, z_t, all_loss, ep)
            dico_traj[metric] =  z_t.cpu().detach()    
    torch.save(dico_traj, "./ConfMetric_traj_uniform.dico")
    
    normed_m2_color: Tensor = torch.tensor([147,112,219]) / 255.0
    normed_m3_color: Tensor = torch.tensor([128,0,128]) / 255.0
    m2_color: tuple = tuple(normed_m2_color.tolist())
    m3_color: tuple = tuple(normed_m3_color.tolist())
    
    dico_color={
     "conf_ebm_logp": [(0.2588, 0.5216, 0.9569), 150, '.', r"$\mathbf{G}_{E_{\theta}}$"], ## blue google (66, 133, 244)
    "conf_ebm_invp": [(0.2039, 0.6588, 0.3255), 150,'.', r"$\mathbf{G}_{1/p_{\theta}}$"], #r"$h(x) \propto 1 / \exp(-E_{\theta}(x))$"], ## green google (52, 168, 83)
    "diag_rbf_invp":[(0.9176, 0.2627, 0.2078), 150,'.', r"$\mathbf{G}_{RBF}$"], #r"$h(x) \propto h_{RBF}$"], ## red google (234, 67, 53)
    "diag_land_invp":[(0.9843, 0.7373, 0.0196), 150,'.', r"$\mathbf{G}_{LAND}$"], ## yellow google (251, 188, 5)
    "conf_true_logp": ['black',150,'+' , r"$\mathbf{G}_{E_{\mathcal{M}}}$"],
    "conf_true_invp": ['black',150, '2', r"$\mathbf{G}_{1/p_{\mathcal{M}}}$"],
    "train_EBM_geodesic.Method2Metric":[m2_color, 150,'.', r"$\mathbf{G}_{M2}$"], ## yellow google (251, 188, 5)
    "train_EBM_geodesic.Method3Metric":[m3_color, 150,'.', r"$\mathbf{G}_{M3}$"], ## yellow google (251, 188, 5)
    }

    fig, ax = plt.subplots(1, 1, figsize=(8,6), dpi=100)
    im = ax.contourf(x_p, y_p, energy_landscape_1.view(62, 100).detach().cpu(), 20,
                                cmap='Blues_r',
                                alpha=0.8,
                                zorder=0,
                                levels=20)
    color = ['orange', 'black', 'grey', 'red', 'green','purple','pink','yellow']

    reg_space = torch.linspace(0, T_STEPS - 1, 25).long()
    
    if cfg.debug:
        ipdb.set_trace()

    for idx, metric in enumerate(dico_traj.keys()):
        print(metric)
        z_t = dico_traj[metric]
        z_t_scat = z_t[reg_space]
        if metric in ['conf_true_logp', 'conf_true_invp']:
            ax.scatter(z_t_scat[1:-1, 0], z_t_scat[1:-1, 1], color=dico_color[metric][0], alpha=1,
                        s=dico_color[metric][1], marker=dico_color[metric][2], label=dico_color[metric][3], zorder=3)
        else:
            ax.plot(z_t[:, 0], z_t[:, 1], color=dico_color[metric][0], linewidth=3, zorder=1,label=dico_color[metric][3], alpha=1)
            ax.scatter(z_t_scat[1:-1, 0], z_t_scat[1:-1, 1], color=dico_color[metric][0], alpha=1,
                        s=dico_color[metric][1], marker=dico_color[metric][2], zorder=2)
            
        #ax.scatter(z_t[1:-1,0], z_t[1:-1,1], s=10, color=dico_color[metric][0], alpha=1, label=str(metric))
        #ax.scatter(z_t[0,0], z_t[0,1], s=10, color='red', alpha=1)
        #ax.scatter(z_t[-1,0], z_t[-1,1], s=10, color='green', alpha=1)
    # Add red and green dots
    print(f"Adding red and green dots")
    print(f"{MEAN[10].shape=}, {MEAN[-10].shape=}")
    ax.scatter(MEAN[10][0].cpu().detach(), MEAN[10][1].cpu().detach(), s=70, color='red',alpha=1)
    ax.scatter(MEAN[-10][0].cpu().detach(), MEAN[-10][1].cpu().detach(), s=70, color='green',alpha=1)
    # ax.set_axis_off()
    plt.legend()
    plt.savefig("plots/final_geodesic_comparison.png")


if __name__ == "__main__":
    main()