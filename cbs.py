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
from metrics import h_diag_RBF, Method3Metric, Method2Metric, h_diag_Land, DiagonalRiemannianMetric, ConformalRiemannianMetric
sys.path.append("../")

def plot_end_comparison(cfg, MEAN, x_p, y_p, energy_landscape_1, T_STEPS, last_traj):
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

    for idx, metric in enumerate(last_traj.keys()):
        print(metric)
        z_t = last_traj[metric]
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