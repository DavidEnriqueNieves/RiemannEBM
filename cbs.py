from dataclasses import dataclass   
from argparse import ArgumentParser, Namespace
from datetime import datetime
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
from typing import Set, Tuple
from utils.toy_dataset import GaussianMixture
import einops
import hydra
import ipdb
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import plotly
import sys
import torch
import torch.nn as nn
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

@dataclass
class AnimationData:
    color_tuple: tuple
    size: int
    symbol: str
    latex_label: str

def create_geodesic_animation(
        all_metric_timed_trajs: dict[str, Tensor],
        animation_savepath: Path,
        subs_epoch_mod: int = 1000,
        animation_title: str = "Geodesic Trajectories Over Training",
        subs_path_mod: int = 5
    ) -> None:
        """
        Create and save an animated visualization of geodesic trajectories for different metrics.
        
        Args:
            all_metric_timed_trajs: Dictionary mapping metric names to trajectory tensors of shape (EPOCH, T_STEPS, 2)
            exp_dir: Directory path where animation and data will be saved
            subsample_amount: Subsampling factor for animation frames (default: 1000)
        """

        lowest_num_epochs: int = min(traj.shape[0] for traj in all_metric_timed_trajs.values())
        epoch: int = int(lowest_num_epochs)

        normed_m2_color: Tensor = torch.tensor([147, 112, 219]) 
        normed_m3_color: Tensor = torch.tensor([128, 0, 128])
        m2_color: tuple = tuple(normed_m2_color.tolist())
        m3_color: tuple = tuple(normed_m3_color.tolist())

            
        if epoch > 50_000:
            all_frame_idxs_subsmpl: Tensor = torch.cat((torch.arange(0, 500, 10), torch.tensor([1_000, 2_000, 3_000, 4_000, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000-1])), dim=0)
        else:
            all_frame_idxs_subsmpl: Tensor = torch.arange(0, epoch, subs_epoch_mod)

        dico_color: dict[str, AnimationData] = {
            "conf_ebm_logp": AnimationData(color_tuple=(66, 133, 244), size=150, symbol='.', latex_label=r"$\mathbf{G}_{E_{\theta}}$"),
            "conf_ebm_invp": AnimationData(color_tuple=(52, 168, 83), size=150, symbol='.', latex_label=r"$\mathbf{G}_{1/p_{\theta}}$"),
            "diag_rbf_invp": AnimationData(color_tuple=(234, 67, 53), size=150, symbol='.', latex_label=r"$\mathbf{G}_{RBF}$"),
            "diag_land_invp": AnimationData(color_tuple=(251, 188, 5), size=150, symbol='.', latex_label=r"$\mathbf{G}_{LAND}$"),
            "conf_true_logp": AnimationData(color_tuple=(0, 0, 0), size=150, symbol='+', latex_label=r"$\mathbf{G}_{E_{\mathcal{M}}}$"),
            "conf_true_invp": AnimationData(color_tuple=(0, 0, 0), size=150, symbol='x', latex_label=r"$\mathbf{G}_{1/p_{\mathcal{M}}}$"),
            "train_EBM_geodesic.Method2Metric": AnimationData(color_tuple=m2_color, size=150, symbol='.', latex_label=r"$\mathbf{G}_{M2}$"),
            "train_EBM_geodesic.Method3Metric": AnimationData(color_tuple=m3_color, size=150, symbol='.', latex_label=r"$\mathbf{G}_{M3}$"),
        }
        
        fig: go.Figure = go.Figure()

        fig.update_layout(
            title=animation_title,
            xaxis_title="X-axis",
            yaxis_title="Y-axis",
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )

        all_frames_data: list[list[go.Trace]] = []
        
        print(f"Preparing animation frames for {len(all_frame_idxs_subsmpl)} frames and metrics: {list(all_metric_timed_trajs.keys())}")

        metric_keys: list[str] = list(all_metric_timed_trajs.keys())
        print(f"Metrics for which we have trajectories: {metric_keys}")
        print(f"Example trajectory shape for metric {metric_keys[0]}: {all_metric_timed_trajs[metric_keys[0]].shape}")

        for t in all_frame_idxs_subsmpl:

            frame_data: list[go.Trace] = [] 
            for m_idx, metric in enumerate(all_metric_timed_trajs.keys()):
                # if m_idx == 0:
                    # print(f"Adding metric {metric} with color {dico_color[metric].color_tuple} and label {dico_color[metric].latex_label}")
                    # print(f"{all_metric_timed_trajs[metric].shape=}")
                    # print(f"{all_metric_timed_trajs[metric][::subs_epoch_mod, ::subs_path_mod, :].shape=}")
                    # print(f"{all_frame_idxs_subsmpl=}")

                frame_data.append(
                    go.Scatter(
                        name=dico_color[metric].latex_label,
                        x=all_metric_timed_trajs[metric][t, :, 0].cpu().numpy(),
                        y=all_metric_timed_trajs[metric][t, :, 1].cpu().numpy(),
                        mode='lines+markers',
                        marker=dict(size=5, color=plotly.colors.label_rgb(dico_color[metric].color_tuple), opacity=0.6)
                    )
                )
            all_frames_data.append(frame_data)

        assert len(all_frames_data) == len(all_frame_idxs_subsmpl), f"Expected {len(all_frame_idxs_subsmpl)} frames, but got {len(all_frames_data)}"
        frame_zero_data: list[go.Trace] = all_frames_data[0]
        fig.add_traces(frame_zero_data)
            
        all_frames: list[go.Frame] = []
        print(f"{len(all_frames_data)=}")

        for t, frame_data in zip(all_frame_idxs_subsmpl.tolist(), all_frames_data):
            all_frames.append(
                go.Frame(
                    data=frame_data,
                    name=str(t)
                )
            )
        fig.frames = all_frames

        slider_steps = [
            dict(
                method="animate",
                args=[
                    [str(t)],
                    dict(
                        mode="immediate",
                        frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0),
                    ),
                ],
                label=str(t),
            )
            for t in all_frame_idxs_subsmpl.tolist()
        ]

        fig.update_layout(
            title="Random Paths on Unit Circle",
            showlegend=True,
            xaxis_title="X-axis",
            yaxis_title="Y-axis",
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            sliders=[
                dict(
                    active=0,
                    currentvalue=dict(prefix="t = "),
                    pad=dict(t=50),
                    steps=slider_steps,
                )
            ],
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=1,
                    x=1.1,
                    xanchor="right",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=100, redraw=True),
                                    fromcurrent=True,
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(frame=dict(duration=0, redraw=False), mode="immediate"),
                            ],
                        ),
                    ],
                )
            ]
        )
        
        print(f"Saving animation to {animation_savepath}")
        fig.write_html(str(animation_savepath), include_mathjax="cdn")


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--traj_paths", type=str, required=True, help="Trajectory directory path containing all_metric_timed_trajs.pt or a similar file with the expected structure")
    parser.add_argument("--output_file", type=str, required=True, help="File where the output will be saved")

    args: Namespace = parser.parse_args()

    traj_path: Path = Path(args.traj_paths)
    assert traj_path.exists(), f"Trajectory directory {traj_path} does not exist"
    assert traj_path.is_file(), f"Provided traj_path {traj_path} is not a file"

    output_file: Path = Path(args.output_file)
    assert output_file.suffix == ".html", f"Output file must have .html extension, got {output_file.suffix}"

    print(f"Loading trajectory data from {traj_path}")
    all_metric_timed_trajs: dict[str, Tensor] = torch.load(traj_path)
    print(f"Successfully loaded trajectory data with keys: {list(all_metric_timed_trajs.keys())}")

    animation_savepath: Path = output_file
    create_geodesic_animation(
        all_metric_timed_trajs=all_metric_timed_trajs,
        animation_savepath=animation_savepath,
        subs_epoch_mod=10000
    )