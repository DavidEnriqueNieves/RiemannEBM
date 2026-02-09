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

    # path to the cached metric trajectories, if they exist. This is useful for
    # avoiding re-training the geodesics if we just want to change the plotting
    # or analysis code. If the file does not exist, it will be created at the
    # end of the training loop.
    metric_traj_cachepath: Path = Path("./metric_traj_paths.pt")

    all_metric_timed_trajs: dict[str, Tensor] = None
    if metric_traj_cachepath.exists():

        if metric_traj_cachepath.is_file():
            all_metric_timed_trajs = torch.load(metric_traj_cachepath)
        else:
            raise ValueError(f"Path {metric_traj_cachepath} exists and is not a file. Please check the path and remove any directories if necessary.")
    else:
        all_metric_timed_trajs: dict[str, Tensor] = {}

    missing_metrics: Set[str] = set(metric_dict.keys()) - set(all_metric_timed_trajs.keys())
    
    if len(missing_metrics) == len(all_metric_timed_trajs):
        print("No cached trajectories found, starting training for all metrics.")
    else:
        print(f"Found cached trajectories for metrics: {set(all_metric_timed_trajs.keys())}. Will only train for missing metrics: {missing_metrics}")

    losses: dict[str, list] = {metric: [] for metric in metric_dict.keys()}

    all_metric_losses: dict[str, list] = {metric: [] for metric in metric_dict.keys()}

    # of shape (E, T_STEPS, 2)
    metric_zts: dict[str, Tensor] = { metric: torch.zeros(EPOCH, T_STEPS, 2) for metric in metric_dict.keys()}
    metric_zdot_ts: dict[str, Tensor] = { metric: torch.zeros(EPOCH, T_STEPS-1, 2) for metric in metric_dict.keys()}

    print(f"Currently missing trajectories for metrics: {missing_metrics}")

    for metric in missing_metrics:
        all_metric_timed_trajs[metric] = torch.zeros(EPOCH, T_STEPS, 2)

    for metric in missing_metrics:
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

            metric_zts[metric][ep] = z_t.cpu().detach()
            metric_zdot_ts[metric][ep] = z_t_dot.cpu().detach()
            
            all_param = 0.0
            with torch.no_grad():
                all_param+=z_i.grad.norm()
            optimizer.step()
            all_loss.append(loss.item())
        
            all_metric_timed_trajs[metric][ep] = z_t.cpu().detach()
            all_metric_losses[metric].append(loss.item())

    torch.save(all_metric_timed_trajs, exp_dir / "ConfMetric_traj_uniform.dico")
            dico_traj[metric] =  z_t.cpu().detach()    
    torch.save(dico_traj, "./ConfMetric_traj_uniform.dico")
    
    normed_m2_color: Tensor = torch.tensor([147,112,219]) 
    normed_m3_color: Tensor = torch.tensor([128,0,128])
    m2_color: tuple = tuple(normed_m2_color.tolist())
    m3_color: tuple = tuple(normed_m3_color.tolist())

    @dataclass
    class AnimationData:
        color_tuple: tuple
        size: int
        symbol: str
        latex_label: str

    dico_color: dict[str, AnimationData]={
        "conf_ebm_logp": AnimationData(color_tuple=(66, 133, 244), size=150, symbol='.', latex_label=r"$\mathbf{G}_{E_{\theta}}$"), ## blue google (66, 133, 244)
        "conf_ebm_invp": AnimationData(color_tuple=(52, 168, 83), size=150, symbol='.', latex_label=r"$\mathbf{G}_{1/p_{\theta}}$"), ## green google (52, 168, 83)
        "diag_rbf_invp": AnimationData(color_tuple=(234, 67, 53), size=150, symbol='.', latex_label=r"$\mathbf{G}_{RBF}$"), ## red google (234, 67, 53)  
        "diag_land_invp": AnimationData(color_tuple=(251, 188, 5), size=150, symbol='.', latex_label=r"$\mathbf{G}_{LAND}$"), ## yellow google (251, 188, 5)
        "conf_true_logp": AnimationData(color_tuple=(0, 0, 0), size=150, symbol='+', latex_label=r"$\mathbf{G}_{E_{\mathcal{M}}}$"),
        "conf_true_invp": AnimationData(color_tuple=(0, 0, 0), size=150, symbol='x', latex_label=r"$\mathbf{G}_{1/p_{\mathcal{M}}}$"),
        "train_EBM_geodesic.Method2Metric":AnimationData(color_tuple=m2_color, size=150, symbol='.', latex_label=r"$\mathbf{G}_{M2}$"), ## yellow google (251, 188, 5)
        "train_EBM_geodesic.Method3Metric":AnimationData(color_tuple=m3_color, size=150, symbol='.', latex_label=r"$\mathbf{G}_{M3}$"), ## yellow google (251, 188, 5)
        }
    

    fig: go.Figure = go.Figure()

    fig.update_layout(
        title="Random Paths on Unit Circle",
        xaxis_title="X-axis",
        yaxis_title="Y-axis",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )

    all_frames_data: list[list[go.Trace]] = []
    for t in range(EPOCH):
        frame_data: list[go.Trace] = [] 

        for metric in all_metric_timed_trajs.keys():
            if t == 0:
                print(f"Adding metric {metric} with color {dico_color[metric].color_tuple} and label {dico_color[metric].latex_label}")

            frame_data.append(
                go.Scatter(
                    name=dico_color[metric].latex_label,
                    x=all_metric_timed_trajs[metric][t, :, 0].cpu().numpy(),
                    y=all_metric_timed_trajs[metric][t, :, 1].cpu().numpy(),
                    mode='lines+markers', marker=dict( size=5, color=plotly.colors.label_rgb(dico_color[metric].color_tuple), opacity=0.6)
                )
            )
        all_frames_data.append(frame_data)

    frame_zero_data: list[go.Trace] = all_frames_data[0]
    fig.add_traces(frame_zero_data)
        
    all_frames: list[go.Frame] = []

    for t, frame_data in enumerate(all_frames_data):
        #  print(f"{t=}, {len(frame_data)=}")
        all_frames.append(
            go.Frame(
                data=frame_data,
                name=str(t)
                )
            )
    fig.frames = all_frames

    # Slider steps (one per frame)
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
        for t in range(EPOCH)
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
    
    animation_savepath: Path = exp_dir / "random_paths_animation.html"
    print(f"Saving animation to {animation_savepath}")
    fig.write_html(str(animation_savepath), include_mathjax="cdn")

    pickle_savepath: Path = exp_dir / "all_metric_timed_trajs.pt"
    print(f"Saving all_metric_timed_trajs to {pickle_savepath}")
    torch.save(all_metric_timed_trajs, pickle_savepath)

    # Caching the commonly used metric trajectories for faster plotting
    # removing trajectories for our two metrics we are training 

    for metric in cfg.training.metrics:
        metric_key = metric._target_
        if metric_key in all_metric_timed_trajs:
            print(f"Removing trajectory for metric {metric_key} from cached trajectories to save space.")
            del all_metric_timed_trajs[metric_key]

    print(f"Caching metric trajectories to {metric_traj_cachepath}")
    torch.save(all_metric_timed_trajs, metric_traj_cachepath)

    


if __name__ == "__main__":
    main()