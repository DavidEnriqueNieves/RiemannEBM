"""
Script to run an EBM training with contrastive divergence, this time with PyTorch Lightning
"""
# %%
import sys
import json
from tqdm import tqdm
from torch.optim import Optimizer
from utils.samplers import SgldSampler
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import ipdb
import math
from utils.toy_dataset import GaussianMixture
from torch.optim.lr_scheduler import CosineAnnealingLR
from hydra.utils import instantiate

import hydra
from omegaconf import OmegaConf, DictConfig

# %% [markdown]
# ## Parameter of the distribution

# %%

def plot_energy_landscape(DEVICE: str, mixture_1: GaussianMixture):
    x_p, y_p = torch.meshgrid(torch.linspace(-1.5, 1.5, 100), torch.linspace(-1.5, 1.5, 100), indexing='xy')
    pos = torch.cat([x_p.flatten().unsqueeze(1), y_p.flatten().unsqueeze(1)], dim=1).to(DEVICE)

    fig, ax = plt.subplots(1, 2, figsize=(2*4, 4), dpi=100)
# ipdb.set_trace()
    sample_1 = mixture_1.sample(1000).cpu().detach()
#offset_1 = torch.tensor([0.0, 4.0])
#mult_1  = torch.tensor([10.0, 6.0])

    energy_landscape_1 = mixture_1.energy(pos)

    ax[0].scatter(sample_1[:,0],sample_1[:,1])
    im = ax[1].contourf(x_p, y_p, energy_landscape_1.view(100, 100).detach().cpu(), 20,
                        cmap='Blues_r',
                        alpha=0.8,
                        zorder=0,
                        levels=20)
    plt.savefig("plots/GaussianMixture_Uniform.png")

@hydra.main(version_base=None, config_path="configs", config_name="train_latent_ebm", )
def main(cfg: DictConfig):

    print(f"{json.dumps(OmegaConf.to_container(cfg), indent=4)}")

    mixture_1 = instantiate(cfg.dataset)
    mixture_1 = mixture_1.to(cfg.device)
    print(f"{type(mixture_1)=}")

    x_p, y_p = torch.meshgrid(torch.linspace(-1.5, 1.5, 100), torch.linspace(-1.5, 1.5, 100), indexing='xy')
    pos = torch.cat([x_p.flatten().unsqueeze(1), y_p.flatten().unsqueeze(1)], dim=1).to(cfg.device)

    plot_energy_landscape(cfg.device, mixture_1)

    netE: nn.Module = instantiate(cfg.model).to(cfg.device) 
    print(f"Instantiated model: {netE}")
    # summary(netE, (2,), device=cfg.device)  

    sampler: SgldSampler = instantiate(cfg.training.sampler)
    print(f"Instantiated sampler: {sampler}")
    optimizer: Optimizer = instantiate(cfg.training.optimizer, params=netE.parameters())
    print(f"Instantiated optimizer: {optimizer}")
    
    EPOCH: int = cfg.training.epochs
    batch_size: int = cfg.training.batch_size
    all_loss = []
    for i in tqdm(range(EPOCH)):
        optimizer.zero_grad()
        #x = (mixture_1.sample(BATCH_SIZE) - mean_mix1)/std_mix1
        x = mixture_1.sample(batch_size)
        
        x_i = torch.randn_like(x)
        x_s = sampler.sample(netE, x_i)
        
        fp_all = netE.forward(x)
        fq_all = netE.forward(x_s)
        
        fp = fp_all.mean()
        fq = fq_all.mean()
        l_p_x = fp - fq
        
        
        loss = l_p_x
        loss.backward()
        optimizer.step()
        
        all_loss.append(loss.item())
        # if i % 10 ==0 :
        #     print(f"{i} -- loss : {loss.item():0.6f}")

        if i % 500 == 0 or i == EPOCH-1:
            x_i = torch.randn_like(x)
            x_s = sampler.sample(netE, x_i)
            all_e = netE.forward(pos)
            fig, ax = plt.subplots(1, 3, figsize=(3*8,8))
            ax[0].plot(all_loss)
            ax[1].scatter(x_s[:,0].cpu().detach(), x_s[:,1].cpu().detach())
            im = ax[2].contourf(x_p, y_p, all_e.view(100,100).detach().cpu(), 20, cmap='Blues_r', alpha=0.8, zorder=0, levels=20)
            fig.colorbar(im, ax=ax[2])
            plt.savefig(f"plots/EBM_Uniform_CD_epoch_{i}.png")
    to_save = {"weight": netE.state_dict(),
                "type": type(netE)}
    torch.save(to_save, f"./EBM_Uniform_CD.pth")
    # else:
    #     ckpt = torch.load(f"./EBM_Uniform_CD.pth", weights_only=False)
    #     netE.load_state_dict(ckpt["weight"], strict=True)
    #     en = lambda x: netE(x)
    #     x_i = torch.randn(BATCH_SIZE, 2).to(DEVICE)
    #     x_s = sgld(en, x_i, n_steps=SGLD_STEPS, sgld_std=SGLD_STD, sgld_lr=SGLD_LR)
    #     #x_p, y_p = torch.meshgrid(torch.linspace(-5, 5, 100), torch.linspace(-5, 5, 100), indexing='xy')
    #     #pos = torch.cat([x_p.flatten().unsqueeze(1), y_p.flatten().unsqueeze(1)], dim=1).to(DEVICE)
    #     all_e = en(pos)
    #     fig, ax = plt.subplots(1, 2, figsize=(2*8,8))
    #     ax[0].scatter(x_s[:,0].cpu().detach(), x_s[:,1].cpu().detach())
    #     im = ax[1].contourf(x_p, y_p, all_e.view(100,100).detach().cpu(), 20, cmap='Blues_r', alpha=0.8, zorder=0, levels=20)
    #     fig.colorbar(im, ax=ax[1])
    #     plt.show()

    print("Training finished!")

if __name__ == "__main__":
    main()


