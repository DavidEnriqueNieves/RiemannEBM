import torch
import argparse
from utils.dataset import get_dataloader_im
import torch.utils.data as data_utils
#from old.EBM.energy_func import MlpEnergy2, MlpEnergy
#from models.VanillaNet import VanillaNet_ELU_lt
from model.rae import RAE2, normalize, RAE_conv, normalize_1
from model.sampler import init_random, get_sample_q, Annealed_Langevin_E
from tqdm import tqdm
import os
import wandb
from utils.monitoring import make_grid, str2bool, plot_img, name_model
from model.VanillaNet import VanillaNet_ELU_2

import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
import numpy as np
from models.unet_base import UNetModelWrapper as Unet


def main(args):
    print(f"Training Latent EBM with {args.training} on {args.dataset} on device {args.device}")
    if args.device == 'meso':
        if args.model_name is None:
            args.model_name = name_model(args)
        args.device = torch.cuda.current_device()
    print(args.db_type)
    d_loader = get_dataloader_im(data_root=args.data_root,
                              dataset=args.dataset,
                              image_size=args.image_size,
                              device='cpu',
                              ae_name=args.db_type,
                              )
    dataloader = data_utils.DataLoader(d_loader,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       drop_last=True)

    if args.energy_func == 'VanillaNet_ELU_2':
        args_ebm = {
            'n_c': 4,
            'n_f': 128 * args.multiplier
        }
        netE = VanillaNet_ELU_2(**args_ebm).to(args.device)

    else:
        raise NotImplementedError()
    a=1


    if args.model_name is not None:
        path_to_save = os.path.join(args.save_root, args.dataset, args.model_name)
        print(path_to_save)
        os.makedirs(path_to_save, exist_ok=True)
        wandb.init(project='IM_mental_latent_ebm_' + args.dataset, config=vars(args), entity='victorboutin')
        wandb.run.name = args.model_name
        torch.save(args, path_to_save + '/param.config')

    if args.training == "cd" or args.training == 'cdsm':
        replay_buffer = init_random((args.buffer_size, 4, 16, 16), init_type=args.init_type)
        sample_q = get_sample_q(args)

    optimizerE = torch.optim.AdamW(netE.parameters(), lr=args.lr_init, weight_decay=1e-5)

    if args.gamma_scheduler != 0:
            scheduler = CosineAnnealingLR(optimizerE, args.epoch, eta_min=1e-6)

    if args.training == "dsm" or args.training == 'cdsm':
        replay_buffer = init_random((args.buffer_size, 4, 16, 16), init_type=args.init_type)
        sample_q = get_sample_q(args)
        diffusion_steps = 500
        s = 0.008
        timesteps = torch.tensor(range(0, diffusion_steps), dtype=torch.float32)
        schedule = torch.cos((timesteps / diffusion_steps + s) / (1 + s) * torch.pi / 2) ** 2

        baralphas = schedule / schedule[0]
        betas = 1 - baralphas / torch.concatenate([baralphas[0:1], baralphas[0:-1]])
        alphas = 1 - betas

        baralphas = baralphas.to(args.device)
        betas = betas.to(args.device)
        alphas = alphas.to(args.device)

        Nsampling = 2000  # exponential schedule with flat region in the beginning and end
        Tmax, Tmin = 100, 1
        T = Tmax * np.exp(-np.linspace(0, Nsampling - 1, Nsampling) * (np.log(Tmax / Tmin) / Nsampling))
        T = np.concatenate((Tmax * np.ones((500,)), T), axis=0)
        T = np.concatenate((T, Tmin * np.linspace(1, 0, 200)), axis=0)

        def noise(data, t):
            eps = torch.randn_like(data)
            t = torch.clip(t, min=1, max=diffusion_steps-1)
            noised_t = (baralphas[t] ** 0.5).repeat(1, data.shape[1], data.shape[2], data.shape[3]) * data + \
                     ((1 - baralphas[t]) ** 0.5).repeat(1, data.shape[1], data.shape[2], data.shape[3]) * eps
            noised_t_m = (baralphas[t-1] ** 0.5).repeat(1, data.shape[1], data.shape[2], data.shape[3]) * data + \
                     ((1 - baralphas[t-1]) ** 0.5).repeat(1, data.shape[1], data.shape[2], data.shape[3]) * eps
            return noised_t, noised_t_m, eps

    nb_param = sum([p.numel() for p in netE.parameters()])
    print(f'param {nb_param:0.2f}')
    for ep in range(args.epoch):
        netE.train()
        pbar = tqdm(dataloader)
        for idx_batch, (latent, label) in enumerate(pbar):
            latent = latent.to(args.device)
            latent = torch.clip(latent, max=2, min=-2)
            latent = latent / 2
    

            if args.training == 'dsm' or args.training == 'cdsm':
                timesteps = torch.randint(0, diffusion_steps, size=[len(latent), 1, 1, 1]).to(args.device)
                latent_noisy, latent_noisy_m, eps = noise(latent, timesteps)
                latent_noisy = latent_noisy.requires_grad_()
                en_dsm = netE(latent_noisy)
                grad_x = torch.autograd.grad(en_dsm.sum(), latent_noisy, create_graph=True)[0]
                latent_noisy.detach()

                
                sig = (1-baralphas[timesteps].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + 1e-3)
                dsm_loss = ((1/sig) * ((latent_noisy_m-latent_noisy) + grad_x)**2).sum()/len(latent)

            else:
                dsm_loss = torch.tensor(0.0)

            if args.training == 'cd' or args.training == 'cdsm':
                z_real = latent + args.sigma * torch.randn_like(latent)
                z_q, norm_gr, replay_buffer, z_lst_grad = sample_q(netE, replay_buffer, clip=args.gradient_clip, w_last_gradient=args.w_last_gradient)
                fp_all = netE(z_real)
                fq_all = netE(z_q)
                fp = fp_all.mean()
                fq = fq_all.mean()

                if args.w_regul != 0:
                    reg_low_en = (fp_all ** 2).mean() + (fq_all ** 2).mean()
                else:
                    reg_low_en = torch.tensor(0.0)
            

                grad_regul = torch.tensor(0.0)
                
                loss_last_grad = torch.tensor(0.0)
                loss_cd = (fp - fq) + args.w_regul*reg_low_en
            else:
                loss_cd = torch.tensor(0.0)
                

            l_p_x = loss_cd + args.dsm_weight * dsm_loss
            optimizerE.zero_grad()
            l_p_x.backward()
            torch.nn.utils.clip_grad_norm_(netE.parameters(), max_norm=1)
            

            all_grad_optim = []
            for param in netE.parameters():
                if param.grad is not None:
                    all_grad_optim.append(param.grad.data.norm(2))
            max_grad = max(all_grad_optim)
            if torch.isnan(max_grad).any():
                print('update')
            
            optimizerE.step()

            description = f"{ep + 1}/{args.epoch} -- loss: {l_p_x.item():.3f}"
            pbar.set_description(description)

        if args.gamma_scheduler != 0:
            scheduler.step()

        if args.model_name is not None:
            dico_log = {"train/loss": l_p_x.item(),
                        "train/lr": optimizerE.param_groups[0]['lr'],
                        "train/max_grad_norm_optim": max_grad.item()
                        }
            if args.training == 'cd' or args.training == 'cdsm':
                dico_log["train/loss_cd"] = loss_cd.item()
                dico_log["train/E_data"] = fp.item()
                dico_log["train/E_sampled"] = fq.item()
                dico_log["train/regul_low_en"] = reg_low_en.item()
                dico_log["train/regul_grad"] = grad_regul.item()
                dico_log["train/norm_grad"] = norm_gr.item()
                dico_log["train/regul_last_gradient"] = loss_last_grad.mean().item()
            if args.training == 'dsm' or args.training == 'cdsm':
                dico_log["train/loss_dsm"] = dsm_loss.item()

            wandb.log(dico_log)

        if ep % 10 == 0:
            netE.eval()
            
            rdn_buff = init_random((0, 4, 16, 16), init_type=args.init_type)
            
            z_q, norm_gr, _, _ = sample_q(netE, rdn_buff, n_steps=1000, clip=args.gradient_clip)
            
            z_q = z_q*2
            x_q = d_loader.to_im(z_q.detach())

            if args.model_name is not None:
                to_plot = wandb.Image(
                    make_grid(x_q.cpu().detach(), nrow=8, ncol=10, normalize=True,
                              scale_each=True),
                    caption='ep:{}'.format(ep))
                wandb.log({"gene": to_plot})


        if ep % 100 == 0:
            if args.model_name is not None:
                netE.cpu()
                to_save = {"weight": netE.state_dict(),
                           "replay_buffer": replay_buffer,
                           "optimizer": optimizerE.state_dict(),
                           "args": args_ebm,
                           "type": type(netE)
                           }
                torch.save(to_save, path_to_save + f'/ep_{ep}.model')
                netE.to(args.device)

    netE.cpu()
    to_save = {"weight": netE.state_dict(),
               "replay_buffer": replay_buffer,
               "optimizer": optimizerE.state_dict(),
               "args": args_ebm,
               "type": type(netE)
               }
    torch.save(to_save, path_to_save + f'/last.model')

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training Energy Based Models")
    ## DATA args
    parser.add_argument('--dataset', type=str, default='afhq', choices=['alphanum', 'cifar10', 'afhq', 'celebahq'])
    parser.add_argument('--db_type', type=str, default='stable_diff_14', choices=['stable_diff_14', 'stable_diff_14_aug', 'stable_diff_14_aug_max', 'stable_diff_14_aug_max_max','stable_diff_14_celeba','stable_diff_14_aug_max_rot'])
    parser.add_argument('--image_size', type=int, default=128, help='batch_size')
    parser.add_argument("--data_root", type=str, default="/media/data_cifs_lrs/projects/prj_mental/datasets")
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')

    ## TRAIN args
    parser.add_argument("--training", type=str, default='cd', choices=['cd', 'dsm', 'cdsm'])
    parser.add_argument('--epoch', type=int, default=10000, help='number of training epoch')
    parser.add_argument('--lr_init', type=float, default=1e-5, help='init lr of the optimizer')
    parser.add_argument('--w_regul', type=float, default=0, help='loss regularizer to keep low energy')
    parser.add_argument('--gamma_scheduler', type=float, default=0, help='decay rate of the sceduler (0= no scheduler)')
    parser.add_argument("--step_scheduler", nargs='+', type=int, default=[50,],
                        help='step (in epoch number) to apply the scheduler decay rate')
    parser.add_argument('--grad_regul', type=float, default=0, help='regularization to supervised the gradient')
    parser.add_argument('--noise_grad_regul', type=float, default=0.1, help='regularization to supervised the gradient')

    ## SAVE args
    parser.add_argument('--save_root', type=str, default="/media/data_cifs/projects/prj_mental/model/LatentEBM/",
                        help='path to save the checkpoint')
    parser.add_argument('--model_name', type=str, default=None,
                        help='name of the model to save')

    ## EBM args
    parser.add_argument("--sigma", type=float, default=5e-2,
                        help="stddev of gaussian noise to add to input, .03 works but .1 is more stable")
    parser.add_argument('--sgld_std', type=float, default=1e-2, help='std in the mcmc sampling')
    parser.add_argument('--sgld_lr', type=float, default=1., help='learning rate of the mcmc sampling')
    parser.add_argument("--n_steps", type=int, default=50,
                        help="number of steps of SGLD per iteration, 100 works for short-run, 20 works for PCD")
    parser.add_argument('--reinit_freq', type=float, default=0.05, help='frequency of buffer reinitialization')
    parser.add_argument('--buffer_size', type=int, default=10000, help='size of the buffer')
    parser.add_argument('--init_type', type=str, default='uniform_[-1,1]', choices=['uniform_[-1,1]', 'uniform_[-2,2]', 'normal', 'uniform_[-6,6]', 'normal_01'])
    parser.add_argument('--energy_func', type=str, default='Res18_Quadratic',
                        choices=['VanillaNet_ELU_2'])
    parser.add_argument("--spec_norm", type=str2bool, nargs='?', const=True, default=False,
                        help="apply spectral normalization")
    parser.add_argument('--gradient_clip', type=float, default=0, help='gradient clippin value (0= no clipping)')
    parser.add_argument("--zero_init", type=str2bool, nargs='?', const=True, default=False,
                        help="zero_init the resnet")
    parser.add_argument("--w_last_gradient", type=float, default=0, help='value of the last_gradient regul')
    parser.add_argument("--dsm_weight", type=float, default=0.01, help='weight of the dsm')
    parser.add_argument("--multiplier", type=int, default=2, help='channel multiplier of the network')

    args = parser.parse_args()
    main(args)