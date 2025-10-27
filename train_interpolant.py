import torch
from utils.riemannian_metric import ConformalMetric, GradDiagonalMetric, DiagonalMetric, GradFullMetric
import os
# from utils.riemannian_metric import load_approximator
import argparse
from utils.dataset import get_dataloader, get_dataloader_ambiant
import torch.utils.data as data_utils
from utils.h_utils import InverseProb, LogProb
from model.rae import normalize
import matplotlib.pyplot as plt
from utils.riemannian_metric import load_metric, h_diag_RBF, h_diag_Land
from utils.monitoring import plot_img, make_grid, name_model_interp2, str2bool
from model.curvature_net import Curve_Net
import wandb as wandb
import numpy as np
import random
from model.Unet_velocity import GeoPathUNet as UNetModel


def main(args):
    with torch.no_grad():
        if args.model_name is None and not args.debug:
            args.model_name = name_model_interp2(args)

        if args.device == 'meso':
            args.device = torch.cuda.current_device()

        if args.dataset == 'afhq':
            ## load the dataset
            d_loader = get_dataloader_ambiant(data_root=args.data_root,
                                              dataset=args.dataset,
                                              image_size=128,
                                              device='cpu',
                                              ae_name="stable_diff_14",
                                              ambiant=True
                                              )


            def x_to_z(x, inverse=False):
                if inverse == False:
                    z = d_loader.to_latent(x)
                    z = torch.clip(z, max=2, min=-2)
                    return z / 2
                else:
                    x *= 2
                    x = d_loader.to_im(x.detach())
                    return x

            lt_cat = d_loader.data_train['latent'][d_loader.data_train['label'] == 0]
            lt_cat = torch.clip(lt_cat, max=2, min=-2) / 2
            lt_dog = d_loader.data_train['latent'][d_loader.data_train['label'] == 1]
            lt_dog = torch.clip(lt_dog, max=2, min=-2) / 2

            lt_size = lt_cat.shape[1:]
            im_size = [3, 128, 128]
            z_test_0 = lt_cat[[35, 49, 1465, 596, 421, 649]].unsqueeze(1).to(args.device)
            z_test_1 = lt_dog[[482, 395, 2485, 3100, 1453, 19]].unsqueeze(1).to(args.device)
            z_test_0 = z_test_0.repeat(1, args.t_steps, *[1 for _ in lt_size])
            z_test_1 = z_test_1.repeat(1, args.t_steps, *[1 for _ in lt_size])
            t_test = torch.linspace(0, 1, args.t_steps).view(1, args.t_steps, *[1 for _ in lt_size]).to(
                args.device).detach()
            t_test = t_test.repeat(z_test_0.size(0), 1, *[1 for _ in lt_size])

            args_curnet = {"geopath_model": True,
                           "dim": lt_size,
                           "num_channels": args.num_channels,
                           "num_res_blocks": 2,
                           "channel_mult": [1, 2, 2],
                           "dropout": 0.0,
                           }
            curvature_net = UNetModel(
                **args_curnet
            ).to(args.device)



        else:
            raise NotImplementedError()

        # print(args.metric)
        substrings = args.metric.split("_")
        metric_type = substrings[0]
        approximator = substrings[1]
        h_function = substrings[2]

        # if args.metric_path is None:
        #    path_approx = os.path.join(args.approx_root, args.approx_ckpt)
        args_normalization = {"min": args.min_h, "max": args.max_h}
        if approximator == 'ebm':
            print("load ebm")
            path_to_param = os.path.join(args.ebm_root, args.ebm_ckpt)
            ebm_param = torch.load(path_to_param, weights_only=False)
            netE = ebm_param["type"](**ebm_param["args"]).to(args.device)
            netE.load_state_dict(ebm_param["weight"])
            netE.eval()

            if h_function == "invp":
                h = InverseProb(ebm=netE, metric_type=metric_type, multiplier=args.ebm_multiplier).to(args.device)
                h.normalize(None, **args_normalization)

            elif h_function == 'logp':
                h = LogProb(ebm=netE, metric_type=metric_type, multiplier=args.ebm_multiplier).to(args.device)
                torch.manual_seed(1)
                np.random.seed(1)
                random.seed(1)
            else:
                raise NotImplementedError()
        elif approximator == 'rbf':
            args_rbf = {"n_centers": args.rbf_center,
                        "latent_size": lt_size,
                        "ambiant_size": im_size,
                        "kappa": args.rbf_kappa}
            torch.manual_seed(1)
            np.random.seed(1)
            random.seed(1)
            flt = torch.logical_or(d_loader.data_train['label'] == 0, d_loader.data_train['label'] == 1)
            idx_f = torch.randint(low=0, high=flt.sum(), size=(5000,))
            data_to_fit_latent = d_loader.data_train['latent'][flt][idx_f].view(-1,
                                                                                np.prod(lt_size))  # .to(args.device)
            data_to_fit_ambiant = d_loader.data_train_ambiant["ambiant"][flt][idx_f].view(-1, np.prod(
                im_size))  # .to(args.device)

            h = h_diag_RBF(**args_rbf,
                           data_to_fit_ambiant=data_to_fit_ambiant,
                           data_to_fit_latent=data_to_fit_latent,
                           ).to(args.device)  ## kappa=5 semms to work
            h.normalize(data_to_fit_latent.to(args.device))

            del data_to_fit_ambiant
            del data_to_fit_latent

            a = 1
        elif approximator == 'land':
            flt = torch.logical_or(d_loader.data_train['label'] == 0, d_loader.data_train['label'] == 1)
            idx_f = torch.randint(low=0, high=flt.sum(), size=(1000,))
            data_to_fit_latent = d_loader.data_train['latent'][flt][idx_f].view(-1,
                                                                                np.prod(lt_size)).to(args.device)
            h = h_diag_Land(reference_sample=data_to_fit_latent, gamma=args.gamma_land).to(
                args.device)  ##O.6 works well


        else:
            raise NotImplementedError()
        print(f"{args.metric} -- min={args.min_h} -- max={args.max_h}")
        if args.model_name is not None:
            path_to_save = os.path.join(args.save_root, args.dataset, args.model_name)
            print(path_to_save)
            os.makedirs(path_to_save, exist_ok=True)
            wandb.init(project='Net_interp_' + args.dataset, config=vars(args), entity='victorboutin')
            wandb.run.name = args.model_name
            torch.save(args, path_to_save + '/param.config')

    metric = load_metric(metric_type, approximator, h)

    dt = torch.tensor(1.0 / (args.t_steps - 1))
    optimizer = torch.optim.Adam(curvature_net.parameters(), lr=args.lr)
    all_loss = []

    t_ = torch.linspace(0, 1, args.t_steps).view(1, args.t_steps, *[1 for _ in lt_size]).to(args.device).detach()
    reg_space = torch.linspace(0, args.t_steps - 1, 15).long()
    for it in range(args.nb_iteration):
        idx_image_cat = torch.randint(low=0, high=lt_cat.size(0), size=(args.batch_size,))
        idx_image_dog = torch.randint(low=0, high=lt_dog.size(0), size=(args.batch_size,))

        z0 = lt_cat[idx_image_cat].unsqueeze(1).to(args.device).view(-1, 1, *lt_size)
        z1 = lt_dog[idx_image_dog].unsqueeze(1).to(args.device).view(-1, 1, *lt_size)

        normalizing_set = torch.clip((1 - t_) * z0 + t_ * z1, max=2, min=-2)

        metric.h.normalize2(normalizing_set, **args_normalization)

        z0 = z0.repeat(1, args.t_steps, *[1 for _ in lt_size]).to(args.device).detach()
        z1 = z1.repeat(1, args.t_steps, *[1 for _ in lt_size]).to(args.device).detach()
        t = t_.repeat(z0.size(0), 1, *[1 for _ in lt_size])  # .expand_as(z0).detach()
        c_t = curvature_net(z0.reshape(-1, *lt_size), z1.reshape(-1, *lt_size), t.view(-1))
        z_t = (1 - t) * z0 + t * z1 + 2 * t * (1 - t) * c_t.view(args.batch_size, args.t_steps, *lt_size)
        z_t = torch.clip(z_t, max=2, min=-2)
        z_t_dot = (z_t[:, 1:] - z_t[:, :-1]) / dt
        energy = metric.kinetic(x_t=z_t[:, :-1], x_t_dot=z_t_dot)
        energy = energy.view(args.batch_size, args.t_steps - 1)
        kinetic_energy = (energy * dt).sum(dim=1).mean()
        loss = kinetic_energy
        loss.backward()
        all_param = 0.0
        for param in curvature_net.parameters():
            all_param += param.grad.norm()
        optimizer.step()
        optimizer.zero_grad()
        all_loss.append(loss.item())

        with torch.no_grad():
            if it % 10 == 0:
                if args.model_name is not None:
                    dico_log = {"train/kinectic_energy": loss.item()}
                    wandb.log(dico_log)

            # if it % 00 == 0:
            # if it % 100 == 0:
            if it % 10000 == 0:
                print(
                    f"{it + 1}/{args.nb_iteration} -- kinectic {kinetic_energy.item():0.3f}  --grad {all_param.item():0.3f}")
                fig, ax = plt.subplots(1, 2, figsize=(2 * 3, 3), dpi=100)
                ax[0].plot(all_loss)
                ax[0].set_title('loss (kinectic)')
                ax[0].set_xlabel('epoch')
                ax[0].set_ylabel('loss')
                ax[1].plot(energy.view(-1, args.t_steps - 1).mean(dim=0).detach().cpu())
                ax[1].set_xlabel('trajectory')
                ax[1].set_ylabel('kinetic')
                if args.model_name is not None:
                    wandb.log({'plot': wandb.Image(fig)})
                else:
                    plt.show()

                c_t_test = curvature_net(z_test_0.reshape(-1, *lt_size), z_test_1.reshape(-1, *lt_size),
                                         t_test.view(-1))
                z_t_test = (1 - t_test) * z_test_0 + t_test * z_test_1 + 2 * t_test * (1 - t_test) * c_t_test.view(-1,
                                                                                                                   args.t_steps,
                                                                                                                   *lt_size)
                z_t_test = torch.clip(z_t_test, max=2, min=-2)
                z_t_test = z_t_test[:, reg_space].view(-1, *lt_size)
                x_t = x_to_z(z_t_test, inverse=True)

                if args.model_name is not None:
                    to_plot = wandb.Image(
                        make_grid(x_t.cpu().detach(), nrow=z_test_0.size(0), ncol=len(reg_space), normalize=True,
                                  scale_each=True),
                        caption='ep:{}'.format(it))

                    wandb.log({'test/geodesic': to_plot})
                else:
                    plot_img(x_t, nrow=z_test_0.size(0), ncol=len(reg_space))

            if (it % 10000 == 0) and (args.model_name is not None):
                curvature_net.cpu()
                to_save = {"weight": curvature_net.state_dict(),
                           "type": type(curvature_net),
                           "args": args_curnet
                           }
                torch.save(to_save, path_to_save + f'/ep_{it}.model')
                curvature_net.to(args.device)

    if args.model_name is not None:
        curvature_net.cpu()
        to_save = {"weight": curvature_net.state_dict(),
                   "type": type(curvature_net),
                   "args": args_curnet
                   }
        torch.save(to_save, path_to_save + f'/last.model')
        curvature_net.to(args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Metric Center")

    ## dataset
    parser.add_argument('--dataset', type=str, choices=['afhq'])
    parser.add_argument("--data_root", type=str, default="/media/data_cifs_lrs/projects/prj_mental/datasets")
    parser.add_argument('--rot_dist', type=str, default='uniform', choices=['uniform', 'gaussian'])
    parser.add_argument('--letter_dataset', type=str, default=None, choices=['P'])
    parser.add_argument('--save_root', type=str, default="/media/data_cifs/projects/prj_mental/model/CurvatureNet/",
                        help='path to save the checkpoint')

    ## interpolation
    parser.add_argument('--t_steps', type=int, default=50)
    parser.add_argument('--nb_iteration', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=1e-3)

    ## metric
    parser.add_argument('--metric', type=str,
                        choices=[
                            'conf_ebm_invp',
                            'conf_ebm_logp',
                            'diag_ebm_logp',
                            'diag_ebm_invp',
                            # 'diag_rbf_invp',
                            'conf_rbf_invp',
                            'diag_land_invp',
                            'full_ebm_invp',
                            'full_ebm_logp',
                        ])
    parser.add_argument('--min_h', type=float, default=0.0)
    parser.add_argument('--max_h', type=float, default=1e3)
    parser.add_argument('--ebm_multiplier', type=int, default=None)
    parser.add_argument('--gamma_land', type=float, default=None)
    ## EBM args
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')
    parser.add_argument('--ebm_root', type=str, default='/media/data_cifs/projects/prj_mental/AFHQ_CCV/')
    parser.add_argument('--ebm_ckpt', type=str,
                        default='CCV_20250316_135439_cdsm_VanillaNet_ELU_2_x3_w_dsm0.01_Stp50_SgldLr=1.0_LR1.0e-05_Wr0.0e+00/ep_900.model')

    ## training args
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--model_name', type=str, default=None,
                        help='name of the model to save')
    parser.add_argument('--num_channels', type=int, default=64)
    parser.add_argument('--debug', type=str2bool, nargs='?', const=True, default=False,
                        help='debug mode (no log)')

    parser.add_argument('--rbf_center', type=int, default=None, help='number of center of the rbf')
    parser.add_argument('--rbf_kappa', type=float, default=None, help='kappa parameter of the rbf')

    args = parser.parse_args()
    main(args)