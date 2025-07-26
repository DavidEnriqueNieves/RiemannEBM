import numpy as np
import torch

def Annealed_Langevin_E(netE, init_x, sigma, T_vect):
    # naive implementation of annealed Langevin sampling
    # T_vect is vector of temperatures, length equal to total sampling step
    # Sample_every is interval between saving a sample of X

    sigma2 = sigma ** 2

    T_vect = np.sqrt(T_vect)

    K = T_vect.shape[0]

    netE.eval()

    E_list = []

    x = init_x

    for i in range(K):
        x.requires_grad_()
        E_values = netE(x)
        E_list.append(E_values.squeeze().detach())
        grad_x = torch.autograd.grad(E_values.sum(), x, create_graph=False)[0]
        #E = E_values.sum()
        #E.backward()
        #x = x.detach() - 0.5 * sigma2 * x.grad + T_vect[i] * sigma * torch.randn_like(x)
        x = x.detach() - 0.5 * sigma2 * grad_x + T_vect[i] * sigma * torch.randn_like(x)

        #if (i + 1) % Sample_every == 0:
        #    x_list.append(x.detach().cpu())
        #    print('langevin step {}'.format(i + 1))

    netE.train()

    E_mtx = torch.stack(E_list, 1)
    # print('sampling finished')
    return x.detach(), E_mtx.cpu().numpy()

def Annealed_Langevin_JEM(netE, init_x, sigma, T_vect, Sample_every, cond=None):
    # naive implementation of annealed Langevin sampling
    # T_vect is vector of temperatures, length equal to total sampling step
    # Sample_every is interval between saving a sample of X

    sigma2 = sigma ** 2

    T_vect = np.sqrt(T_vect)

    K = T_vect.shape[0]

    netE.eval()

    E_list = []

    x = init_x

    x_list = []
    for i in range(K):
        x.requires_grad_()
        E_values = netE(x, cond).logsumexp(1)
        E_list.append(E_values.squeeze().detach())
        grad_x = torch.autograd.grad(E_values.sum(), x, create_graph=False)[0]
        #E = E_values.sum()
        #E.backward()
        #x = x.detach() - 0.5 * sigma2 * x.grad + T_vect[i] * sigma * torch.randn_like(x)
        x = x.detach() - 0.5 * sigma2 * grad_x + T_vect[i] * sigma * torch.randn_like(x)

        if (i + 1) % Sample_every == 0:
            x_list.append(x.detach().cpu())
            print('langevin step {}'.format(i + 1))

    netE.train()

    E_mtx = torch.stack(E_list, 1)
    # print('sampling finished')
    return x_list, E_mtx.cpu().numpy()

def init_random(size, init_type='uniform'):
    if init_type == "uniform_[-1,1]":
        buff = 2 * torch.rand(size) - 1
    elif init_type == "uniform_[-2,2]":
        buff = 4 * torch.rand(size) - 2
    elif init_type == "uniform_[-6,6]":
        buff = 12 * torch.rand(size) - 6
    elif init_type == "normal":
        buff = 2 * torch.randn(size)
    elif init_type == "normal_01":
        buff = torch.randn(size)
    elif init_type == "truncated_normal_01":
        buff = torch.clip(torch.randn(size), min=-1, max=1)
    else:
        raise NotImplementedError()
    return buff

    #return torch.FloatTensor(bs, 1, 32, 32).uniform_(-1, 1)

def Simple_Langevin(f, init_x, sgld_std, sgld_lr, n_steps):
    """this func takes in replay_buffer now so we have the option to sample from
    scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
    """
    f.eval()
    # get batch size
    x_k = torch.autograd.Variable(init_x, requires_grad=True)
    r_s_t = torch.zeros(1).to(x_k.device)
    # sgld
    for k in range(n_steps):
        energy = f(x_k)
        f_prime = torch.autograd.grad(
            outputs=energy,
            inputs=x_k,
            grad_outputs=torch.ones_like(energy),  # same shape as f
            #create_graph=True,
            retain_graph=True
        )[0]
        #f_prime = torch.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
        r_s_t += f_prime.view(f_prime.shape[0], -1).norm(dim=1).mean()
        x_k.data = x_k.data - sgld_lr * f_prime + sgld_std * torch.randn_like(x_k)
    f.train()
    final_samples = x_k.detach()

    return final_samples, r_s_t

def get_sample_q(args):
    def sample_p_0(replay_buffer, bs, y=None):
        if len(replay_buffer) == 0:
            return init_random((bs, *replay_buffer.size()[1:]), init_type = args.init_type).to(args.device), []
        buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // args.n_classes
        inds = torch.randint(0, buffer_size, (bs,))
        # if cond, convert inds to class conditional inds
        if y is not None:
            inds = y.cpu() * buffer_size + inds
            assert not args.uncond, "Can't drawn conditional samples without giving me y"
        buffer_samples = replay_buffer[inds]
        random_samples = init_random(buffer_samples.size(), init_type = args.init_type)

        #choose_random = (torch.rand(bs) < args.reinit_freq).float()[:, None, None, None]
        #choose_random = (torch.rand(bs) < args.reinit_freq).float()[:, None]
        choose_random = (torch.rand(bs) < args.reinit_freq).float().reshape(-1, *([1] * (buffer_samples.dim() - 1)))
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
        return samples.to(args.device), inds

    def sample_q(f, replay_buffer, y=None, n_steps=args.n_steps, clip=0, w_last_gradient=0, grad_norm_max=0, sgld_lr=args.sgld_lr):
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
        """
        f.eval()
        # get batch size
        bs = args.batch_size if y is None else y.size(0)
        # generate initial samples and buffer inds of those samples (if buffer is used)
        init_sample, buffer_inds = sample_p_0(replay_buffer, bs=bs, y=y)
        x_k = torch.autograd.Variable(init_sample, requires_grad=True)
        r_s_t = torch.zeros(1).to(x_k.device)
        # sgld
        for k in range(n_steps):
            energy = f(x_k)
            f_prime = torch.autograd.grad(
                outputs=energy,
                inputs=x_k,
                grad_outputs=torch.ones_like(energy),  # same shape as f
                #create_graph=True,
                retain_graph=True
            )[0]
            if clip != 0:
                f_prime = f_prime.clip(-clip, clip)
                #torch.nn.utils.clip_grad_norm_(, max_norm=1)
            grad_norm = f_prime.view(f_prime.shape[0], -1).norm(dim=1)
            if grad_norm.any() > 1000 or torch.isnan(grad_norm).any():
                print(f"in langevin at steps {k}")
            r_s_t += grad_norm.mean()

            if w_last_gradient != 0 and k == n_steps-1:
                x_k_ori = x_k
                x_k.data = x_k.data - sgld_lr * f_prime + args.sgld_std * torch.randn_like(x_k)
                x_last_grad = x_k_ori
                energy = f(x_last_grad)
                x_last_grad_prime = torch.autograd.grad(
                    outputs=energy,
                    inputs=x_last_grad,
                    grad_outputs=torch.ones_like(energy),  # same shape as f
                    create_graph=True,
                    retain_graph=True
                )[0]
                x_last_grad = x_last_grad - args.sgld_lr * x_last_grad_prime + args.sgld_std * torch.randn_like(
                    x_last_grad)
            else:

                x_k.data = x_k.data - args.sgld_lr * f_prime + args.sgld_std * torch.randn_like(x_k)
                if grad_norm_max != 0:
                    norm = x_k.data.view(f_prime.shape[0], -1).norm(dim=1)
                    ones = torch.ones_like(norm)*grad_norm_max
                    x_k.data = (x_k.data * grad_norm_max) / torch.maximum(norm, ones).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                x_k.data = torch.clip(x_k.data, min=-1, max=1)
                x_last_grad = None
        f.train()
        final_samples = x_k.detach()
        # update replay buffer
        if len(replay_buffer) > 0:
            replay_buffer[buffer_inds] = final_samples.cpu()
        return final_samples, r_s_t, replay_buffer, x_last_grad
    return sample_q

