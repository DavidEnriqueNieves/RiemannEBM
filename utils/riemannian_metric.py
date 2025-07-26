import torch.nn as nn
import torch
import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class RiemannianMetric(nn.Module):
    def __init__(self, h, euclid_weight=0):
        super().__init__()
        self.h = h
        self.euclid_weight = euclid_weight

    def monitor_g(self, x_t):
        pass

    def g_fast(self, x_t):
        pass

    def kinetic(self, x_t, x_t_dot):
        pass


class DiagonalMetric(RiemannianMetric):
    def __init__(self, h, euclid_weight=0):
        super().__init__(h, euclid_weight=euclid_weight)

    def monitor_g(self, x_t):
        return self.g_fast(x_t)

    def g_fast(self, x_t):
        return self.euclid_weight + self.h(x_t)

    def kinetic(self, x_t, x_t_dot):
        self.nb_samples, self.nb_interp, *self.feature_size = x_t.shape
        g = self.g_fast(x_t)
        x_t_dot = x_t_dot.view(-1, np.prod(self.feature_size))
        kinectic = torch.einsum('bi,bi->b', x_t_dot, g * x_t_dot)
        return kinectic


class GradDiagonalMetric(DiagonalMetric):
    def __init__(self, h, euclid_weight=1):
        super().__init__(h, euclid_weight=euclid_weight)

    def monitor_g(self, x_t):
        return self.g_fast(x_t)

    def g_fast(self, x_t):
        return self.euclid_weight + self.compute_grad(x_t).abs()

    def compute_grad(self, x_t):
        x_t.requires_grad_(True)
        h_t = self.h(x_t)
        grad = torch.autograd.grad(h_t.sum(), x_t, create_graph=True)[0]
        return grad


class ConformalMetric(RiemannianMetric):
    def __init__(self, h, euclid_weight=0):
        super().__init__(h, euclid_weight=euclid_weight)

    def monitor_g(self, x_t):
        return self.g_fast(x_t)

    def g_fast(self, x_t):
        return self.euclid_weight + self.h(x_t)

    def kinetic(self, x_t, x_t_dot):
        self.nb_samples, self.nb_interp, *self.feature_size = x_t.shape
        g = self.g_fast(x_t)
        x_t_dot = x_t_dot.view(-1, np.prod(self.feature_size))
        kinectic = g*(x_t_dot.pow(2).sum(dim=-1))
        return kinectic


class GradFullMetric(RiemannianMetric):
    def __init__(self, h, euclid_weight=1):
        super().__init__(h, euclid_weight=euclid_weight)

    def compute_grad(self, x_t):
        x_t.requires_grad_(True)
        h_t = self.h(x_t)
        grad = torch.autograd.grad(h_t.sum(), x_t, create_graph=True)[0]
        return grad

    def monitor_g(self, x_t):
        b, d = x_t.shape
        grad = self.compute_grad(x_t)
        outer = torch.einsum('bi,bj->bij', grad, grad)  # Shape: (B, D, D)
        I = torch.eye(d, device=x_t.device).expand(b, d, d)  # Shape: (B, D, D)
        return (I * self.euclid_weight) + outer

    def g_fast(self, x_t):
        ## Here this not g, but the gradient (faster not to compute the outer product)
        return self.compute_grad(x_t)

    def kinetic(self, x_t, x_t_dot):
        grad = self.g_fast(x_t)
        euclid_kinetic = x_t_dot.pow(2).sum(dim=-1)
        riem_kinetic = torch.einsum('bi, bi-> b', x_t_dot, grad) ** 2
        return self.euclid_weight * euclid_kinetic + riem_kinetic

"""
def load_approximator(approximator, path_approx, latent_sample, ambiant_sample=None, args_rbf=None):
    if approximator == 'ebm':
        print("load ebm")
        path_to_param = os.path.join(path_approx)
        ebm_param = torch.load(path_to_param, weights_only=False)
        netE = ebm_param["type"](**ebm_param["args"])
        netE.load_state_dict(ebm_param["weight"])
        netE.eval()
        p_ebm = lambda x: torch.exp(-netE(x))
        normalizer = torch.exp(-netE(latent_sample)).mean().detach().item()
        to_return = NormalizedApproximator(h=p_ebm, normalizer=normalizer)

    elif approximator == "rbf":
        assert ambiant_sample is not None, "need ambiant sample to assess the cluster of the rbf"
        assert args_rbf is not None, "need args to set up the rbf"
        rbf_approx = h_diag_RBF(**args_rbf, data_to_fit_ambiant=ambiant_sample, data_to_fit_latent=latent_sample, kappa=1)
        normalizer = 1
        to_return = NormalizedApproximator(h=rbf_approx, normalizer=normalizer)
    elif approximator == "land":
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    return to_return
"""
def load_metric(metric_type, approximator, h):
    if metric_type == "conf": ## conformal riemannian metric
        assert approximator in ["ebm", "rbf"]
        metric = ConformalMetric(h)

    elif metric_type == "diag":  ## diagonal riemannian metric
        if approximator in ["land", "rbf"]:
            metric = DiagonalMetric(h)
        else:
            metric = GradDiagonalMetric(h)

    elif metric_type == "full":  ## full riemannian metric
        assert approximator == "ebm"
        metric = GradFullMetric(h)
    else:
        raise NotImplementedError()
    return metric

class NormalizedApproximator(nn.Module):
    def __init__(self, h, normalizer):
        super().__init__()
        self.h = h
        self.register_buffer('normalizer', normalizer)

    def forward(self, x_t):
        return self.h(x_t)/self.normalizer


class h_diag_RBF(nn.Module):
    def __init__(self, n_centers, latent_size, ambiant_size, data_to_fit_ambiant=None, data_to_fit_latent=None, kappa=1):
        super().__init__()
        self.K = n_centers
        self.latent_size = latent_size
        self.ambiant_size = ambiant_size
        self.latent_size_flat = np.prod(latent_size)
        self.ambiant_size_flat = np.prod(ambiant_size)
        #self.data_size = np.prod(data_size)
        self.kappa = kappa
        #self.register_buffer('W', torch.rand(self.K, self.latent_size_flat))
        self.register_buffer('W', torch.rand(self.K, 1))

        #sigmas = np.ones((self.K, self.latent_size_flat))
        sigmas = np.ones((self.K, 1))
        data_to_fit_latent = data_to_fit_latent.view(-1, self.latent_size_flat)
        data_to_fit_ambiant = data_to_fit_ambiant.view(-1, self.ambiant_size_flat)

        if (data_to_fit_ambiant is not None) and (data_to_fit_latent is not None):
            data_to_fit_a = data_to_fit_ambiant.cpu().detach().numpy()
            data_to_fit_l = data_to_fit_latent.cpu().detach().numpy()
            print("fitting")
            clustering_model = KMeans(n_clusters=self.K)
            clustering_model.fit(data_to_fit_a)
            clusters = self.calculate_centroids(data_to_fit_l, clustering_model.labels_)
            #clusters = clustering_model.cluster_centers_
            self.register_buffer('C', torch.tensor(clusters, dtype=torch.float32))#.to(data_to_fit_latent.device))
            labels = clustering_model.labels_
            for k in range(self.K):
                points = data_to_fit_l[labels == k]
                variance = ((points - clusters[k]) ** 2).mean(axis=0)
                #variance = ((points - self.C[k]) ** 2).mean(axis=0)
                # print('variance', variance.shape)
                #sigmas[k, :] = np.sqrt(variance) + 1e-3
                sigmas[k, :] = np.sqrt(variance.sum()) + 1e-5 # + 1e-3#.sum()# + 1e-3
            del data_to_fit_ambiant
            del data_to_fit_latent
            del clustering_model
        else:
            self.register_buffer('C', torch.zeros(self.K, self.data_size))
        lbda = torch.tensor(0.5 / (self.kappa * sigmas) ** 2, dtype=torch.float32)#.to(data_to_fit_latent.device)
        self.register_buffer('lamda', lbda)

        a=1

    def calculate_centroids(self, all_data, labels):
        unique_labels = np.unique(labels)
        centroids = np.zeros((len(unique_labels), all_data.shape[1]))
        for i, label in enumerate(unique_labels):
            centroids[i] = all_data[labels == label].mean(axis=0)
        return centroids

    def normalize2(self, x, min, max):
        pass

    def forward(self, x_t):
        if len(x_t.shape) > 2:
            self.nb_samples, self.nb_interp, *self.feature_size = x_t.shape
            x_t = x_t.reshape(-1, np.prod(self.feature_size))
        dist2 = torch.cdist(x_t, self.C) ** 2
        phi_x = torch.exp(-0.5 * self.lamda[None, :, :] * dist2[:, :, None])
        #phi_x = torch.exp(-0.5 * dist2[:, :, None])
        h_x = (self.W.unsqueeze(0)*phi_x).sum(dim=1)
        #te = h_x.view(8, 50)
        #plt.plot(te.t().cpu().detach())
        #plt.show()
        return 1/(h_x.squeeze() + 1e-3)

    def forward_training(self, x_t):
        if len(x_t.shape) > 2:
            self.nb_samples, self.nb_interp, *self.feature_size = x_t.shape
            x_t = x_t.reshape(-1, np.prod(self.feature_size))
        dist2 = torch.cdist(x_t, self.C) ** 2
        phi_x = torch.exp(-0.5 * self.lamda[None, :, :] * dist2[:, :, None]).detach()

        #phi_x = torch.exp(-0.5 * dist2[:, :, None]).detach()
        h_x = (self.W.unsqueeze(0)*phi_x).sum(dim=1)
        return h_x

    def normalize(self, data_to_train):
        with torch.enable_grad():
            self.W.requires_grad_(True)
            optimizer = torch.optim.Adam([self.W], lr=1e-3)
            for i in range(30000):
                idx_z = torch.randint(low=0, high=len(data_to_train), size=(128,))
                z = data_to_train[idx_z].detach()
                loss = ((1 - self.forward_training(z)) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    if i % 100 == 0:
                        print(f"loss : {loss.item():0.3f} -- param {self.W.sum().item():0.3f}")
        self.register_buffer('W', self.W.data)

class h_diag_Land(nn.Module):
    def __init__(self, reference_sample, gamma=0.2):
        super().__init__()
        self.reference_sample = reference_sample.detach()
        self.gamma = gamma
        self.register_buffer('W', torch.ones(1,1))

        #self.register_buffer('W', torch.ones(1.0))

    def weighting_function(self, x):
        pairwise_sq_diff = (x[:, None, :] - self.reference_sample[None, :, :].detach()) ** 2
        pairwise_sq_dist = pairwise_sq_diff.sum(-1)
        weights = torch.exp(-pairwise_sq_dist / (2 * self.gamma ** 2))
        #weights = torch.exp(-pairwise_sq_dist)
        return weights

    def normalize2(self, x, min, max):
        pass

    def forward(self, x_t):
        if len(x_t.shape) > 2:
            self.nb_samples, self.nb_interp, *self.feature_size = x_t.shape
            x_t = x_t.reshape(-1, np.prod(self.feature_size))

        weights = self.weighting_function(x_t)  # Shape [B, N]
        differences = self.reference_sample[None, :, :].detach() - x_t[:, None, :]  # Shape [B, N, D]
        squared_differences = differences ** 2  # Shape [B, N, D]

        # Compute the sum of weighted squared differences for each dimension
        M_dd_diag = torch.einsum("bn,bnd->bd", weights, squared_differences)
        #plt.plot(M_dd_diag.mean(dim=-1).view(8,50).t().cpu().detach())
        #plt.show()
        return 1 / (self.W*M_dd_diag + 1e-3)

    def forward_training(self, x_t):
        if len(x_t.shape) > 2:
            self.nb_samples, self.nb_interp, *self.feature_size = x_t.shape
            x_t = x_t.reshape(-1, *self.feature_size)
            #x_t = x_t.reshape(x_t.shape[0], -1)
        weights = self.weighting_function(x_t)  # Shape [B, N]
        differences = self.reference_sample[None, :, :] - x_t[:, None, :]  # Shape [B, N, D]
        squared_differences = differences ** 2  # Shape [B, N, D]

        # Compute the sum of weighted squared differences for each dimension
        M_dd_diag = torch.einsum("bn,bnd->bd", weights, squared_differences)
        return self.W*M_dd_diag

    def normalize(self, data_to_train):
        with torch.enable_grad():
            self.W.requires_grad_(True)
            optimizer = torch.optim.Adam([self.W], lr=1e-1)
            for i in range(30000):
                idx_z = torch.randint(low=0, high=len(data_to_train), size=(128,))
                z = data_to_train[idx_z].detach()
                loss = ((1 - self.forward_training(z).mean(dim=1)) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    if i % 100 == 0:
                        print(f"loss : {loss.item():0.3f} -- param {self.W.sum().item():0.3f}")
        self.register_buffer('W', self.W.data)