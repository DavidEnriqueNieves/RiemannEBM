import torch.nn as nn
import torch
import matplotlib.pyplot as plt

def linear_normalization(maxi, mini, target_max=1, target_min=1):
    target_max = torch.tensor(target_max).expand_as(maxi).to(maxi.device)
    target_min = torch.tensor(target_min).expand_as(mini).to(mini.device)
    alpha = (target_max - target_min)/(maxi - mini)
    beta = target_min - alpha*mini
    return alpha, beta

class InverseProb(nn.Module):
    def __init__(self, ebm, metric_type, multiplier=1):
        super().__init__()
        self.ebm = ebm
        self.register_buffer('alpha_n', torch.tensor(1.0))
        self.register_buffer('beta_n', torch.tensor(0.0))
        #self.register_buffer('max_iid', torch.tensor(1.0))
        self.register_buffer('multiplier', torch.tensor(multiplier))
        self.metric_type = metric_type

    def scaled_ebm(self, x):
        return self.multiplier*(self.ebm(x))

    def global_normalization(self, normalizing_set, min, max):
        with torch.no_grad():
            self.nb_samples, self.nb_interp, *self.feature_size = normalizing_set.shape
            en = self.scaled_ebm(normalizing_set.view(-1, *self.feature_size)).detach()
            p = torch.exp(-en)
            max_p = p.view(self.nb_samples, self.nb_interp).max(dim=1)[0].mean()
            min_p = p.view(self.nb_samples, self.nb_interp).min(dim=1)[0].mean()
            #max_p = p.max()
            #min_p = p.min()
            #min_p = p.view(self.nb_samples, self.nb_interp).min()[0]
            alpha_n, beta_n = linear_normalization(max_p, min_p, max, min)
            self.alpha_n = alpha_n.view(1, 1)
            self.beta_n = beta_n.view(1, 1)
            #te =
            #plt.plot()


    def normalize2(self, normalizing_set, min, max):
        pass


        #te = 1/(self.alpha_n*torch.exp(-en).view(self.nb_samples, self.nb_interp) + self.beta_n)
        #a=1

    """
    def normalize2(self, normalizing_set, min, max):
        with torch.no_grad():
            self.nb_samples, self.nb_interp, *self.feature_size = normalizing_set.shape
            en = self.scaled_ebm(normalizing_set.view(-1, *self.feature_size)).detach()
            p = torch.exp(-en)
            max_p = p.view(self.nb_samples, self.nb_interp).max(dim=1)[0]
            min_p = p.view(self.nb_samples, self.nb_interp).min(dim=1)[0]
            alpha_n, beta_n = linear_normalization(max_p, min_p, max, min)
            self.alpha_n = alpha_n.unsqueeze(1)
            self.beta_n = beta_n.unsqueeze(1)

        #te = 1/(self.alpha_n*torch.exp(-en).view(self.nb_samples, self.nb_interp) + self.beta_n)
        #a=1

    """
    def normalize(self, normalization_set, min, max):
        self.min = min
        self.max = max
    #def forward(self, x_t): ## works for alphanum
    #    self.nb_samples, self.nb_interp, *self.feature_size = x_t.shape
    #    en = 100 + self.scaled_ebm(x_t.reshape(-1, *self.feature_size))
    #    invp = 1 / (torch.exp(-en).view(self.nb_samples, self.nb_interp) + 1e-4)
    #    return invp.flatten()

    def forward(self, x_t): ## works for afhq
        self.nb_samples, self.nb_interp, *self.feature_size = x_t.shape
        en = self.multiplier*self.ebm(x_t.reshape(-1, *self.feature_size)) + 1
        #en = 10 * self.ebm(x_t.reshape(-1, *self.feature_size)) + 1
        #en = 100 + self.scaled_ebm(x_t.reshape(-1, *self.feature_size))
        invp = 1 / (torch.exp(-en) + self.min)
        return invp.flatten()
    """
    def forward(self, x_t):
        self.nb_samples, self.nb_interp, *self.feature_size = x_t.shape
        en = self.scaled_ebm(x_t.reshape(-1, *self.feature_size))
        invp = 1/(self.alpha_n*torch.exp(-en).view(self.nb_samples, self.nb_interp) + self.beta_n)
        return invp.flatten()
    """
    """
    def normalize(self, ood_set, iid_set, min, max):
        if self.metric_type == "conf":
            max_iid = self.ebm(iid_set).max().detach()
            #max_iid = self.ebm(iid_set).max().detach()
            self.max_iid = max_iid
            self.beta_n = torch.tensor(min)
        elif self.metric_type == "diag":
            self.ebm(iid_set).mean()
        #elif self.metric_type == 'diag':
            #max_iid = self.ebm(iid_set).mean().detach()
            #self.max_iid = max_iid
            #self.beta_n = torch.tensor(min)
        #max_iid = self.ebm(iid_set).max().detach()

        #ebm_min = torch.maximum(torch.exp(-1*self.ebm(ood_set)).min().detach(), torch.tensor(1/max)) ## probability is minimal in ood region
        #ebm_max = torch.exp(-1*self.ebm(iid_set)).max().detach()  ## probability is maximal in iid region
        #mean = torch.exp(-1*self.ebm(iid_set)).mean().detach()

        #alpha_n, beta_n = linear_normalization(ebm_max, ebm_min, target_max=max, target_min=min)
        #self.alpha_n = alpha_n
        #self.beta_n = beta_n
        #self.alpha_n = torch.tensor(2)#(1/max)*(1/mean)
        #self.beta_n = torch.tensor(min).to(iid_set.device)

    def forward(self, x_t):
        if self.metric_type == "conf":
            #normalized_energy = (torch.clip(self.ebm(x_t), min=self.max_iid) - self.max_iid)*100
            #normalized_energy = (torch.clip(self.ebm(x_t), min=self.max_iid) - self.max_iid) * 100
            #normalized_energy = 100*(self.ebm(x_t))# - self.max_iid)
            normalized_energy = self.ebm(x_t)*200 # - self.max_iid) %% 200 works fine
            #return 1/(self.alpha_n * torch.exp(-normalized_energy) + self.beta_n)
            return 1 / (torch.exp(-normalized_energy) + 1e-4)
        elif self.metric_type in ["diag", "full"]:
            normalized_energy = self.ebm(x_t)#(torch.clip(self.ebm(x_t), min=self.max_iid) - self.max_iid)
            return 1/(torch.exp(-normalized_energy) + 1e-3)
    """

class LogProb(nn.Module):
    def __init__(self, ebm, metric_type, multiplier=1):
        super().__init__()
        self.ebm = ebm
        self.register_buffer('alpha_n', torch.tensor(1.0))
        self.register_buffer('beta_n', torch.tensor(0.0))
        #self.register_buffer('max_iid', torch.tensor(1.0))
        self.register_buffer('multiplier', torch.tensor(multiplier))
        self.metric_type = metric_type

    def scaled_ebm(self, x):
        return self.multiplier*self.ebm(x)

    ### works for alphanum
    """
    def normalize2(self, normalizing_set, min, max):
        with torch.no_grad():
            self.nb_samples, self.nb_interp, *self.feature_size = normalizing_set.shape
            en = self.scaled_ebm(normalizing_set.view(-1, *self.feature_size)).detach()
            #p = torch.exp(-en)
            max_en = en.view(self.nb_samples, self.nb_interp).max(dim=1)[0]
            min_en = en.view(self.nb_samples, self.nb_interp).min(dim=1)[0]
            alpha_n, beta_n = linear_normalization(max_en, min_en, max, min)
            self.alpha_n = alpha_n.unsqueeze(1)
            self.beta_n = beta_n.unsqueeze(1)
    """

    def normalize2(self, normalizing_set, min, max):
        pass

    ### works for alphanum
    """
    def forward(self, x_t):
        self.nb_samples, self.nb_interp, *self.feature_size = x_t.shape
        en = self.scaled_ebm(x_t.reshape(-1, *self.feature_size))
        normalized_en = 1 + (self.alpha_n * en.view(self.nb_samples, self.nb_interp) + self.beta_n).clamp(-0.5)
        return normalized_en.flatten()
    """
    def forward(self, x_t):
        self.nb_samples, self.nb_interp, *self.feature_size = x_t.shape
        en = self.ebm(x_t.reshape(-1, *self.feature_size))
        normalized_en = torch.log(1 + torch.exp(self.multiplier*(en.view(self.nb_samples, self.nb_interp))))
        #normalized_en = (1 + (10 * en.view(self.nb_samples, self.nb_interp))).clamp(-2)
        return normalized_en.flatten()

    """
    def normalize(self, ood_set, iid_set, min, max):
        if self.metric_type == 'conf':
            max_iid = self.ebm(iid_set).mean().detach()
            self.max_iid = max_iid
            self.beta_n = torch.tensor(min)

        #mean = self.ebm(iid_set).mean()
        #mean = self.ebm(iid_set).min()
        #self.alpha_n = min*(1/mean)
        #self.beta_n = torch.tensor(max).to(iid_set.device)


    def forward(self, x_t):
        if self.metric_type == 'conf':
            #mean = self.ebm(x_t).min()
            #normalized_energy = 1 + (torch.clip(self.ebm(x_t), min=self.max_iid) - self.max_iid) * 100
            normalized_energy = 1 + 1000*((self.ebm(x_t) - self.max_iid)**3)
            #normalized_energy = 1 + 1000 * self.ebm(x_t)
            #normalized_energy = 1 + 1000 * (6*torch.log(1+3*(self.ebm(x_t))))

            return normalized_energy
        #return (self.alpha_n * self.ebm(x_t)).clamp(max=self.beta_n.item())
    """