import sys
sys.path.append("..")

import torch.nn as nn
import torch
import torch.nn.functional as F


class Leaky_softplus(nn.Module):
    def __init__(self, leak=0.05):
        super(Leaky_softplus, self).__init__()
        self.sofplus = nn.Softplus()
        self.leak = leak

    def forward(self, x):
        return self.leak * x + (1 - self.leak) * self.sofplus(x)

class VanillaNet(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.05, cond=False):
        super(VanillaNet, self).__init__()
        self.cond = cond
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),  # n_f, 32, 32
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),  # n_f, 16, 16
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f*2, n_f*4, 4, 2, 1),  # n_f, 8, 8
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f*4, n_f*8, 4, 2, 1),   # n_f, 4, 4
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f*8, n_f*8, 2, 1, 0),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f*8, 10, 2, 1, 0))

    def forward(self, x, y=None, guidance=None):
        if guidance is not None:
            x = torch.cat([x, guidance], dim=1)
        out_feat = self.f(x).view(x.size(0), -1)
        if self.cond:
            assert y is not None
            out_feat = torch.gather(out_feat, 1, y[:, None]).squeeze()
        return out_feat

class VanillaNet_smoothLR(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.05, cond=False):
        super(VanillaNet_smoothLR, self).__init__()
        self.cond = cond
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),  # n_f, 32, 32
            Leaky_softplus(leak),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),  # n_f, 16, 16
            Leaky_softplus(leak),
            nn.Conv2d(n_f*2, n_f*4, 4, 2, 1),  # n_f, 8, 8
            Leaky_softplus(leak),
            nn.Conv2d(n_f*4, n_f*8, 4, 2, 1),   # n_f, 4, 4
            Leaky_softplus(leak),
            nn.Conv2d(n_f*8, n_f*8, 2, 1, 0),
            Leaky_softplus(leak),
            nn.Conv2d(n_f*8, 10, 2, 1, 0))

    def forward(self, x, y=None, guidance=None):
        if guidance is not None:
            x = torch.cat([x, guidance], dim=1)
        out_feat = self.f(x).view(x.size(0), -1)
        if self.cond:
            assert y is not None
            out_feat = torch.gather(out_feat, 1, y[:, None]).squeeze()
        return out_feat

class VanillaNet_ELU(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.05, cond=False):
        super(VanillaNet_ELU, self).__init__()
        self.cond = cond
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),  # n_f, 32, 32
            nn.ELU(),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),  # n_f, 16, 16
            nn.ELU(),
            nn.Conv2d(n_f*2, n_f*4, 4, 2, 1),  # n_f, 8, 8
            nn.ELU(),
            nn.Conv2d(n_f*4, n_f*8, 4, 2, 1),   # n_f, 4, 4
            nn.ELU(),
            nn.Conv2d(n_f*8, n_f*8, 2, 1, 0),
            nn.ELU(),
            nn.Conv2d(n_f*8, 10, 2, 1, 0))

    def forward(self, x, y=None, guidance=None):
        if guidance is not None:
            x = torch.cat([x, guidance], dim=1)
        out_feat = self.f(x).view(x.size(0), -1)
        if self.cond:
            assert y is not None
            out_feat = torch.gather(out_feat, 1, y[:, None]).squeeze()
        return out_feat

class Square(nn.Module):
    def __init__(self):
        super(Square, self).__init__()
        pass

    def forward(self, in_vect):
        return in_vect ** 2

class VanillaNet_ELU_2(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.05, cond=False):
        super(VanillaNet_ELU_2, self).__init__()
        self.cond = cond
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),  # n_f, 16, 16
            #nn.ELU(),
            #nn.LeakyReLU(0.05),
            nn.SiLU(),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),  # n_f, 8, 8
            #nn.ELU(),
            #nn.LeakyReLU(0.05),
            nn.SiLU(),
            nn.Conv2d(n_f*2, n_f*4, 4, 2, 1),  # n_f, 4, 4
            #nn.ELU(),
            #nn.LeakyReLU(0.05),
            nn.SiLU(),
            nn.Conv2d(n_f*4, n_f*8, 4, 2, 1),   # n_f, 2, 2
            #nn.ELU(),
            #nn.LeakyReLU(0.05),
            nn.SiLU(),
            #nn.Conv2d(n_f*8, 1, 2, 1, 0),  # n_f, 1, 1
            #Square()
            #nn.ELU(),
            #nn.SiLU()
            #nn.Conv2d(n_f*8, 10, 2, 1, 0)
         )

        self.cv1 = nn.Conv2d(n_f * 8, 1, 2, 1, 0)
        self.cv2 = nn.Conv2d(n_f * 8, 1, 2, 1, 0)
        self.cv3 = nn.Conv2d(n_f * 8, 1, 2, 1, 0)
        self.Square = Square()

    def forward(self, x, y=None, guidance=None):
        if guidance is not None:
            x = torch.cat([x, guidance], dim=1)
        out_feat = self.f(x)#.view(x.size(0), -1)
        out_feat = self.cv1(out_feat)*self.cv2(out_feat) + self.cv3(self.Square(out_feat))
        #out_feat = 0.5*(out_feat.view(x.size(0), -1))**2
        if self.cond:
            assert y is not None
            out_feat = torch.gather(out_feat, 1, y[:, None]).squeeze()
        return out_feat.squeeze()


class VanillaNet_ELU_2_att(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.05, cond=False):
        super(VanillaNet_ELU_2_att, self).__init__()
        self.cond = cond
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),  # n_f, 16, 16
            nn.SiLU(),
            SA(n_f),
            nn.SiLU(),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),  # n_f, 8, 8
            nn.SiLU(),
            SA(n_f * 2),
            nn.SiLU(),
            nn.Conv2d(n_f*2, n_f*4, 4, 2, 1),  # n_f, 4, 4
            nn.SiLU(),
            SA(n_f*4),
            nn.SiLU(),
            nn.Conv2d(n_f*4, n_f*8, 4, 2, 1),   # n_f, 2, 2
            nn.SiLU(),
         )

        self.cv1 = nn.Conv2d(n_f * 8, 1, 2, 1, 0)
        self.cv2 = nn.Conv2d(n_f * 8, 1, 2, 1, 0)
        self.cv3 = nn.Conv2d(n_f * 8, 1, 2, 1, 0)
        self.Square = Square()

    def forward(self, x, y=None, guidance=None):
        if guidance is not None:
            x = torch.cat([x, guidance], dim=1)
        out_feat = self.f(x)#.view(x.size(0), -1)
        out_feat = self.cv1(out_feat)*self.cv2(out_feat) + self.cv3(self.Square(out_feat))
        #out_feat = 0.5*(out_feat.view(x.size(0), -1))**2
        if self.cond:
            assert y is not None
            out_feat = torch.gather(out_feat, 1, y[:, None]).squeeze()
        return out_feat.squeeze()


class VanillaNet_DyT_JamesBond(nn.Module):
    def __init__(self, n_c=3, n_f=32, hw=16, leak=0.05, cond=False):
        super(VanillaNet_DyT_JamesBond, self).__init__()
        self.cond = cond
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),  # n_f, 16, 16
            DyT_JB([n_f]),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),  # n_f, 8, 8
            DyT_JB([n_f * 2]),
            nn.Conv2d(n_f*2, n_f*4, 4, 2, 1),  # n_f, 4, 4
            DyT_JB([n_f*4]),
            nn.Conv2d(n_f*4, n_f*8, 4, 2, 1),   # n_f, 2, 2
            DyT_JB([n_f*8]),
         )

        self.cv1 = nn.Conv2d(n_f * 8, 1, 2, 1, 0)
        self.cv2 = nn.Conv2d(n_f * 8, 1, 2, 1, 0)
        self.cv3 = nn.Conv2d(n_f * 8, 1, 2, 1, 0)
        self.Square = Square()

    def forward(self, x, y=None, guidance=None):
        if guidance is not None:
            x = torch.cat([x, guidance], dim=1)
        out_feat = self.f(x)#.view(x.size(0), -1)
        out_feat = self.cv1(out_feat)*self.cv2(out_feat) + self.cv3(self.Square(out_feat))
        #out_feat = 0.5*(out_feat.view(x.size(0), -1))**2
        if self.cond:
            assert y is not None
            out_feat = torch.gather(out_feat, 1, y[:, None]).squeeze()
        return out_feat.squeeze()


class VanillaNet_DyT_JamesBond_att(nn.Module):
    def __init__(self, n_c=3, n_f=32, hw=16, leak=0.05, cond=False):
        super(VanillaNet_DyT_JamesBond_att, self).__init__()
        self.cond = cond
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),  # n_f, 16, 16
            DyT_JB(n_f),
            SA(n_f),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),  # n_f, 8, 8
            DyT_JB(n_f*2),
            SA(n_f*2),
            nn.Conv2d(n_f*2, n_f*4, 4, 2, 1),  # n_f, 4, 4
            DyT_JB(n_f * 4),
            SA(n_f * 4),
            nn.Conv2d(n_f*4, n_f*8, 4, 2, 1),   # n_f, 2, 2
            DyT_JB(n_f * 8),
         )

        self.cv1 = nn.Conv2d(n_f * 8, 1, 2, 1, 0)
        self.cv2 = nn.Conv2d(n_f * 8, 1, 2, 1, 0)
        self.cv3 = nn.Conv2d(n_f * 8, 1, 2, 1, 0)
        self.Square = Square()

    def forward(self, x, y=None, guidance=None):
        if guidance is not None:
            x = torch.cat([x, guidance], dim=1)
        out_feat = self.f(x)#.view(x.size(0), -1)
        out_feat = self.cv1(out_feat)*self.cv2(out_feat) + self.cv3(self.Square(out_feat))
        #out_feat = 0.5*(out_feat.view(x.size(0), -1))**2
        if self.cond:
            assert y is not None
            out_feat = torch.gather(out_feat, 1, y[:, None]).squeeze()
        return out_feat.squeeze()

class VanillaNet_ELU_3(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.05, cond=False):
        super(VanillaNet_ELU_3, self).__init__()
        self.cond = cond
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),  # n_f, 16, 16
            #nn.ELU(),
            #nn.LeakyReLU(0.05),
            nn.SiLU(),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),  # n_f, 8, 8
            #nn.ELU(),
            #nn.LeakyReLU(0.05),
            nn.SiLU(),
            nn.Conv2d(n_f*2, n_f*2, 4, 2, 1),  # n_f, 4, 4
            #nn.ELU(),
            #nn.LeakyReLU(0.05),
            nn.SiLU(),
            nn.Conv2d(n_f*2, n_f*4, 4, 2, 1),   # n_f, 2, 2
            #nn.ELU(),
            #nn.LeakyReLU(0.05),
            nn.SiLU(),
            nn.Conv2d(n_f * 4, n_f * 4, 4, 2, 1),  # n_f, 2, 2
            nn.SiLU(),
            #nn.Conv2d(n_f*8, 1, 2, 1, 0),  # n_f, 1, 1
            #Square()
            #nn.ELU(),
            #nn.SiLU()
            #nn.Conv2d(n_f*8, 10, 2, 1, 0)
         )

        self.cv1 = nn.Conv2d(n_f * 4, 1, 2, 1, 0)
        self.cv2 = nn.Conv2d(n_f * 4, 1, 2, 1, 0)
        self.cv3 = nn.Conv2d(n_f * 4, 1, 2, 1, 0)
        self.Square = Square()

    def forward(self, x, y=None, guidance=None):
        if guidance is not None:
            x = torch.cat([x, guidance], dim=1)
        out_feat = self.f(x)#.view(x.size(0), -1)
        out_feat = self.cv1(out_feat)*self.cv2(out_feat) + self.cv3(self.Square(out_feat))
        #out_feat = 0.5*(out_feat.view(x.size(0), -1))**2
        if self.cond:
            assert y is not None
            out_feat = torch.gather(out_feat, 1, y[:, None]).squeeze()
        return out_feat.squeeze()

class VanillaNet_ELU_2_l2(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.05, cond=False):
        super(VanillaNet_ELU_2_l2, self).__init__()
        self.cond = cond
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),  # n_f, 16, 16
            nn.ELU(),
            #nn.LeakyReLU(0.05),
            #nn.SiLU(),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),  # n_f, 8, 8
            nn.ELU(),
            #nn.LeakyReLU(0.05),
            #nn.SiLU(),
            nn.Conv2d(n_f*2, n_f*4, 4, 2, 1),  # n_f, 4, 4
            nn.ELU(),
            #nn.LeakyReLU(0.05),
            #nn.SiLU(),
            nn.Conv2d(n_f*4, n_f*8, 4, 2, 1),   # n_f, 2, 2
            nn.ELU(),
            #nn.LeakyReLU(0.05),
            #nn.SiLU(),
            #nn.Conv2d(n_f*8, 1, 2, 1, 0),  # n_f, 1, 1
            #Square()
            #nn.ELU(),
            #nn.SiLU()
            nn.Conv2d(n_f*8, 64, 2, 1, 0)
         )

        #self.cv1 = nn.Conv2d(n_f * 8, 1, 2, 1, 0)
        #self.cv2 = nn.Conv2d(n_f * 8, 1, 2, 1, 0)
        #self.cv3 = nn.Conv2d(n_f * 8, 1, 2, 1, 0)
        #self.Square = Square()

    def forward(self, x, y=None, guidance=None):
        if guidance is not None:
            x = torch.cat([x, guidance], dim=1)
        out_feat = self.f(x)#.view(x.size(0), -1)
        out_feat = 0.5*(out_feat.pow(2).sum(dim=[-1, -2, -3]))
        #out_feat = self.cv1(out_feat)*self.cv2(out_feat) + self.cv3(self.Square(out_feat))
        #out_feat = 0.5*(out_feat.view(x.size(0), -1))**2
        if self.cond:
            assert y is not None
            out_feat = torch.gather(out_feat, 1, y[:, None]).squeeze()
        return out_feat.squeeze()


class VanillaNet_ELU_lt(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.05, cond=False):
        super(VanillaNet_ELU_lt, self).__init__()
        self.cond = cond
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),  # n_f, 8, 8
            nn.ELU(),
            nn.Conv2d(n_f, n_f * 2, 3, 1, 1),  # n_f, 8, 8
            nn.ELU(),
            nn.Conv2d(n_f*2, n_f*4, 4, 2, 1),  # n_f, 4, 4
            nn.ELU(),
            nn.Conv2d(n_f*4, n_f*8, 4, 2, 1),   # n_f, 2, 2
            nn.ELU(),
            nn.Conv2d(n_f*8, 10, 2, 1, 0) # n_f, 1, 1
        )

    def forward(self, x, y=None, guidance=None):
        if guidance is not None:
            x = torch.cat([x, guidance], dim=1)
        out_feat = self.f(x).view(x.size(0), -1)
        if self.cond:
            assert y is not None
            out_feat = torch.gather(out_feat, 1, y[:, None]).squeeze()
        return out_feat

class VanillaNet_SiLU(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.05, cond=False):
        super(VanillaNet_SiLU, self).__init__()
        self.cond = cond
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),  # n_f, 32, 32
            nn.SiLU(),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),  # n_f, 16, 16
            nn.SiLU(),
            nn.Conv2d(n_f*2, n_f*4, 4, 2, 1),  # n_f, 8, 8
            nn.SiLU(),
            nn.Conv2d(n_f*4, n_f*8, 4, 2, 1),   # n_f, 4, 4
            nn.SiLU(),
            nn.Conv2d(n_f*8, n_f*8, 2, 1, 0),
            nn.SiLU(),
            nn.Conv2d(n_f*8, 10, 2, 1, 0))

    def forward(self, x, y=None, guidance=None):
        if guidance is not None:
            x = torch.cat([x, guidance], dim=1)
        out_feat = self.f(x).view(x.size(0), -1)
        if self.cond:
            assert y is not None
            out_feat = torch.gather(out_feat, 1, y[:, None]).squeeze()
        return out_feat


class Upsample(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(
            x, scale_factor=2, mode="nearest"
        )
        x = self.conv(x)
        return x

class VanillaUNET_Tanh(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.05, cond=False):
        super(VanillaUNET_Tanh, self).__init__()
        ##conv down
        self.conv_down_1 = nn.Conv2d(n_c, n_f, 3, 1, 1)
        self.conv_down_2 = nn.Conv2d(n_f, n_f * 2, 4, 2, 1)
        self.conv_down_3 = nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1)
        self.conv_down_4 = nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1)
        self.conv_down_5 = nn.Conv2d(n_f * 8, n_f * 16, 4, 2, 1)

        ## bottleneck
        self.conv_last = nn.Conv2d(n_f * 16, n_f * 32, 4, 2, 1)

        ## convup
        self.conv_up_5 = Upsample(n_f * 32, n_f * 16)
        self.conv_up_4 = Upsample(2 * n_f * 16, n_f * 8)
        self.conv_up_3 = Upsample(2 * n_f * 8, n_f * 4)
        self.conv_up_2 = Upsample(2 * n_f * 4, n_f * 2)
        self.conv_up_1 = Upsample(2 * n_f * 2, n_f)

        ## final_conv
        self.final_conv = nn.Conv2d(2 * n_f, 1, 3, padding=1)

    def forward(self, x, y=None, guidance=None):
        hs = []
        h1 = torch.relu(self.conv_down_1(x))
        hs.append(h1)
        h2 = torch.relu(self.conv_down_2(h1))
        hs.append(h2)
        h3 = torch.relu(self.conv_down_3(h2))
        hs.append(h3)
        h4 = torch.relu(self.conv_down_4(h3))
        hs.append(h4)
        h5 = torch.relu(self.conv_down_5(h4))
        hs.append(h5)

        btnck = torch.relu(self.conv_last(h5))

        h5_u = torch.relu(self.conv_up_5(btnck))
        h5_c = torch.cat([h5_u, hs.pop()], dim=1)
        h4_u = torch.relu(self.conv_up_4(h5_c))
        h4_c = torch.cat([h4_u, hs.pop()], dim=1)
        h3_u = torch.relu(self.conv_up_3(h4_c))
        h3_c = torch.cat([h3_u, hs.pop()], dim=1)
        h2_u = torch.relu(self.conv_up_2(h3_c))
        h2_c = torch.cat([h2_u, hs.pop()], dim=1)
        h1_u = torch.relu(self.conv_up_1(h2_c))
        h1_c = torch.cat([h1_u, hs.pop()], dim=1)
        im = self.final_conv(h1_c)
        return 0.5*im.pow(2).sum(dim=[1,2,3])


class NonlocalNet(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.05):
        super(NonlocalNet, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=n_c, out_channels=n_f, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=leak),
            nn.MaxPool2d(2),

            NonLocalBlock(in_channels=n_f),
            nn.Conv2d(in_channels=n_f, out_channels=n_f * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=leak),
            nn.MaxPool2d(2),

            NonLocalBlock(in_channels=n_f * 2),
            nn.Conv2d(in_channels=n_f * 2, out_channels=n_f * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=leak),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=(n_f * 4) * 2 * 2, out_features=n_f * 8),
            nn.LeakyReLU(negative_slope=leak),
            nn.Linear(in_features=n_f * 8, out_features=1)
        )

    def forward(self, x):
        conv_out = self.convs(x).view(x.shape[0], -1)
        return self.fc(conv_out).squeeze()

# structure of non-local block (from Non-Local Neural Networks https://arxiv.org/abs/1711.07971)
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, sub_sample=True):
        super(NonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = max(1, in_channels // 2)

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_c = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_c, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        w_y = self.W(y)
        z = w_y + x

        return z

class DyT_JB(nn.Module):
    def __init__(self, channel, alpha_init_value=0.5):
        super().__init__()
        self.channel = channel
        self.alpha_init_value = alpha_init_value

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(channel))
        self.bias = nn.Parameter(torch.zeros(channel))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return x

class SA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = q.view(B, C, H * W).permute(0, 2, 1)
        k = k.view(B, C, H * W)
        v = v.view(B, C, H * W).permute(0, 2, 1)
        attn = torch.bmm(q, k) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).view(B, C, H, W)
        return self.proj(out) + x