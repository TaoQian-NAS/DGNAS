import torch
import torch.nn as nn
import torch.nn.functional as F


class Gnorm:
    def __init__(self, norm_type, unbiased=False, eps=1e-5, power_root=2):
        self.unbiased = unbiased
        self.eps = eps
        self.norm_type = norm_type
        self.power = 1 / power_root

    def norm(self, x):

        if self.norm_type == "PairNorm":
            col_mean = x.mean(dim=0)
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = x / rownorm_mean
            return x

        elif self.norm_type == "MeanNorm":
            col_mean = x.mean(dim=0)
            x = x - col_mean
            return x

        elif self.norm_type == "NodeNorm":
            mean = torch.mean(x, dim=1, keepdim=True)
            std = (
                    torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = (x - mean) / std
            return x
        else:
            return x


class mean_norm(torch.nn.Module):
    def __init__(self):
        super(mean_norm, self).__init__()
        # print(f'------ ›››››››››  {self._get_name()}')

    def forward(self, x):
        col_mean = x.mean(dim=0)
        x = x - col_mean
        return x
