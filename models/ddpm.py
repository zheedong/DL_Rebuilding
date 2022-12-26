# https://nn.labml.ai/diffusion/ddpm/index.html
# https://developers-shack.tistory.com/8
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

from labml_nn.diffusion.ddpm.utils import gather


class DenoiseDiffusion:
    def __init__(
            self,
            eps_model: nn.Module,
            n_steps: int,
            device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta

    def q_xt_x0(self, x0: torch.Tensor,
                t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        return mean, var

    def q_sample(self, x0: torch.Tensor,
                 t: torch.Tensor,
                 eps: Optional[torch.Tensor] = None):
        if eps is None:
            esp = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps
