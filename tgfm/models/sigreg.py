r"""SIGReg: Sketched Isotropic Gaussian Regularization (Epps-Pulley variant).

Reference:
    Balestriero & LeCun, "LeJEPA: Provable and Scalable Self-Supervised
    Learning Without the Heuristics" (arXiv:2511.08544), Sec. 3.

Core idea:
    Project embeddings Z in R^{B x d} onto M random unit directions to obtain
    M empirical 1-D distributions, then penalize each projection's deviation
    from N(0, 1) with the Epps-Pulley statistic

        EP = B * \int |phi_B(t) - exp(-t^2 / 2)|^2 w(t) dt,

    where phi_B is the empirical characteristic function and w is the
    standard normal density. The characteristic-function form gives bounded
    gradients and O(B * M * T) cost.

If the shared tgfm ``LeJEPALoss`` module is preferred, this file can be
replaced by a thin wrapper around its SIGReg term; the interface below is
kept minimal on purpose (Z -> scalar).
"""

import math

import torch
from torch import Tensor


class SIGReg(torch.nn.Module):
    def __init__(
        self,
        num_slices: int = 256,
        num_quad_points: int = 17,
        t_max: float = 4.0,
    ) -> None:
        super().__init__()
        self.num_slices = num_slices
        # Quadrature grid on [0, t_max]; the integrand is even in t, so the
        # integral over R is twice the integral over [0, t_max] (the normal
        # weight makes the tail beyond ~4 negligible).
        t = torch.linspace(0.0, t_max, num_quad_points)
        weight = torch.exp(-0.5 * t**2) / math.sqrt(2.0 * math.pi)
        self.register_buffer('t', t)
        self.register_buffer('weight', weight)

    def forward(self, z: Tensor) -> Tensor:
        """z: (B, d) embeddings. Returns scalar Epps-Pulley SIGReg loss."""
        assert isinstance(self.t, Tensor)
        batch_size, dim = z.shape
        directions = torch.randn(dim, self.num_slices, device=z.device, dtype=z.dtype)
        directions = directions / directions.norm(dim=0, keepdim=True)

        proj = z @ directions  # (B, M)
        tp = proj.unsqueeze(-1) * self.t  # (B, M, T)
        ecf_real = tp.cos().mean(dim=0)  # (M, T)
        ecf_imag = tp.sin().mean(dim=0)  # (M, T)
        target = torch.exp(-0.5 * self.t**2)  # (T,)

        sq_dev = (ecf_real - target) ** 2 + ecf_imag**2  # (M, T)
        # 2x for the even symmetry; trapezoidal quadrature over t.
        ep = 2.0 * batch_size * torch.trapezoid(sq_dev * self.weight, self.t, dim=-1)
        return ep.mean()
