import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def is_weight_module(m, n):
    return hasattr(m, 'weight') and isinstance(getattr(m, 'weight', None), torch.Tensor)


class _SpectralShift(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        dim: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        ndim = weight.ndim
        self.dim = dim if dim >= 0 else dim + ndim
        if ndim > 1:
            weight_mat = self._reshape_weight_to_matrix(weight)
            with torch.no_grad():
                u, s, vh = torch.linalg.svd(weight_mat, full_matrices=False)
            self.register_buffer('_u', u.detach())
            self.register_buffer('_s', s.detach())
            self.register_buffer('_vh', vh.detach())
            self._u.requires_grad = False
            self._s.requires_grad = False
            self._vh.requires_grad = False
            eigenvalues_delta = torch.randn_like(s) * 0.0
        else:
            eigenvalues_delta = torch.randn_like(weight) * 0.0
        self.eigenvalues_delta = nn.Parameter(eigenvalues_delta, requires_grad=True)

    def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        assert weight.ndim > 1

        if self.dim != 0:
            # permute dim to front
            weight = weight.permute(self.dim, *(d for d in range(weight.dim()) if d != self.dim))

        return weight.flatten(1)  # NOTE: for conv2d it will be (c_out, c_in * k_x * k_y)
    
    def _reshape_matrix_to_weight(self, weight_mat: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        if self.dim != 0:
            # dim permuted to front
            weight = weight_mat.reshape(shape[self.dim], *(shape[d] for d in range(len(shape)) if d != self.dim))
            weight = weight.permute(*np.argsort([self.dim] + [d for d in range(weight.dim()) if d != self.dim]))
        else:
            weight = weight_mat.reshape(shape)

        return weight

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        eigenvalues_delta = self.eigenvalues_delta if self.eigenvalues_delta_cached is None else self.eigenvalues_delta_cached
        if weight.ndim == 1:
            delta = eigenvalues_delta
            return weight.detach() + delta
        else:
            weight_mat = self._u @ torch.diag(F.relu(self._s + eigenvalues_delta)) @ self._vh
            weight_mat = weight_mat + self._residual.detach()
            return self._reshape_matrix_to_weight(weight_mat, weight.shape)


def spectral_shift(module, name='weight', dim=None, cached_svd_params=None, svd_kwargs=None):
    r"""
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter. Default: ``"weight"``.
        dim (int, optional): dimension corresponding to number of outputs.
            Default: ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with a new parametrization registered to the specified
        weight
    """
    if svd_kwargs is None:
        svd_kwargs = {}
    weight = getattr(module, name, None)
    if not isinstance(weight, torch.Tensor):
        raise ValueError(
            "Module '{}' has no parameter or buffer with name '{}'".format(module, name)
        )

    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    new_module = _SpectralShift(weight, dim=dim,
        cached_svd_params=cached_svd_params, **svd_kwargs)
    nn.utils.parametrize.register_parametrization(module, name, new_module)
    return module, new_module  # NOTE: module is parametrized inplace

