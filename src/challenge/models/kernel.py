import abc
import logging

import numpy as np
from numpy.typing import ArrayLike
from overrides import overrides  # type: ignore

log = logging.getLogger("challenge")


class Kernel(abc.ABC):
    """Blue print for an implementation of a specific kernel :math:`k`."""

    @abc.abstractmethod
    def gram(self, X: ArrayLike, Y: ArrayLike) -> np.ndarray:
        """Computes the Gram matrix K(X, Y).

        Parameters
        ----------
        X : array like
            Input array of dimensions (..., N, d) or (d)
        Y : array like
            Input array of dimensions (..., M, d) or (d)
        Returns
        -------
        K : numpy.ndarray
            The Gram matrix of the elements :math:`k(x_i, x_j)`.

        Notes
        -----
        - if `X` and `Y` are 1 dimensional, `K` is 0-dimensional
        - if only one of `X` and `Y` are 1 dimensional, `K` is 1-dimensional
        - else all dimensions but the before-last one of `X` and `Y` must match, and `K` has one less dimension than them.
        """
        ...


class RBF(Kernel):
    r"""Radial Basis Function kernel implementation.

    The RBF kernel is defined such that
    .. math:: k(x, y) = \exp{-\frac{||x-y||^2_2}{2\sigma^2}}
    """

    def __init__(self, sigma: float = 1.0):
        """
        Parameters
        ----------
        sigma
            The variance of the kernel
        """
        self.sigma = sigma

    @overrides(check_signature=False)
    def gram(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)

        if X.ndim == 1 and Y.ndim == 1:
            norm = np.sum((X - Y) ** 2)
        elif X.ndim == 1:
            norm = np.sum((X[None, :] - Y) ** 2, axis=1)
        elif Y.ndim == 1:
            norm = np.sum((X - Y[None, :]) ** 2, axis=1)
        else:
            norm = np.sum((X[..., :, None, :] - Y[..., None, :, :]) ** 2, axis=-1)
        ret = np.exp(-norm / (2 * self.sigma**2))
        log.debug(f"Computed Gram matrix for RBF Kernel of size {ret.size}", extra={"markup": True})
        return ret


class Linear(Kernel):
    r"""Linear kernel implementation.

    The Linear kernel is defined such that
    .. math:: k(x, y) = \langle x, y \rangle
    """

    @overrides(check_signature=False)
    def gram(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)

        if X.ndim == 1 and Y.ndim == 1:
            ret = X * Y
        elif X.ndim == 1:
            ret = Y @ X
        elif Y.ndim == 1:
            ret = X @ Y
        else:
            ret = np.einsum("...id,...jd->...ij", X, Y)
        log.debug(f"Computed Gram matrix of Linear Kernel of size {ret.size}", extra={"markup": True})
        return ret
