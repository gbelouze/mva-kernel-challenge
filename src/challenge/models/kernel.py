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
            normX = np.sum(X**2, axis=-1)
            normY = np.sum(Y**2, axis=-1)
            XY = np.einsum("...id,...jd->...ij", X, Y)
            norm = normX[..., :, None] + normY[..., None, :] - 2 * XY
        ret = np.exp(-norm / (2 * self.sigma**2))
        shape = f"{ret.shape[0]}" if ret.ndim == 1 else f"{ret.shape[-2]}x{ret.shape[-1]}"
        log.debug(f"Computed Gram matrix for RBF Kernel of size [bold cyan]{shape}[/]", extra={"markup": True})
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
        shape = f"{ret.shape[0]}" if ret.ndim == 1 else f"{ret.shape[-2]}x{ret.shape[-1]}"
        log.debug(f"Computed Gram matrix of Linear Kernel of size [bold cyan]{shape}[/]", extra={"markup": True})
        return ret


class Polynomial(Kernel):
    r"""The polynomial kernel of degree :math:`\gamma` is defined as
    .. math:: k(x, y) = \langle x, y \rangle^{\gamma}
    """

    def __init__(self, degree: int = 3):
        """
        Parameters
        ----------
            degree
                The polynomial degree of the kernel
        """
        self.degree = degree

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
        ret = ret**self.degree

        shape = f"{ret.shape[0]}" if ret.ndim == 1 else f"{ret.shape[-2]}x{ret.shape[-1]}"
        log.debug(f"Computed Gram matrix of Polynomial Kernel of size [bold cyan]{shape}[/]", extra={"markup": True})
        return ret


class Chi2(Kernel):
    r"""chi-squared kernel implementation.

    The chi-squared kernel is defined on non negative data such that
    .. math:: k(x, y) = \exp{-\gamma \sum_i \frac{(x_i-y_i)}^2{x_i + y_i}}
    """

    def __init__(self, gamma: float = 1.0):
        """
        Parameters
        ----------
        sigma
            The variance of the kernel
        """
        self.gamma = gamma

    @overrides(check_signature=False)
    def gram(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)

        assert (X >= 0).all(), "data must be non negative"
        assert (Y >= 0).all(), "data must be non negative"

        if X.ndim == 1 and Y.ndim == 1:
            mantiss = np.sum((X - Y) ** 2 / (X + Y))
        elif X.ndim == 1:
            mantiss = (X[None, :] - Y) ** 2 / (X[None, :] + Y)
            np.nan_to_num(mantiss, copy=False)  # replace 0/0 with 0
            mantiss = mantiss.sum(axis=1)
        elif Y.ndim == 1:
            mantiss = (X - Y[None, :]) ** 2 / (X + Y[None, :])
            np.nan_to_num(mantiss, copy=False)  # replace 0/0 with 0
            mantiss = mantiss.sum(axis=1)
        else:
            mantiss = (X[..., :, None, :] - Y[..., None, :, :]) ** 2 / (X[..., :, None, :] + Y[..., None, :, :])
            np.nan_to_num(mantiss, copy=False)  # replace 0/0 with 0
            mantiss = mantiss.sum(axis=-1)
        ret = np.exp(-self.gamma * mantiss)
        shape = f"{ret.shape[0]}" if ret.ndim == 1 else f"{ret.shape[-2]}x{ret.shape[-1]}"
        log.debug(f"Computed Gram matrix for Chi2 Kernel of size [bold cyan]{shape}[/]", extra={"markup": True})
        return ret
