import abc
import logging

import numpy as np
from overrides import overrides  # type: ignore
from scipy import optimize  # type: ignore

from challenge.models.kernel import Kernel

log = logging.getLogger("challenge")


class Predictor(abc.ABC):
    """Very general blue print for an implementation of a predictor."""

    fitted = False

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Learns the model to fit input `X` to output `y`.

        Parameters
        ----------
        X : (N, d) array
        y : (N,) array
        """
        ...

    @abc.abstractmethod
    def f(self, X: np.ndarray):
        """Prediction function.

        This can be directly linked to prediction for regressors, or indirectly linked for classifiers.

        Notes
        -----
        This function cannot be called before the model was fitted with the `fit` method on some training data.
        """

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts output from input `X`.
        Parameters
        ----------
        X : (N, d) array

        Returns
        -------
        y : (N,) array

        Notes
        -----
        This function cannot be called before the model was fitted with the `fit` method on some training data.
        """
        ...


class KernelRidgeRegressor(Predictor):
    r"""Kernel Ridge Regression Classifier

    This finds the function :maths:`f` that solves the optimisation problem
    .. math:: \min_{f \in H} \frac{1}{n} \sum_i (y_i - f(x_i))^2 + \lambda ||f||^2_H
    """

    def __init__(self, kernel: Kernel, lambd: float = 1):
        self.kernel = kernel
        self.lambd = lambd
        self.X_fit = None
        self.beta = None  # coordinates from Representer theorem

    @overrides(check_signature=False)
    def fit(self, X, y):
        K = self.kernel.gram(X, X)
        n = len(y)
        self.beta = np.linalg.solve(K + self.lambd * n * np.identity(n), y)
        self.X_fit = X
        self.fitted = True

    @overrides(check_signature=False)
    def f(self, X):
        assert self.fitted, "Classifier needs to be fitted first"
        K = self.kernel.gram(X, self.X_fit)
        return K @ self.beta

    @overrides(check_signature=False)
    def predict(self, X):
        return self.f(X)


class KernelSVC(Predictor):
    """Kernel Support Vector Classifier"""

    def __init__(self, C: float, kernel: Kernel, epsilon: float = 1e-3):
        assert epsilon < 1

        self.type = "non-linear"
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.beta_support = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None

    @overrides(check_signature=False)
    def fit(self, X, y):
        K = self.kernel.gram(X, X)
        YKY = np.einsum("ij,i,j->ij", K, y, y)
        N = len(y)

        def loss(alpha: np.ndarray):
            """Lagrange dual problem"""
            assert alpha.ndim <= 1, "alpha must be 0 or 1 dimensional"
            return 1 / 2 * alpha.T @ YKY @ alpha - alpha.sum()

        def grad_alpha_loss(alpha: np.ndarray):
            """Partial derivate of Ld on alpha"""
            return YKY @ alpha - np.ones(alpha.shape)

        def equality_constraint(alpha):
            return alpha @ y

        def jacobian_eq_cons(alpha):
            return y

        def inequality_constraint(alpha):
            return np.concatenate((alpha, self.C - alpha), axis=0)

        _jacobian_ineq_cons = np.concatenate((np.diag(np.ones(N)), -np.diag(np.ones(N))), axis=0)

        def jacobian_ineq_cons(alpha):
            return _jacobian_ineq_cons

        constraints = (
            {"type": "eq", "fun": equality_constraint, "jac": jacobian_eq_cons},
            {"type": "ineq", "fun": inequality_constraint, "jac": jacobian_ineq_cons},
        )

        optRes = optimize.minimize(
            fun=loss,
            x0=np.ones(N),
            method="SLSQP",
            jac=grad_alpha_loss,
            constraints=constraints,
        )
        self.alpha = optRes.x

        self.alpha[self.alpha < self.epsilon] = 0
        self.alpha[self.alpha > self.C * (1 - self.epsilon)] = self.C

        self.beta_support = self.alpha[self.alpha > 0] * y[self.alpha > 0]
        self.X_support = X[self.alpha > 0]
        self.b = np.mean((1 / y - K @ (y * self.alpha))[(0 < self.alpha) & (self.alpha < self.C)])

        self.fitted = True

    @overrides(check_signature=False)
    def f(self, x: np.ndarray):
        """Separating function :maths:`f` evaluated at `x`"""
        assert self.fitted, "Classifier needs to be fitted first"
        K = self.kernel.gram(self.X_support, x)
        return self.beta_support @ K

    @overrides(check_signature=False)
    def predict(self, X: np.ndarray):
        """Predict y values in {-1, 1}"""
        return 2 * (self.f(X) + self.b > 0) - 1
