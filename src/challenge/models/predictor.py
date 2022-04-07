import abc
import logging

import numpy as np
import tqdm
from overrides import overrides  # type: ignore
from scipy import optimize  # type: ignore

from challenge.models.kernel import Kernel

log = logging.getLogger("challenge")


class Predictor(abc.ABC):
    """Very general blue print for an implementation of a predictor."""

    fitted = False

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

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
    def fit(self, X, y, warn_unbalance=True):
        assert y.dtype == np.bool_, "Y argument must be a boolean array"
        y = 2 * y - 1

        if warn_unbalance and np.abs(y.sum() / len(y)) > 0.15:
            log.warn("Y vectors contains unbalanced labels")

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
        return self.beta_support @ K + self.b

    @overrides(check_signature=False)
    def predict(self, X: np.ndarray):
        """Predict y values in {-1, 1}"""
        return self.f(X) > 0


class OneVsAll(Predictor):
    """Multi-class predictor"""

    def __init__(self, get_predictor):
        self.get_predictor = get_predictor
        self.predictors = {}
        self.classes = set()
        self.idx_to_cl = {}

    @overrides(check_signature=False)
    def fit(self, X, y):
        self.fitted = True
        self.classes = set(y)
        self.idx_to_cl = {i: cl for i, cl in enumerate(self.classes)}
        for idx, cl in tqdm.tqdm(
            self.idx_to_cl.items(), desc="Computing all 1 v All predictors", total=len(self.classes)
        ):
            predictor = self.get_predictor()
            predictor.fit(X, y == cl)
            self.predictors[idx] = predictor

    @overrides(check_signature=False)
    def f(self, X: np.ndarray):
        assert self.fitted, "Classifier needs to be fitted first"
        return np.stack([pred.f(X) for pred in self.predictors.values()], axis=0)

    @overrides(check_signature=False)
    def predict(self, X: np.ndarray):
        """Predict y values"""
        if X.ndim == 1:
            X = X[None, :]
        return np.vectorize(lambda idx: self.idx_to_cl[idx])(np.argmax(self.f(X), axis=0))


class OneVsOne(Predictor):
    """Multi-class predictor"""

    def __init__(self, get_predictor):
        self.get_predictor = get_predictor
        self.classifiers = {}
        self.classes = set()
        self.idx_to_cl = {}

    @overrides(check_signature=False)
    def fit(self, X, y):
        self.fitted = True
        self.classes = set(y)
        self.idx_to_cl = {i: cl for i, cl in enumerate(self.classes)}
        for i in tqdm.trange(len(self.classes) - 1, desc="Computing all 1v1 classifiers", position=0):
            for j in tqdm.trange(i + 1, len(self.classes), desc=f"Involving {i}", position=1, leave=False):
                cl1, cl2 = self.idx_to_cl[i], self.idx_to_cl[j]
                select = (y == cl1) | (y == cl2)
                X_ = X[select]
                y_ = y[select]
                predictor = self.get_predictor()
                predictor.fit(X_, y_ == cl1, warn_unbalance=False)
                self.classifiers[(i, j)] = predictor

    @overrides(check_signature=False)
    def f(self, X: np.ndarray):
        assert self.fitted, "Classifier needs to be fitted first"
        if X.ndim == 1:
            X = X[None, :]
        shape = X.shape
        X = X.reshape(-1, shape[-1])

        Y = np.empty(len(X))
        for x_idx, x in tqdm.tqdm(
            enumerate(X),
            total=len(X),
        ):
            count_wins = [0 for _ in range(len(self.classes))]
            for (i, j), classifier in self.classifiers.items():
                y = classifier(x)
                if y.sum() > (1 - y).sum():
                    count_wins[i] += 1
                else:
                    count_wins[j] += 1
            cl = self.idx_to_cl[np.argmax(count_wins)]
            Y[x_idx] = cl
        return Y

    @overrides(check_signature=False)
    def predict(self, X: np.ndarray):
        """Predict y values"""
        return self.f(X)
