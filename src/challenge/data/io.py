from pathlib import Path

import challenge.data.paths as paths
import numpy as np
import pandas as pd  # type: ignore


def xload(path: Path) -> np.ndarray:
    """Loads an image dataset"""
    X = np.genfromtxt(path, delimiter=",")
    assert (
        X.shape[1] == 3072
    ), "Did you prepare the data files ? (see challenge.make.prepare)"
    return X


def xdump(x: np.ndarray, out: Path):
    np.savetxt(out, x, delimiter=",")


def yload(path: Path) -> np.ndarray:
    """Loads a label dataset"""
    return np.genfromtxt(path, delimiter=",", skip_header=1, usecols=1)


def ydump(y: np.ndarray, out: Path = paths.y_test):
    """Writes a label dataset"""
    df = pd.DataFrame({"Prediction": y})
    df.index += 1
    df.to_csv(out, index_label="Id")


def imreshape(X: np.ndarray):
    """Reshapes to an image-like array.
    Parameters
    ----------
    X
        Input array of dimension (3072) or (N, 3072)

    Returns
    -------
    (32, 32, 3) or (N, 32, 32, 3) array
    """
    if X.ndim == 1:
        return X.reshape((32, 32, 3), order="F").transpose(1, 0, 2)
    elif X.ndim == 2:
        return X.reshape((-1, 32, 32, 3), order="F").transpose(0, 2, 1, 3)
    raise np.AxisError("Too many dimensions")
