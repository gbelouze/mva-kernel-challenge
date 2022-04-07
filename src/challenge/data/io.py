import logging
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd  # type: ignore
from PIL import Image

import challenge.data.paths as paths

log = logging.getLogger("challenge")


def repr(path: Path):
    """Path name relative to working directory"""
    cwd = Path.cwd()
    path = path.resolve()
    if path.is_relative_to(cwd):
        return str(path.relative_to(cwd))
    return path


def loadx(path: Path) -> np.ndarray:
    """Loads an image dataset"""
    X = np.genfromtxt(path, delimiter=",")
    assert X.shape[1] == 3072, "Did you prepare the data files ? (see challenge.make.prepare)"
    log.info(f"X file loaded from [magenta]{repr(path)}[/]", extra={"markup": True})
    return X


def loady(path: Path) -> np.ndarray:
    """Loads a label dataset"""
    log.info(f"Y file loaded from [magenta]{repr(path)}[/]", extra={"markup": True})
    return np.genfromtxt(path, delimiter=",", skip_header=1, usecols=1).astype(np.int64)


def overwrite_guard(func):
    def wrapped(path: Path, *args, overwrite=False, **kwargs):
        if path.exists() and not overwrite:
            log.error(f"File already exists [magenta]{repr(path)}[/]", extra={"markup": True})
            raise FileExistsError
        return func(path, *args, **kwargs)

    return wrapped


@overwrite_guard
def dumpx(out: Path, x: np.ndarray):
    np.savetxt(out, x, delimiter=",")


@overwrite_guard
def dumpy(out: Path, y: np.ndarray):
    """Writes a label dataset"""
    df = pd.DataFrame({"Prediction": y})
    df.index += 1
    df.to_csv(out, index_label="Id")


def imreshape(X: np.ndarray) -> np.ndarray:
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


def toPil(X: np.ndarray) -> Union[Image.Image, List[Image.Image]]:
    """Convert a vector to a PIL image.
    Parameters
    ----------
    X
        Input array of dimension (3072) or (N, 3072)

    Returns
    -------
    1 or N Image objects of dimension (32, 32, 3) or (32, 32, 3)
    """
    unsqueeze = X.ndim == 1
    X = imreshape(X)
    if unsqueeze:
        X = X[None, :]
    ret = []
    for x in X:
        x = 255 * (x - x.min()) / (x.max() - x.min())
        x = x.astype(np.uint8)
        ret.append(Image.fromarray(x, mode="RGB"))
    if unsqueeze:
        return ret[0]
    return ret
