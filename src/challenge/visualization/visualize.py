import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from challenge.data.io import imreshape


def plot(row: np.ndarray, ax=None):
    row = (row - row.min()) / (row.max() - row.min())
    im = imreshape(row)
    if ax is not None:
        plt.sca(ax)
    plt.imshow(im)
