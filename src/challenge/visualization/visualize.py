import matplotlib.pyplot as plt  # type: ignore
import numpy as np

from challenge.data.io import imreshape


def plot(row: np.ndarray, ax=None):
    row = (row - row.min()) / (row.max() - row.min())
    im = imreshape(row)
    if ax is not None:
        plt.sca(ax)
    plt.imshow(im)


labels = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}
