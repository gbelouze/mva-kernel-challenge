import logging

import cv2
import numpy as np
import torch
import tqdm
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, Dataset

from challenge.data.cifar10 import resize, transform
from challenge.data.io import toPil

from .resnet import PartialResNet

log = logging.getLogger("challenge")


def siftBOW(X: np.ndarray, D: int = 100):
    """
    Create a dictionnary of size D of SIFT features taken from the images in X
    """
    assert X.ndim == 2
    sift = cv2.SIFT_create()

    vocab = []
    for x_idx, x in tqdm.tqdm(enumerate(X), total=len(X), desc="Computing SIFT features"):
        x = toPil(x)
        x = resize(x)
        x = np.array(x)
        kp, descs = sift.detectAndCompute(x, None)
        if len(kp) == 0:
            log.warn(f"No SIFT keypoints detected for point {x_idx}")
        else:
            for desc in descs:
                vocab.append(desc)
    log.debug("Computed all SIFT features")
    ret = MiniBatchKMeans(n_clusters=D, batch_size=1_000).fit(vocab)
    log.debug(f"Clustered SIFT features into {D} classes")
    return ret


def sift(X: np.ndarray, bow) -> np.ndarray:
    """
    Compute bag-of-words histograms of sift features
    """
    unsqueeze = X.ndim == 1
    if unsqueeze:
        X = X[None, :]
    sift = cv2.SIFT_create()

    ret = np.zeros((len(X), bow.n_clusters))
    for x_idx, x in tqdm.tqdm(enumerate(X), total=len(X), desc="Computing SIFT histograms"):
        x = toPil(x)
        x = resize(x)
        x = np.array(x)
        kp, descs = sift.detectAndCompute(x, None)
        if len(kp) == 0:
            log.warn(f"No SIFT keypoints detected for point {x_idx}")
        else:
            desc_clusters = bow.predict(descs.astype(np.float64))
            for desc_cluster in desc_clusters:
                ret[x_idx, desc_cluster] += 1
    log.debug(f"Computed SIFT histogram for {len(X)} points")
    if unsqueeze:
        return ret.squeeze()
    return ret


class XDataset(Dataset):
    def __init__(self, X, transform=None):
        self.X = toPil(X)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        if self.transform is not None:
            x = self.transform(x)
        return x


def resnet_embedding(X: np.ndarray) -> np.ndarray:
    """Compute features taken from intermediate layers of a ResNet network
    Parameters
    ----------
    X
        Input array of dimension (3072) or (N, 3072)

    Returns
    -------
    Array of features for each input image (n_features, d) or (N, n_features, d)
    """
    net = PartialResNet()
    dataset = XDataset(X, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16)

    with torch.no_grad():
        ret_batches = []
        for X_batch in dataloader:
            ret_batches.append(net.features(X_batch).numpy())
    ret = np.concatenate(ret_batches, axis=0)

    n_samples, n_features, *_ = ret.shape
    log.debug(f"Computed ResNet embeddings for {n_samples} images ({n_samples * n_features} features in total)")
    ret = ret.reshape(n_samples, n_features, -1)

    return ret.squeeze()
