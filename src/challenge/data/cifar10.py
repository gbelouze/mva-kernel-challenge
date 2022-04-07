import torch
import torchvision as V
import torchvision.transforms as T

from challenge.data.paths import data_dir

transform = T.Compose([T.Resize(224), T.ToTensor()])

resize = T.Resize(224)

transform_normalize = T.Compose([transform, T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def cifar_loader():
    dataset = V.datasets.CIFAR10(root=data_dir, train=True, transform=transform_normalize, download=True)
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=16)
