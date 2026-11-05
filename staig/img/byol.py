"""Minimal BYOL (Grill et al. 2020) for histology patches, implemented in plain PyTorch.

Same augmentations (applied per batch), symmetric cosine loss, projector/predictor shapes, EMA decay and
un-normalised 2048-d embedding output as the workflow of the paper (which used byol-pytorch 0.6); no
external BYOL package is needed.

* backbone: torchvision ResNet-50 (ImageNet V1 weights), representation = avgpool output (2048-d)
* projector / predictor: Linear -> BatchNorm1d -> ReLU -> Linear (hidden 4096, output 256)
* target network: EMA copy of the online encoder+projector (decay 0.99), no predictor
* loss: 2 - 2 cos(pred, target) summed over the two symmetric view pairs
* both views use the same augmentation as the v1 notebook (colour jitter, grayscale, flips,
  blur, rotation, resized crop, ImageNet normalisation)
"""

from __future__ import annotations

import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn, self.p = fn, p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


def default_augmentation(image_size: int = 256) -> nn.Module:
    import torchvision.transforms as T

    return nn.Sequential(
        RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
        T.RandomGrayscale(p=0.2),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
        T.RandomRotation(degrees=(0, 360)),
        T.RandomResizedCrop((image_size, image_size)),
        T.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
    )


class MLP(nn.Module):
    def __init__(self, dim, projection_size=256, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(inplace=True),
                                 nn.Linear(hidden_size, projection_size))

    def forward(self, x):
        return self.net(x)


class EncoderWithProjector(nn.Module):
    """Backbone up to the global pooling + projector."""

    def __init__(self, backbone: nn.Module, feat_dim: int, projection_size: int, hidden_size: int):
        super().__init__()
        self.backbone = backbone
        self.projector = MLP(feat_dim, projection_size, hidden_size)

    def representation(self, x):
        return torch.flatten(self.backbone(x), 1)

    def forward(self, x):
        rep = self.representation(x)
        return self.projector(rep), rep


def make_backbone(name: str = "resnet50", pretrained: bool = True) -> tuple[nn.Module, int]:
    import torchvision

    if name == "resnet50":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet50(weights=weights)
    elif name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
    else:
        raise ValueError(f"unknown backbone {name!r}")
    feat_dim = net.fc.in_features
    net.fc = nn.Identity()           # output of avgpool (flattened by ResNet.forward)
    return net, feat_dim


class BYOL(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, image_size: int = 256, projection_size: int = 256,
                 projection_hidden_size: int = 4096, moving_average_decay: float = 0.99,
                 augment_fn: nn.Module | None = None, augment_fn2: nn.Module | None = None):
        super().__init__()
        self.augment1 = augment_fn if augment_fn is not None else default_augmentation(image_size)
        self.augment2 = augment_fn2 if augment_fn2 is not None else self.augment1
        self.online = EncoderWithProjector(backbone, feat_dim, projection_size, projection_hidden_size)
        self.predictor = MLP(projection_size, projection_size, projection_hidden_size)
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.beta = moving_average_decay

    @staticmethod
    def _loss(x, y):
        return 2 - 2 * (F.normalize(x, dim=-1) * F.normalize(y, dim=-1)).sum(-1)

    def forward(self, x):
        v1, v2 = self.augment1(x), self.augment2(x)
        p1 = self.predictor(self.online(v1)[0])
        p2 = self.predictor(self.online(v2)[0])
        with torch.no_grad():
            t1 = self.target(v1)[0].detach()
            t2 = self.target(v2)[0].detach()
        return (self._loss(p1, t2) + self._loss(p2, t1)).mean()

    @torch.no_grad()
    def update_moving_average(self):
        for pt, po in zip(self.target.parameters(), self.online.parameters()):
            pt.mul_(self.beta).add_(po.detach(), alpha=1 - self.beta)
        for bt, bo in zip(self.target.buffers(), self.online.buffers()):
            bt.copy_(bo)

    @torch.no_grad()
    def embed(self, x):
        return self.online.representation(x)


class PatchDataset(Dataset):
    """uint8 (n, h, w, 3) patches -> float tensors resized to ``image_size`` (PIL bilinear, as v1).

    ``patches`` may be an array or a path to a ``.npy`` file; a file is memory-mapped lazily in every
    worker process so the dataset stays cheap to pickle (needed for spawn-based DataLoader workers).
    """

    def __init__(self, patches, image_size: int = 256):
        import torchvision.transforms as T

        self._path = None
        self._patches = None
        if isinstance(patches, (str, os.PathLike)):
            self._path = str(patches)
            self._n = int(np.load(self._path, mmap_mode="r").shape[0])
        else:
            self._patches = patches
            self._n = int(patches.shape[0])
        self.tf = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

    @property
    def patches(self):
        if self._patches is None:
            self._patches = np.load(self._path, mmap_mode="r")
        return self._patches

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        from PIL import Image

        return self.tf(Image.fromarray(np.asarray(self.patches[i])))


def _loader_kwargs(num_workers: int) -> dict:
    if num_workers <= 0:
        return dict(num_workers=0)
    import multiprocessing as mp

    return dict(num_workers=num_workers, multiprocessing_context=mp.get_context("spawn"),
                persistent_workers=True)


def train_byol(
    patches: np.ndarray,
    epochs: int = 1,
    batch_size: int = 43,
    lr: float = 3e-4,
    image_size: int = 256,
    backbone: str = "resnet50",
    pretrained: bool = True,
    moving_average_decay: float = 0.99,
    seed: int = 0,
    device: torch.device | str = "cuda",
    amp: bool = False,
    num_workers: int = 0,
    normalize_for_embedding: bool = False,
    verbose: bool = True,
    embed_batch_size: int = 128,
) -> tuple[np.ndarray, dict]:
    """Train BYOL on the patches and return (features (n, feat_dim) float32, info dict).

    ``normalize_for_embedding=False`` reproduces v1, which fed un-normalised [0, 1] images to the
    encoder when extracting the final representation.
    """
    from tqdm.auto import tqdm

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(device)
    net, feat_dim = make_backbone(backbone, pretrained)
    learner = BYOL(net, feat_dim, image_size=image_size, moving_average_decay=moving_average_decay).to(device)
    opt = torch.optim.Adam(learner.parameters(), lr=lr)
    ds = PatchDataset(patches, image_size)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=False,
                        pin_memory=device.type == "cuda", **_loader_kwargs(num_workers))
    if device.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    scaler = torch.amp.GradScaler(enabled=amp and device.type == "cuda")
    losses = []
    learner.train()
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for images in tqdm(loader, disable=not verbose, leave=False, desc=f"BYOL epoch {ep + 1}/{epochs}"):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                loss = learner(images)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            learner.update_moving_average()
            tot += float(loss.detach())
            nb += 1
        losses.append(tot / max(nb, 1))
    learner.eval()
    feats = []
    eval_loader = DataLoader(ds, batch_size=embed_batch_size, shuffle=False, **_loader_kwargs(num_workers))
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    with torch.no_grad():
        for images in tqdm(eval_loader, disable=not verbose, leave=False, desc="BYOL embed"):
            images = images.to(device, non_blocking=True)
            if normalize_for_embedding:
                images = (images - mean) / std
            feats.append(learner.embed(images).float().cpu().numpy())
    feats = np.vstack(feats).astype(np.float32)
    info = dict(epochs=epochs, batch_size=batch_size, lr=lr, image_size=image_size, backbone=backbone,
                pretrained=pretrained, moving_average_decay=moving_average_decay, seed=seed, amp=amp,
                losses=losses, feat_dim=int(feat_dim), normalize_for_embedding=normalize_for_embedding)
    return feats, info


def pretrained_features(patches, image_size: int = 256, backbone: str = "resnet50",
                        device="cuda", num_workers: int = 0, batch_size: int = 128,
                        normalize: bool = True) -> np.ndarray:
    """ImageNet features without BYOL training (``method="resnet50"``)."""
    device = torch.device(device)
    net, _ = make_backbone(backbone, True)
    net = net.to(device).eval()
    ds = PatchDataset(patches, image_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, **_loader_kwargs(num_workers))
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            if normalize:
                images = (images - mean) / std
            out.append(torch.flatten(net(images), 1).float().cpu().numpy())
    return np.vstack(out).astype(np.float32)
