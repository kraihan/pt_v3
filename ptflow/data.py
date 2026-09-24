"""ImageNet SD-VAE latent cache (the W-Flow format) and a synthetic stand-in for tests.

Cache layout (memory-mapped .npy, latents already multiplied by 0.18215, NHWC):
    {split}_moments.npy       (N, 32, 32, 4) float32
    {split}_moments_flip.npy  (N, 32, 32, 4) float32   horizontally flipped image
    {split}_targets.npy       (N,) int64

Build it once with
    python -m ptflow.data build-cache --imagenet /path/to/imagenet --out $IMAGENET_CACHE_PATH
"""

from __future__ import annotations

import argparse
import os
from typing import Iterator, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from ptflow import dist


class LatentDataset(Dataset):
    def __init__(self, root: str, split: str = "train", flip: bool = True):
        path = lambda name: os.path.join(root, f"{split}_{name}.npy")
        self.moments = np.load(path("moments"), mmap_mode="r")
        self.moments_flip = np.load(path("moments_flip"), mmap_mode="r") if flip else None
        self.targets = np.load(path("targets"), mmap_mode="r")
        if len(self.moments) != len(self.targets):
            raise ValueError(f"Latent cache {root}/{split} is inconsistent")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int):
        # torch's RNG is seeded per DataLoader worker; numpy's global RNG is not.
        flip = self.moments_flip is not None and torch.rand(()).item() < 0.5
        src = self.moments_flip if flip else self.moments
        return torch.from_numpy(np.array(src[i], dtype=np.float32)), int(self.targets[i])


class SyntheticLatents(Dataset):
    """Class-dependent Gaussian latents; only for smoke tests."""

    def __init__(self, n: int = 512, shape=(8, 8, 4), num_classes: int = 10, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.means = torch.randn((num_classes, *shape), generator=g)
        self.labels = torch.randint(0, num_classes, (n,), generator=g)
        self.x = self.means[self.labels] + 0.5 * torch.randn((n, *shape), generator=g)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return self.x[i], int(self.labels[i])


def build_dataset(cfg: dict, split: str = "train") -> Dataset:
    if cfg.get("synthetic"):
        s = cfg["synthetic"]
        return SyntheticLatents(n=s.get("n", 512), shape=tuple(s.get("shape", (8, 8, 4))),
                                num_classes=s.get("num_classes", 10), seed=0 if split == "train" else 1)
    return LatentDataset(cfg["cache_path"], split, flip=(split == "train"))


def infinite_loader(ds: Dataset, batch_size: int, *, seed: int = 0, num_workers: int = 4) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    sampler = DistributedSampler(ds, num_replicas=dist.world(), rank=dist.rank(), shuffle=True, seed=seed, drop_last=True)
    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=num_workers,
                        pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        for x, y in loader:
            yield x, y
        epoch += 1


def _build_cache(imagenet: str, out: str, batch_size: int, num_workers: int) -> None:
    """Encode ImageNet train/val with SD-VAE (center crop 256), exactly as W-Flow does."""
    from PIL import Image
    from torchvision import datasets, transforms
    from tqdm import tqdm
    from ptflow.vae import load_vae

    def center_crop(img, size=256):
        while min(*img.size) >= 2 * size:
            img = img.resize(tuple(x // 2 for x in img.size), resample=Image.BOX)
        scale = size / min(*img.size)
        img = img.resize(tuple(round(x * scale) for x in img.size), resample=Image.BICUBIC)
        arr = np.array(img)
        y0, x0 = (arr.shape[0] - size) // 2, (arr.shape[1] - size) // 2
        return Image.fromarray(arr[y0:y0 + size, x0:x0 + size])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_vae(device)
    tf = transforms.Compose([transforms.Lambda(center_crop), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    os.makedirs(out, exist_ok=True)
    for split in ("train", "val"):
        ds = datasets.ImageFolder(os.path.join(imagenet, split), transform=tf)
        n = len(ds)
        mm = {k: np.lib.format.open_memmap(os.path.join(out, f"{split}_{k}.npy"), mode="w+", dtype=np.float32, shape=(n, 32, 32, 4))
              for k in ("moments", "moments_flip")}
        tg = np.lib.format.open_memmap(os.path.join(out, f"{split}_targets.npy"), mode="w+", dtype=np.int64, shape=(n,))
        loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        i = 0
        for step, (img, lab) in enumerate(tqdm(loader, desc=split)):
            img = img.to(device)
            for key, batch in (("moments", img), ("moments_flip", torch.flip(img, dims=(3,)))):
                g = torch.Generator(device=device).manual_seed(step)
                with torch.no_grad():
                    lat = vae.encode(batch).latent_dist.sample(generator=g) * 0.18215
                mm[key][i:i + len(lab)] = lat.permute(0, 2, 3, 1).cpu().numpy()
            tg[i:i + len(lab)] = lab.numpy()
            i += len(lab)
        for a in (*mm.values(), tg):
            a.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-cache")
    b.add_argument("--imagenet", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--batch-size", type=int, default=128)
    b.add_argument("--num-workers", type=int, default=8)
    a = ap.parse_args()
    _build_cache(a.imagenet, a.out, a.batch_size, a.num_workers)
