"""Minimal torch.distributed helpers (torchrun).

Gradients are averaged by an explicit all-reduce rather than DDP: both networks
are called several times per step in different roles (generator step, frozen
proposal, HVPs), which DDP's single-forward reducer does not support.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def world() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return rank() == 0


def init() -> torch.device:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not is_dist():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        return torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    return torch.device("cpu")


def barrier() -> None:
    if is_dist():
        dist.barrier()


def cleanup() -> None:
    if is_dist():
        dist.destroy_process_group()


def allreduce_grads(module: torch.nn.Module) -> None:
    """Average parameter gradients over ranks in one flat buffer."""
    if not is_dist():
        return
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads:
        return
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat)
    flat.div_(world())
    for g, s in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
        g.copy_(s)


@torch.no_grad()
def broadcast_module(module: torch.nn.Module) -> None:
    if is_dist():
        for t in list(module.parameters()) + list(module.buffers()):
            dist.broadcast(t.data, src=0)


def mean_scalar(value: float, device: torch.device) -> float:
    if not is_dist():
        return float(value)
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.all_reduce(t)
    return float(t.item()) / world()


def all_gather_cat(x: torch.Tensor) -> torch.Tensor:
    """Concatenate equally-shaped per-rank tensors along dim 0."""
    if not is_dist():
        return x
    out = [torch.empty_like(x) for _ in range(world())]
    dist.all_gather(out, x.contiguous())
    return torch.cat(out, dim=0)
