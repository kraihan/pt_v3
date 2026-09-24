import pytest
import torch

from ptflow.build import build_generator, build_potential, init_from_wflow
from ptflow.models.generator import load_wflow_generator, noise_code_table, read_wflow_state
from ptflow.models.potential import grad_phi, guided_grad, guided_hvp, prox_residual


def test_wflow_init_is_the_pretrained_sampler(smoke_cfg, wflow_ckpt):
    gen, pot = build_generator(smoke_cfg), build_potential(smoke_cfg)
    smoke_cfg["init"]["wflow_ckpt"] = str(wflow_ckpt)
    info = init_from_wflow(gen, pot, smoke_cfg)
    assert info["source_step"] == 200000
    state, _ = read_wflow_state(str(wflow_ckpt))
    table = noise_code_table(state)
    assert torch.allclose(gen.code_bias, table[:, 0].sum(0))        # fixed:0 folds code 0 of every coordinate
    x0, c = torch.randn(3, 8, 8, 4), torch.tensor([1, 2, 3])
    m, s = gen(x0, c, 0.5)
    assert torch.isfinite(m).all() and m.abs().max() > 0
    assert torch.equal(s, torch.zeros_like(s))                     # S = I at step 0
    g, phi = grad_phi(pot, x0, c)
    assert phi.abs().max() == 0 and g.abs().max() == 0              # phi == 0 -> prox = identity


def test_noise_code_modes_and_strict_architecture(smoke_cfg, wflow_ckpt):
    gen = build_generator(smoke_cfg)
    load_wflow_generator(gen, str(wflow_ckpt), noise_code="mean")
    table = noise_code_table(read_wflow_state(str(wflow_ckpt))[0])
    assert torch.allclose(gen.code_bias, table.mean(1).sum(0))
    bad = dict(smoke_cfg["generator"], depth=3)
    with pytest.raises(ValueError):
        load_wflow_generator(build_generator({"generator": bad}), str(wflow_ckpt))


def test_potential_derivatives_match_autograd(smoke_cfg):
    torch.manual_seed(0)
    pot = build_potential(smoke_cfg)
    with torch.no_grad():
        for p in pot.parameters():
            p.add_(0.05 * torch.randn_like(p))
    x, c, w = torch.randn(2, 8, 8, 4), torch.tensor([1, 4]), torch.tensor([0.0, 0.7])
    g = guided_grad(pot, x, c, w)
    xr = x.clone().requires_grad_(True)
    phi_w = (1 + w) * pot.phi(xr, c) - w * pot.phi(xr, pot.null_labels(c))
    (g_ref,) = torch.autograd.grad(phi_w.sum(), xr)
    assert torch.allclose(g, g_ref, atol=1e-5)
    v = torch.randn_like(x)
    hv = guided_hvp(pot, x, c, w, v)
    h = 1e-3
    fd = (guided_grad(pot, x + h * v, c, w) - guided_grad(pot, x - h * v, c, w)) / (2 * h)
    assert torch.allclose(hv, fd, atol=2e-3, rtol=2e-2)
    r = prox_residual(pot, x, x, c, w)
    assert torch.allclose(r, g, atol=1e-6)
    assert all(p.grad is None for p in pot.parameters())            # derivatives never touch theta.grad
