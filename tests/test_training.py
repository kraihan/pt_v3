import json
import subprocess
import sys
from pathlib import Path

import torch

from ptflow.build import build_generator, build_potential
from ptflow.losses import calibration_loss, generator_loss, hutchinson_diag
from ptflow.models.potential import guided_hvp
from ptflow.schedule import BROKEN, DEGRADING, HEALTHY, PTSchedule

ROOT = Path(__file__).resolve().parents[1]


def test_controller_follows_algorithm_1():
    s = PTSchedule(eps_max=0.2, eps_min=0.1, anneal_steps=10, ess_ema=0.0, calib_min_steps=1, calib_max_steps=5)
    s.observe_calibration(resid_rel=0.05, control_ess=0.9)
    assert s.phase == "pt" and "converged" in s.calib_exit
    s.observe(0.9)
    assert s.state == HEALTHY and s.anneal_progress == 1 and s.eps() < 0.2 and s.update_theta()
    eps = s.eps()
    s.observe(0.1)
    assert s.state == DEGRADING and s.eps() == eps and s.alpha() == s.alpha_degraded and s.eta_steps() == 2
    s.observe(0.01)
    assert s.state == BROKEN and not s.update_theta() and s.eta_steps() == s.n_eta_broken
    for _ in range(20):
        s.observe(1.0)
    assert s.eps() == 0.1                                              # cosine anneal reaches eps_min
    t = PTSchedule(calib_min_steps=1, calib_max_steps=3)
    for _ in range(3):
        t.observe_calibration(resid_rel=0.9, control_ess=0.0)
    assert t.phase == "pt" and "max steps" in t.calib_exit and t.state == BROKEN


def test_calibration_makes_generator_the_prox(smoke_cfg):
    """A conservative map m(x0) = prox of a fixed quadratic; calibration must drive the residual down."""
    torch.manual_seed(0)
    pot = build_potential(smoke_cfg)
    opt = torch.optim.Adam(pot.parameters(), lr=3e-3)
    mu = torch.randn(1, 8, 8, 4)
    c = torch.zeros(16, dtype=torch.long)
    first = last = None
    for i in range(150):
        x0 = torch.randn(16, 8, 8, 4)
        m = (x0 + 0.5 * mu) / 1.5
        loss, met = calibration_loss(pot, m, x0, c, 0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or float(met["calib/resid_rel"])
        last = float(met["calib/resid_rel"])
    assert first > 0.9 and last < 0.5 * first


def test_generator_loss_full_and_detach(smoke_cfg):
    torch.manual_seed(0)
    gen, pot = build_generator(smoke_cfg), build_potential(smoke_cfg)
    x0, c, w = torch.randn(3, 8, 8, 4), torch.tensor([1, 2, 3]), torch.tensor([0.0, 0.5, 1.0])
    for mode in ("full", "detach"):
        gen.zero_grad()
        loss, met = generator_loss(pot, gen, x0, c, w, mode=mode, lambda_scale=0.1)
        loss.backward()
        assert torch.isfinite(loss) and any(p.grad is not None for p in gen.parameters())
        assert all(p.grad is None for p in pot.parameters())            # theta is frozen in step (A)
    diag = hutchinson_diag(pot, x0, c, 0.0, probes=1)
    assert torch.allclose(diag, torch.ones_like(diag))                  # phi == 0 at init -> I


def test_train_resume_and_experiments(tmp_path, wflow_ckpt):
    run = tmp_path / "run"
    base = [sys.executable, str(ROOT / "train.py"), "--config", str(ROOT / "configs/smoke.yaml"),
            "--workdir", str(run), "-o", f"init.wflow_ckpt={wflow_ckpt}"]
    subprocess.run(base, check=True, cwd=ROOT)
    status = json.loads((run / "status.json").read_text())
    assert status["phase"] == "pt" and status["pt_step"] == 3 and (run / "COMPLETED").exists()
    subprocess.run(base + ["-o", "train.total_steps=5"], check=True, cwd=ROOT)
    assert json.loads((run / "status.json").read_text())["pt_step"] == 5
    ckpt = sorted((run / "checkpoints").glob("state_*.pt"))[-1]
    for exp in ("proposals", "eps-sweep", "prox", "mismatch", "nll"):
        subprocess.run([sys.executable, str(ROOT / "experiments.py"), exp, "--ckpt", str(ckpt), "--out", str(tmp_path / "exp"),
                        "--n", "2", "--batch", "2", "--K", "4", "--ks", "2,4", "--probes", "1", "--eps-list", "0.2,0.1",
                        "--k-outer", "2", "--k-inner", "2", "--seeds", "1", "--refine", "1"], check=True, cwd=ROOT)
        assert (tmp_path / "exp" / f"{exp.replace('-', '_')}.json").exists()


def test_jacobian_audit_matches_autograd(smoke_cfg, wflow_ckpt):
    import importlib.util
    spec = importlib.util.spec_from_file_location("experiments", ROOT / "experiments.py")
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    gen = build_generator(smoke_cfg)
    from ptflow.models.generator import load_wflow_generator
    load_wflow_generator(gen, str(wflow_ckpt))
    x0, c = torch.randn(8, 8, 4), torch.tensor([3])
    J = ex.generator_jacobian(gen, x0, c, chunk=100)
    ref = torch.autograd.functional.jacobian(lambda x: gen(x[None], c, 0.0, with_scale=False)[0].flatten(), x0)
    assert torch.allclose(J.float(), ref.reshape(256, 256), atol=1e-5)


def test_toy_gap_is_order_eps(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "experiments.py"), "toy", "--out", str(tmp_path),
                    "--eps-list", "0.1,0.05,0.02,0.01"], check=True, cwd=ROOT)
    slope = json.loads((tmp_path / "toy.json").read_text())["slope_log_gap_vs_log_eps"]
    assert 0.8 < slope < 1.2
