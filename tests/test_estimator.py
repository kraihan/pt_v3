import math

import torch
import torch.nn as nn

from ptflow.estimator import draw_noise, log_kernel_over_proposal, proposal_points, tilted_estimate, weight_stats
from ptflow.likelihood import nested_log_likelihood


class Quadratic(nn.Module):
    """phi(y) = a/2 |y - mu|^2: psi_0, prox and the Laplace scale are closed-form."""

    def __init__(self, mu, a):
        super().__init__()
        self.mu, self.a = mu, a
        self.w = nn.Parameter(torch.ones(()))

    def phi(self, y, c):
        return 0.5 * self.a * self.w * (y - self.mu).flatten(1).square().sum(1)

    def null_labels(self, c):
        return c

    def exact_log_psi0(self, x, eps):
        d = x[0].numel()
        return -0.5 * d * math.log(1 + self.a) - self.a / (4 * eps * (1 + self.a)) * (x - self.mu).flatten(1).square().sum(1)

    def prox(self, x):
        return (x + self.a * self.mu) / (1 + self.a)


def test_exact_laplace_proposal_is_exact():
    torch.manual_seed(0)
    q, eps = Quadratic(torch.randn(1, 4, 4, 4), 1.5), 0.1
    x0, c = torch.randn(5, 4, 4, 4), torch.zeros(5, dtype=torch.long)
    s = torch.full_like(x0, -math.log(1 + q.a))                      # e^{-s} = diag(I + grad^2 phi)
    est = tilted_estimate(q.phi, x0, c, q.prox(x0), s, eps, K=8, alpha=0.0)
    assert torch.allclose(est.log_psi0, q.exact_log_psi0(x0, eps).double(), atol=1e-3)
    assert weight_stats(est.log_w)["control_ess"].min() > 0.999
    assert torch.allclose(est.posterior_mean(), q.prox(x0), atol=1e-4)   # T_eps = prox for a quadratic


def test_defensive_estimator_is_unbiased():
    torch.manual_seed(1)
    q, eps, n = Quadratic(torch.randn(1, 1, 2, 2), 0.3), 0.1, 20000
    x0, c = torch.randn(2, 1, 2, 2), torch.zeros(2, dtype=torch.long)
    m = q.prox(x0) + 0.1
    est = tilted_estimate(q.phi, x0.repeat(n, 1, 1, 1), c.repeat(n), m.repeat(n, 1, 1, 1), None, eps, K=8, alpha=0.3)
    mean_psi = torch.logsumexp(est.log_psi0.view(n, 2), 0) - math.log(n)
    assert (mean_psi - q.exact_log_psi0(x0, eps).double()).abs().max() < 0.01


def test_potential_gradient_flows_and_mixture_limits():
    q, eps = Quadratic(torch.zeros(1, 2, 2, 1), 1.0), 0.2
    x0, c = torch.randn(3, 2, 2, 1), torch.zeros(3, dtype=torch.long)
    est = tilted_estimate(q.phi, x0, c, q.prox(x0), None, eps, K=4, alpha=0.1)
    est.log_psi0.sum().backward()
    assert q.w.grad is not None and torch.isfinite(q.w.grad)
    z, fd = draw_noise(x0, 4, 1.0)
    assert fd.all() and torch.allclose(z[:, :2], -z[:, 2:])           # antithetic pairs
    y = proposal_points(x0, q.prox(x0), None, eps, z, fd)
    assert torch.equal(log_kernel_over_proposal(y, x0, q.prox(x0), None, eps, 1.0), torch.zeros(3, 4, dtype=torch.float64))


def test_weight_stats_extremes():
    uniform = weight_stats(torch.zeros(2, 8, dtype=torch.float64))
    degenerate = weight_stats(torch.tensor([[0.0] + [-1e4] * 7], dtype=torch.float64))
    assert torch.allclose(uniform["control_ess"], torch.ones(2, dtype=torch.float64))
    assert degenerate["control_ess"].item() < 1e-6 and degenerate["max_w"].item() > 0.999


def test_likelihood_is_normalized_in_one_dimension():
    """Thm 3.5: the model marginal integrates to one.  Exact prox + Laplace scale, large budgets."""
    torch.manual_seed(2)
    q, eps = Quadratic(torch.full((1, 1, 1, 1), 0.7), 0.8), 0.2

    def gen(y, c, w, with_scale=True):
        return q.prox(y), torch.full_like(y, -math.log(1 + q.a))

    xs = torch.linspace(-6, 6, 241).view(-1, 1, 1, 1)
    c = torch.zeros(len(xs), dtype=torch.long)
    out = nested_log_likelihood(gen, q, xs, c, eps, k_outer=512, k_inner=8, alpha=0.1)
    integral = (out["log_likelihood"].exp() * (xs[1] - xs[0]).item()).sum()
    assert abs(integral.item() - 1.0) < 0.03
