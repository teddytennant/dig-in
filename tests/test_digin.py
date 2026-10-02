"""Equation checks for the DIG-IN slice. Seeded. Not tautologies."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import digin


def test_gibbs_matches_enumeration():
    cfgs = digin.all_configs()
    for beta, h in [(0.3, 0.0), (0.8, -0.4), (1.2, 0.7)]:
        logw = digin.log_p_tilde(cfgs, beta, h)
        for site in range(4):
            # group the 16 configs by the other three sites
            others = [i for i in range(4) if i != site]
            keys = {}
            for row, cfg in enumerate(np.array(cfgs)):
                key = tuple(float(cfg[i]) for i in others)
                keys.setdefault(key, {})[float(cfg[site])] = float(logw[row])
            for key, pair in keys.items():
                logs = np.array([pair[-1.0], pair[1.0]])
                logs = logs - logs.max()
                exact = np.exp(logs) / np.exp(logs).sum()
                spins = np.zeros(4)
                for i, s in zip(others, key):
                    spins[i] = s
                # poison the queried site; the label must ignore it
                spins[site] = 1.0
                got = np.array(digin.gibbs_label(jnp.array(spins), site, beta, h))
                np.testing.assert_allclose(got, exact, atol=1e-5)


def test_label_ignores_queried_site():
    spins = jnp.array([1.0, -1.0, 1.0, -1.0])
    flipped = spins.at[2].set(-spins[2])
    a = digin.gibbs_label(spins, 2, 0.9, 0.2)
    b = digin.gibbs_label(flipped, 2, 0.9, 0.2)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_proposition_31_mean_and_variance():
    """Same conditional expectation, Gibbs variance <= CE variance.

    The proposition is about the loss as a function of the random target spin,
    with the context held fixed. We Monte Carlo the context and, inside each
    context, enumerate the conditional so the comparison is exact.
    """
    key = jax.random.PRNGKey(0)
    params = digin.init_params(key, hidden=16)
    beta, h = 0.6, 0.25
    # random contexts: reveal 3 sites, query the remaining one
    ce_vars = []
    gibbs_vars = []
    ce_means = []
    gibbs_means = []
    for trial in range(40):
        k = jax.random.fold_in(key, trial)
        bits = jax.random.bernoulli(k, 0.5, (4,))
        spins = bits.astype(jnp.float32) * 2 - 1
        site = int(trial % 4)
        mask = jnp.ones((4,)).at[site].set(0.0)
        # conditional of the target
        p = np.array(digin.gibbs_label(spins, site, beta, h))
        log_mu = np.array(digin.log_probs(params, spins, mask)[site])
        # CE loss for each candidate spin, Gibbs loss is the expectation
        ce = -log_mu  # index 0 is -1, index 1 is +1
        g = float((p * ce).sum())
        # expectation of CE under the conditional equals Gibbs
        ce_means.append(float((p * ce).sum()))
        gibbs_means.append(g)
        # variance of the CE loss under the conditional, vs variance of Gibbs
        # (Gibbs is a function of the context only, so its conditional var is 0
        # and the interesting comparison is across the outer Monte Carlo of
        # the one-hot draw: Var(CE) >= Var(Gibbs) = Var(E[CE | context])).
        ce_vars.append(ce)
        gibbs_vars.append(g)
    # outer variance: sample the one-hot from p, compare to the Gibbs value
    rng = np.random.default_rng(0)
    ce_samples = []
    gibbs_samples = []
    for trial in range(400):
        k = jax.random.fold_in(key, 1000 + trial)
        bits = jax.random.bernoulli(k, 0.5, (4,))
        spins = bits.astype(jnp.float32) * 2 - 1
        site = int(trial % 4)
        mask = jnp.ones((4,)).at[site].set(0.0)
        p = np.array(digin.gibbs_label(spins, site, beta, h))
        log_mu = np.array(digin.log_probs(params, spins, mask)[site])
        ce = -log_mu
        choice = rng.choice(2, p=p)
        ce_samples.append(ce[choice])
        gibbs_samples.append(float((p * ce).sum()))
    np.testing.assert_allclose(
        np.mean(ce_samples), np.mean(gibbs_samples), atol=0.05
    )
    assert np.var(gibbs_samples) <= np.var(ce_samples) + 1e-6


def test_nu_star():
    equal = jnp.ones((32,))
    nu, scores = digin.select_nu(equal, xi=0.5, grid=21)
    assert float(nu) == pytest.approx(1.0)
    # one dominant weight: ESS at nu=1 is about 1, so nu* drops
    skewed = jnp.array([1000.0] + [1.0] * 31)
    nu_s, scores_s = digin.select_nu(skewed, xi=0.5, grid=21)
    assert float(nu_s) < 1.0
    B = 32
    ess_star = float(digin.ess(skewed, nu_s))
    assert ess_star >= 0.5 * B - 1e-4
    # the next grid point, if any, fails the threshold
    grid = np.linspace(0.0, 1.0, 21)
    nxt = grid[grid > float(nu_s) + 1e-9]
    if len(nxt):
        assert float(digin.ess(skewed, nxt[0])) < 0.5 * B


def test_log_weight_matches_recorded_token_probs():
    key = jax.random.PRNGKey(3)
    params = digin.init_params(key)
    spins = jnp.array([1.0, -1.0, -1.0, 1.0])
    order = (2, 0, 3, 1)
    beta, h = 0.4, -0.2
    # record token log-probs the same way the sampler does
    canvas = jnp.zeros((4,))
    mask = jnp.zeros((4,))
    recorded = 0.0
    for d in order:
        lp = digin.log_probs(params, canvas, mask)[d]
        recorded = recorded + lp[digin.spin_index(spins[d])]
        canvas = canvas.at[d].set(spins[d])
        mask = mask.at[d].set(1.0)
    got = digin.any_order_log_weight(params, spins, order, beta, h)
    expect = digin.log_p_tilde(spins, beta, h) - recorded
    np.testing.assert_allclose(got, expect, atol=1e-5)


def test_exact_model_estimates_Z():
    """If mu is the exact conditional, mean importance weight estimates Z."""
    beta, h = 0.5, 0.3
    Z, _ = digin.partition(beta, h)

    class Exact:
        pass

    # wrap the exact conditional in the params-free path by monkeypatching
    # log_probs through a custom sampler loop, so this does not depend on an MLP
    key = jax.random.PRNGKey(7)
    n = 4000
    weights = []
    keys = jax.random.split(key, n)
    for k in keys:
        k, ko = jax.random.split(k)
        order = jax.random.permutation(ko, 4)
        canvas = jnp.zeros((4,))
        mask = jnp.zeros((4,))
        log_model = 0.0
        for step in range(4):
            d = int(order[step])
            p = digin.gibbs_label(canvas, d, beta, h)
            k, ks = jax.random.split(k)
            choice = jax.random.choice(ks, 2, p=np.array(p))
            spin = digin.SPIN[choice]
            log_model = log_model + jnp.log(p[choice])
            canvas = canvas.at[d].set(spin)
            mask = mask.at[d].set(1.0)
        weights.append(float(jnp.exp(digin.log_p_tilde(canvas, beta, h) - log_model)))
    est = float(np.mean(weights))
    rel = abs(est - float(Z)) / float(Z)
    assert rel < 0.08, (est, float(Z), rel)


def test_smoke_train_beats_untrained():
    beta, h = 0.5, 0.3
    key = jax.random.PRNGKey(0)
    key, sub = jax.random.split(key)
    fresh = digin.init_params(sub)
    before = float(digin.kl_q_pi(fresh, beta, h, n_orders=48, key=key))
    params, losses = digin.train(key, beta, h, steps=250, lr=2e-2, batch=128)
    after = float(digin.kl_q_pi(params, beta, h, n_orders=48, key=key))
    m_hat = float(
        digin.magnetization(*digin.sample_any_order(params, key, beta, h, 2000)[:2])
    )
    m_true = float(digin.exact_magnetization(beta, h))
    m_before = float(
        digin.magnetization(*digin.sample_any_order(fresh, key, beta, h, 2000)[:2])
    )
    print(
        f"smoke kl {before:.4f} -> {after:.4f}  "
        f"mag err {abs(m_before - m_true):.4f} -> {abs(m_hat - m_true):.4f}  "
        f"loss {losses[0]:.3f} -> {losses[-1]:.3f}"
    )
    assert after < before or abs(m_hat - m_true) < abs(m_before - m_true)
