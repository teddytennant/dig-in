"""DIG-IN: Discrete Gibbs Iterative Neural Sampler (arXiv 2610.01739).

A tiny honest slice of the method on a 2x2 open Ising lattice. The model is a
masked any-order network. Training uses the Gibbs cross-entropy of Proposition
3.1. Sampling uses any-order generation and the importance weight of the paper,
with ESS-selected tempering nu*.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

# 2x2 open grid. Sites in row-major order. Edges are right and down only.
EDGES = ((0, 1), (0, 2), (1, 3), (2, 3))
N_SITES = 4
SPIN = jnp.array([-1.0, 1.0])


def log_p_tilde(spins, beta, h):
    """Unnormalized log-density. spins is (..., 4) in {-1, +1}."""
    spins = jnp.asarray(spins, dtype=jnp.float32)
    pair = sum(spins[..., i] * spins[..., j] for i, j in EDGES)
    field = spins.sum(axis=-1)
    return beta * (pair + h * field)


def all_configs():
    """All 16 configurations, shape (16, 4), in {-1, +1}."""
    idx = jnp.arange(16)
    bits = ((idx[:, None] >> jnp.arange(4)) & 1).astype(jnp.float32)
    return bits * 2.0 - 1.0


def partition(beta, h):
    cfgs = all_configs()
    logw = log_p_tilde(cfgs, beta, h)
    return jnp.exp(jax.nn.logsumexp(logw)), logw


def gibbs_label(spins, site, beta, h):
    """Exact one-site conditional of the unnormalized target.

    Does not depend on the current value at `site`. Returns probabilities over
    {-1, +1} in that order, shape (..., 2).
    """
    spins = jnp.asarray(spins, dtype=jnp.float32)
    logs = []
    for c in (-1.0, 1.0):
        filled = spins.at[..., site].set(c)
        logs.append(log_p_tilde(filled, beta, h))
    logits = jnp.stack(logs, axis=-1)
    return jax.nn.softmax(logits, axis=-1)


def gibbs_labels_all(spins, beta, h, mask=None):
    """Conditionals at every site. mask is 1 where the site is observed.

    Returns (..., 4, 2). Masked sites get a zero row (they are not trained).
    """
    labels = jnp.stack(
        [gibbs_label(spins, d, beta, h) for d in range(N_SITES)], axis=-2
    )
    if mask is not None:
        labels = labels * (1.0 - mask)[..., None]
    return labels


def init_params(key, hidden=32):
    """MLP: concat(spins, mask) -> hidden -> hidden -> logits (4, 2)."""
    d_in = N_SITES * 2
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "w1": jax.random.normal(k1, (d_in, hidden)) * 0.1,
        "b1": jnp.zeros((hidden,)),
        "w2": jax.random.normal(k2, (hidden, hidden)) * 0.1,
        "b2": jnp.zeros((hidden,)),
        "w3": jax.random.normal(k3, (hidden, N_SITES * 2)) * 0.1,
        "b3": jnp.zeros((N_SITES * 2,)),
    }


def logits(params, spins, mask):
    """Logits over {-1, +1} at each site. Shape (..., 4, 2)."""
    x = jnp.concatenate(
        [jnp.asarray(spins, dtype=jnp.float32), jnp.asarray(mask, dtype=jnp.float32)],
        axis=-1,
    )
    h = jax.nn.gelu(x @ params["w1"] + params["b1"])
    h = jax.nn.gelu(h @ params["w2"] + params["b2"])
    out = h @ params["w3"] + params["b3"]
    return out.reshape(out.shape[:-1] + (N_SITES, 2))


def log_probs(params, spins, mask):
    return jax.nn.log_softmax(logits(params, spins, mask), axis=-1)


def spin_index(spins):
    """Map {-1, +1} to {0, 1}."""
    return ((jnp.asarray(spins) + 1.0) / 2.0).astype(jnp.int32)


def gibbs_loss(params, spins, mask, beta, h):
    """Cross-entropy of model logits against the Gibbs label, unmasked sites.

    ell_Gibbs = -sum_c h^d(c | x^{-d}) log mu(c | x_R), summed over d not in R.
    """
    target = gibbs_labels_all(spins, beta, h)
    log_mu = log_probs(params, spins, mask)
    # zero the already-revealed sites
    weight = (1.0 - mask)[..., None]
    ce = -(target * log_mu * weight).sum(axis=(-2, -1))
    denom = jnp.maximum((1.0 - mask).sum(axis=-1), 1.0)
    return (ce / denom).mean()


def ce_loss(params, spins, mask, beta, h):
    """One-hot cross-entropy against the true spin at unmasked sites.

    ell_CE = -log mu(X^d | X_R). Used only to check Proposition 3.1.
    """
    log_mu = log_probs(params, spins, mask)
    idx = spin_index(spins)
    gathered = jnp.take_along_axis(log_mu, idx[..., None], axis=-1)[..., 0]
    weight = 1.0 - mask
    ce = -(gathered * weight).sum(axis=-1)
    denom = jnp.maximum(weight.sum(axis=-1), 1.0)
    return ce / denom


def any_order_log_weight(params, spins, order, beta, h):
    """log w = log p_tilde(x) - sum_reveals log mu(x^d | x_R).

    `order` is a permutation of site indices, the reveal order. R before step
    k is order[:k]. Returns a scalar log-weight for one configuration.
    """
    spins = jnp.asarray(spins, dtype=jnp.float32)
    log_model = 0.0
    mask = jnp.zeros((N_SITES,))
    # start from a fully masked canvas; the value under the mask must not matter
    canvas = jnp.zeros((N_SITES,))
    for k in range(N_SITES):
        d = int(order[k])
        lp = log_probs(params, canvas, mask)[d]
        log_model = log_model + lp[spin_index(spins[d])]
        canvas = canvas.at[d].set(spins[d])
        mask = mask.at[d].set(1.0)
    return log_p_tilde(spins, beta, h) - log_model


def ess(weights, nu):
    """ESS(nu) = (sum w^nu)^2 / sum (w^nu)^2. weights are positive."""
    w = jnp.asarray(weights, dtype=jnp.float64)
    # stabilize: w^nu = exp(nu * log w)
    logw = jnp.log(jnp.maximum(w, 1e-300))
    tempered = jnp.exp(nu * logw - jax.nn.logsumexp(nu * logw))
    # ESS of the unnormalized tempered weights equals ESS of the normalized ones
    # times 1, so compute from the tempered values before normalizing.
    tw = jnp.exp(nu * (logw - logw.max()))
    return (tw.sum() ** 2) / (tw**2).sum()


def select_nu(weights, xi=0.5, grid=21):
    """Largest grid point in [0, 1] whose ESS is at least xi * B."""
    w = jnp.asarray(weights)
    B = w.shape[0]
    nus = jnp.linspace(0.0, 1.0, grid)
    scores = jax.vmap(lambda nu: ess(w, nu))(nus)
    ok = scores >= xi * B
    # if none pass, take 0; otherwise the largest passing nu
    idx = jnp.where(ok, jnp.arange(grid), -1).max()
    idx = jnp.maximum(idx, 0)
    return nus[idx], scores


def sample_any_order(params, key, beta, h, n, order=None):
    """Generate n configurations in a fixed or random reveal order.

    Returns spins (n, 4), positive weights (n,), and the orders used (n, 4).
    """
    if order is None:
        orders = jax.vmap(lambda k: jax.random.permutation(k, N_SITES))(
            jax.random.split(key, n)
        )
        key = jax.random.fold_in(key, 1)
    else:
        orders = jnp.broadcast_to(jnp.asarray(order), (n, N_SITES))

    def one(key, order):
        canvas = jnp.zeros((N_SITES,))
        mask = jnp.zeros((N_SITES,))
        log_model = 0.0
        for k in range(N_SITES):
            d = order[k]
            lp = log_probs(params, canvas, mask)[d]
            key, sub = jax.random.split(key)
            choice = jax.random.categorical(sub, lp)
            spin = SPIN[choice]
            log_model = log_model + lp[choice]
            canvas = canvas.at[d].set(spin)
            mask = mask.at[d].set(1.0)
        log_w = log_p_tilde(canvas, beta, h) - log_model
        return canvas, jnp.exp(log_w), order

    keys = jax.random.split(key, n)
    spins, weights, orders = jax.vmap(one)(keys, orders)
    return spins, weights, orders


def kl_q_pi(params, beta, h, n_orders=24, key=None):
    """KL(q || pi) estimated by enumerating configs under random orders.

    q(x) = E_order[ prod_k mu(x^{d_k} | x_{R_k}) ]. We average the model
    probability over `n_orders` permutations, then compare to the exact pi.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    cfgs = all_configs()
    log_pi = log_p_tilde(cfgs, beta, h)
    log_pi = log_pi - jax.nn.logsumexp(log_pi)
    orders = jax.vmap(lambda k: jax.random.permutation(k, N_SITES))(
        jax.random.split(key, n_orders)
    )

    def model_logprob(spins, order):
        canvas = jnp.zeros((N_SITES,))
        mask = jnp.zeros((N_SITES,))
        lp_sum = 0.0
        for k in range(N_SITES):
            d = order[k]
            lp = log_probs(params, canvas, mask)[d, spin_index(spins[d])]
            lp_sum = lp_sum + lp
            canvas = canvas.at[d].set(spins[d])
            mask = mask.at[d].set(1.0)
        return lp_sum

    def avg_q(spins):
        lps = jax.vmap(lambda order: model_logprob(spins, order))(orders)
        return jax.nn.logsumexp(lps) - jnp.log(n_orders)

    log_q = jax.vmap(avg_q)(cfgs)
    log_q = log_q - jax.nn.logsumexp(log_q)
    return (jnp.exp(log_q) * (log_q - log_pi)).sum()


def magnetization(spins, weights=None):
    m = jnp.asarray(spins).mean(axis=-1)
    if weights is None:
        return m.mean()
    w = jnp.asarray(weights)
    w = w / w.sum()
    return (w * m).sum()


def exact_magnetization(beta, h):
    cfgs = all_configs()
    logw = log_p_tilde(cfgs, beta, h)
    p = jax.nn.softmax(logw)
    return (p * cfgs.mean(axis=-1)).sum()


def train_step(params, opt_state, optimizer, key, beta, h, batch=64):
    """One Gibbs-loss step. Each row is a random config with a random mask."""

    def loss_fn(p):
        k1, k2 = jax.random.split(key)
        bits = jax.random.bernoulli(k1, 0.5, (batch, N_SITES))
        spins = bits.astype(jnp.float32) * 2.0 - 1.0
        # reveal a random nonempty proper subset so the label is nontrivial
        n_reveal = jax.random.randint(k2, (batch,), 0, N_SITES)
        thresh = n_reveal[:, None]
        ranks = jnp.argsort(jax.random.uniform(k2, (batch, N_SITES)), axis=-1)
        mask = (ranks < thresh).astype(jnp.float32)
        return gibbs_loss(p, spins, mask, beta, h)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def train(key, beta, h, steps=400, lr=1e-2, batch=64, hidden=32):
    key, sub = jax.random.split(key)
    params = init_params(sub, hidden=hidden)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    losses = []
    for t in range(steps):
        key, sub = jax.random.split(key)
        params, opt_state, loss = train_step(
            params, opt_state, optimizer, sub, beta, h, batch=batch
        )
        losses.append(float(loss))
    return params, losses
