import numpy as np

def _to_positive_tail(data, tail):
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if tail == 'right':
        y = data
    elif tail == 'left':
        y = -data
    elif tail in ('abs', 'two-sided', 'magnitude'):
        y = np.abs(data)
    else:
        raise ValueError("tail must be 'right', 'left', or 'abs'")
    y = y[y > 0]
    return y

def _hill_estimator_pos(y, k, min_n=20, tol=1e-12, require_ratio_min=5):
    y = np.asarray(y)
    y = y[np.isfinite(y)]
    y = y[y > 0]
    n = len(y)
    if n < max(min_n, k+1):
        return np.nan, 0 
    y_sorted = np.sort(y)[::-1]
    top_k = y_sorted[:k]
    x_k = y_sorted[k-1]
    ratios = top_k / x_k

    
    ratios = ratios[ratios > 1.0 + tol]
    if len(ratios) < max(require_ratio_min, int(0.5*k)):
        return np.nan, len(ratios)
    alpha = len(ratios) / np.sum(np.log(ratios))
    return float(alpha), int(len(ratios))

def _bootstrap_estimates_pos(y, k, B=500, rng=None, min_n=20, tol=1e-12, require_ratio_min=5):
    rng = np.random.default_rng(rng)
    y = np.asarray(y)
    y = y[np.isfinite(y)]
    y = y[y > 0]
    n = len(y)
    if n < max(min_n, k+1):
        return np.array([])
    ests = []
    for _ in range(B):
        sample = rng.choice(y, size=n, replace=True)
        a, rcount = _hill_estimator_pos(sample, k, min_n=min_n, tol=tol, require_ratio_min=require_ratio_min)
        if np.isfinite(a):
            ests.append(a)
    return np.array(ests)

def _bootstrap_mse_pos(y, k, B=500, rng=None, min_n=20, tol=1e-12, require_ratio_min=5):
    a_hat, _ = _hill_estimator_pos(y, k, min_n=min_n, tol=tol, require_ratio_min=require_ratio_min)
    if not np.isfinite(a_hat):
        return np.inf
    ests = _bootstrap_estimates_pos(y, k, B=B, rng=rng, min_n=min_n, tol=tol, require_ratio_min=require_ratio_min)
    if len(ests) == 0:
        return np.inf
    return float(np.mean((ests - a_hat)**2))



def hill_α_estimator_Bootstrap_mse(
    data, k_min=5, k_max=None, B=1000, ci=0.95, rng=None,
    tail='auto', min_n=20, tol=1e-12, require_ratio_min=5
):
    """
    """
    tails_to_try = ['right', 'left', 'abs'] if tail == 'auto' else [tail]
    rng = np.random.default_rng(rng)

    results = []
    for t in tails_to_try:
        y = _to_positive_tail(data, t)
        n = len(y)
        if n < max(min_n, k_min+1):
            results.append((np.nan, None, np.nan, np.nan, np.inf, t,
                            {'n_pos': n, 'k_range': (None, None), 'ratio_count': 0}))
            continue

        # k-Bereich
        kmax_default = max(6, n // 4) if k_max is None else k_max
        k_min_eff = max(5, k_min)
        k_max_eff = min(kmax_default, n-1)
        if k_min_eff >= k_max_eff:
            results.append((np.nan, None, np.nan, np.nan, np.inf, t,
                            {'n_pos': n, 'k_range': (k_min_eff, k_max_eff), 'ratio_count': 0}))
            continue

        ks = list(range(k_min_eff, k_max_eff+1))
        mses = []
        # sparsamer bootstrap für k-Auswahl
        for k in ks:
            m = _bootstrap_mse_pos(
                y, k, B=min(500, max(100, B//2)), rng=rng,
                min_n=min_n, tol=tol, require_ratio_min=require_ratio_min
            )
            mses.append(m)

        
        k_opt = ks[int(np.argmin(mses))]
        mse_opt = mses[int(np.argmin(mses))]

        
        alpha_hat, ratio_count = _hill_estimator_pos(
            y, k_opt, min_n=min_n, tol=tol, require_ratio_min=require_ratio_min
        )
        ests = _bootstrap_estimates_pos(
            y, k_opt, B=B, rng=rng, min_n=min_n, tol=tol, require_ratio_min=require_ratio_min
        )
        if len(ests) >= 50:
            lo = float(np.quantile(ests, (1-ci)/2))
            hi = float(np.quantile(ests, 1-(1-ci)/2))
        else:
            lo, hi = np.nan, np.nan

        results.append((
            float(alpha_hat) if np.isfinite(alpha_hat) else np.nan,
            int(k_opt) if k_opt is not None else None,
            lo, hi,
            float(mse_opt) if np.isfinite(mse_opt) else np.inf,
            t,
            {'n_pos': n, 'k_range': (k_min_eff, k_max_eff), 'ratio_count': int(ratio_count)}
        ))

    valid = [r for r in results if np.isfinite(r[4]) and not np.isnan(r[0])]
    if not valid:
        best_idx = int(np.argmax([r[6]['n_pos'] for r in results])) if results else 0
        r = results[best_idx]
        return np.nan, None, np.nan, np.nan, np.inf, None, {'tried': results}
        
    mse_vals = [r[4] for r in valid]
    best_mse = np.min(mse_vals)
    candidates = [r for r in valid if np.isclose(r[4], best_mse)]
    if len(candidates) > 1:
        best_idx = int(np.argmax([r[6]['n_pos'] for r in candidates]))
        best = candidates[best_idx]
    else:
        best = candidates[0]

    alpha_hat, k_opt, lo, hi, mse, chosen_tail, diag = best
    diag_all = {'chosen': diag, 'tried': results}
    return alpha_hat, k_opt, lo, hi, mse, chosen_tail, diag_all

