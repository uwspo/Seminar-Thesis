#For Multithreading, speeds up the complexity
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from functools import partial

from NeuralNetwork import NN, MLP, TinyCNN
from SGD1 import stochastic_gradient_descent, rss, mse
from typing import List
from αEstimator import hill_α_estimator_Bootstrap_mse

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import os, random

n = 100000
sigma_x = 3.0
sigma = 1.0
sigma_y = 3.0
d_input = 2

x0  = torch.normal(0.0, sigma_x, size=(d_input, 1))
A   = torch.normal(0.0, sigma,   size=(n, d_input))
eps = torch.normal(0.0, sigma_y, size=(n, 1))

Y = A @ x0 + eps
X = A @ x0

η1, η2, η3, η4, η5 = 0.01, 0.3, 0.1, 0.001, 0.0003
b = 32

"""
Resets all random seeds. 
Used to ensure indenpendent Runs.
"""
def reseed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

"""
"""
def simulate_one_seed(seed: int) -> List[List[float]]:
    L = 500
    reseed(seed)
    model = MLP(sizes=(1,3,3,1), activation=nn.ReLU, add_softmax=False)
    θ0 = model.get_θ()

    x0  = torch.normal(0.0, sigma_x, size=(d_input, 1))
    A   = torch.normal(0.0, sigma,   size=(n, d_input))
    eps = torch.normal(0.0, sigma_y, size=(n, 1))

    Y = A @ x0 + eps
    X = A @ x0

    training_dataset = list(zip(X,Y))
    θ1, θ_iterates, grad_iterates = stochastic_gradient_descent(
        model, θ0, training_dataset, rss, η4, b, plot=False
    )
    return θ1

"""
"""
def simulate_parallel(random_seeds: List[int]) -> List[List[float]]:
    results = [None] * len(random_seeds)
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(simulate_one_seed, seed): idx for idx, seed in enumerate(random_seeds)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Simulating"):
            idx = futures[future]
            results[idx] = future.result()
    return results

"""
"""
def flatten_θ(θ: List[torch.tensor]):
    vecs = []
    for t in θ:
        if isinstance(t, torch.Tensor):
            arr = t.detach().cpu().numpy()
        else:
            arr = np.asarray(t)
        vecs.append(arr.reshape(-1))
    return np.concatenate(vecs)


def generate_θ_matrix(Last_θ_iterates):
    flattend_θs = []
    for θ in Last_θ_iterates:
        flattended_θ = flatten_θ(θ)
        flattend_θs.append(flattended_θ)
    return np.vstack(flattend_θs)


def project_θ_matrix_along_u(θ_matrix, u):
    u = np.asarray(u, float)
    u /= np.linalg.norm(u)
    T = θ_matrix @ u
    return T[T > 0]


def run_URV_test(projections, lambdas=(1.5, 2.0, 3.0), m_top=30, B=2000, min_pos=50, rng=None):
    x = np.asarray(projections, float)
    x = x[np.isfinite(x) & (x > 0)]
    n = x.size
    α_estimate, k_opt, lo, hi, mse, chosen_tail, diag_all = hill_α_estimator_Bootstrap_mse(projections)

    x_sorted = np.sort(x)
    if x_sorted.size == 0:
        return dict(valid=False, reason="No positive values", n=n)

    x_grid = x_sorted[-m_top:]
    x_max = x_sorted[-1]

    def empirical_cumulative_dist(y):
        idx = np.searchsorted(x_sorted, y, side="left")
        return (n - idx) / n

    deviation_max = -np.inf
    for x0 in x_grid:
        sx = empirical_cumulative_dist(x0)
        if sx <= 0:
            continue
        for λ in lambdas:
            y = λ * x0
            if y > x_max:
                continue
            sy = empirical_cumulative_dist(y)
            act_val = (sy / max(sx, 1.0 / n))
            theo_val = λ ** (-float(α_estimate))
            deviation = abs(act_val - theo_val)
            if deviation > deviation_max:
                deviation_max = deviation
    T_obs = deviation_max if np.isfinite(deviation_max) else np.nan

    if not np.isfinite(T_obs):
        return dict(valid=False, reason="Test statistic could not be calculated", n=n)

    if rng is None:
        rng = np.random.default_rng()

    def pareto_rvs(alpha, size, x_min=1.0):
        U = rng.random(size)
        return x_min * (1 - U) ** (-1.0 / alpha)

    def tail_ratio_stat_from_sample(z):
        z = np.asarray(z, float)
        z = z[np.isfinite(z) & (z > 0)]
        if z.size != n:
            return np.nan

        z_sorted = np.sort(z)
        z_max = z_sorted[-1]

        def empirical_cumulative_dist_b(y):
            j = np.searchsorted(z_sorted, y, side="left")
            return (z.size - j) / z.size

        z_grid = z_sorted[-m_top:]

        deviation_b = -np.inf
        for x0 in z_grid:
            sx = empirical_cumulative_dist_b(x0)
            if sx <= 0:
                continue
            for λ in lambdas:
                y = λ * x0
                if y > z_max:
                    continue
                sy = empirical_cumulative_dist_b(y)
                act_val = (sy / max(sx, 1.0 / z.size))
                theo_val = λ ** (-float(α_estimate))
                deviation = abs(act_val - theo_val)
                if deviation > deviation_b:
                    deviation_b = deviation
        return deviation_b if np.isfinite(deviation_b) else np.nan

    Tb = np.empty(B)
    for b in range(B):
        Zb = pareto_rvs(α_estimate, size=n, x_min=1.0)
        Tb[b] = tail_ratio_stat_from_sample(Zb)

    p_val = (1 + np.sum(Tb >= T_obs)) / (B + 1)

    return dict(
        valid=True,
        n=n,
        alpha=α_estimate,
        ci=(lo, hi),
        k=int(k_opt),
        stat=float(T_obs),
        p=float(p_val)
    )


BLUE = "#1f77b4"
rng = np.random.default_rng(1338)
seeds = rng.integers(0, 2**31 - 1, size=150, dtype=np.int64).tolist()

def simulate_one_seed_with_path(seed: int, keep_every: int = 50):
    reseed(seed)
    model = MLP(sizes=(1,3,3,1), activation=nn.ReLU, add_softmax=False)
    θ0 = model.get_θ()

    x0  = torch.normal(0.0, sigma_x, size=(d_input, 1))
    A   = torch.normal(0.0, sigma,   size=(n, d_input))
    eps = torch.normal(0.0, sigma_y, size=(n, 1))

    Y = A @ x0 + eps
    X = A @ x0

    training_dataset = list(zip(X,Y))
    θ1, θ_iterates, grad_iterates = stochastic_gradient_descent(
        model, θ0, training_dataset, rss, η4, b, plot=False
    )
    path = [θ_iterates[k] for k in range(0, len(θ_iterates), keep_every)]
    if path[-1] is not θ_iterates[-1]:
        path.append(θ_iterates[-1])
    return path


def simulate_paths_parallel(random_seeds: List[int], keep_every: int = 50):
    all_paths = [None] * len(random_seeds)
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(simulate_one_seed_with_path, seed, keep_every): idx
                   for idx, seed in enumerate(random_seeds)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Simulating paths"):
            all_paths[futures[fut]] = fut.result()
    Lmin = min(len(p) for p in all_paths)
    all_paths = [p[:Lmin] for p in all_paths]
    return all_paths


def vectorize_theta_list(theta_list):
    return np.vstack([flatten_θ(θ) for θ in theta_list])


def _emp_ccdf_sorted(x_sorted):
    n = x_sorted.size
    return (n - np.arange(n)) / n


if __name__ == "__main__":
    def _emp_ccdf(x_sorted):
        n = x_sorted.size
        idx = np.arange(n)
        return (n - idx) / n
    def _tail_ratio_grid(x, alpha_hat, lambdas=(1.5,2.0,3.0), m_top=30):
        x = np.asarray(x, float)
        x = x[np.isfinite(x) & (x > 0)]
        if x.size < 10:
            return None, None, None
        x = np.sort(x)
        m_eff = min(m_top, max(5, x.size // 3))
        xg = x[-m_eff:]
        lam_list = list(lambdas)
        grid = np.empty((len(lam_list), m_eff))
        def surv(y):
            idx = np.searchsorted(x, y, side='left')
            return (x.size - idx) / x.size
        for i, lam in enumerate(lam_list):
            for j2, x0 in enumerate(xg):
                sx = surv(x0)
                y = lam * x0
                if y > x[-1] or sx <= 0:
                    grid[i, j2] = np.nan
                else:
                    sy = surv(y)
                    r_hat = sy / max(sx, 1.0/x.size)
                    r_theo = lam ** (-alpha_hat)
                    grid[i, j2] = abs(r_hat - r_theo)
        return np.array(lam_list), xg, grid


    SHOW_CONVERGENCE = True
    if SHOW_CONVERGENCE:
        rng = np.random.default_rng(1338)
        keep_every = 30
        paths = simulate_paths_parallel(seeds, keep_every=keep_every)
        R_runs = len(paths)
        L_time = len(paths[0])

        end_thetas = [paths[r][-1] for r in range(R_runs)]
        end_mat = vectorize_theta_list(end_thetas)
        D = end_mat.shape[1]

        best_alpha = np.inf
        j_star = 0
        for j in range(D):
            xj = end_mat[:, j]
            xj = xj[np.isfinite(xj) & (xj > 0)]
            if xj.size < 30:
                continue
            res = run_URV_test(xj, rng=rng)
            if res.get("valid", False) and res["alpha"] < best_alpha:
                best_alpha = float(res["alpha"])
                j_star = j

        J = 40 
        idxs = np.unique(np.linspace(0, L_time - 1, J, dtype=int)).tolist()
        titles = [f"t={i*keep_every}" for i in idxs]

        alpha_t, p_t, tgrid = [], [], []
        for idx in idxs:
            thetas_t = [paths[r][idx] for r in range(R_runs)]
            M_t = vectorize_theta_list(thetas_t)
            x = M_t[:, j_star]
            x = x[np.isfinite(x) & (x > 0)]
            if x.size < 30:
                alpha_t.append(np.nan); p_t.append(np.nan); tgrid.append(idx * keep_every); continue
            res = run_URV_test(x, rng=rng)
            alpha_t.append(res["alpha"] if res.get("valid", False) else np.nan)
            p_t.append(res["p"] if res.get("valid", False) else np.nan)
            tgrid.append(idx * keep_every)

        plt.figure()
        plt.plot(tgrid, alpha_t, marker='o', color="blue")
        plt.axhline(2.0, linestyle=':', color='grey', linewidth=1)
        plt.xlabel("Iteration")
        plt.ylabel("Estimated Tail-Index α")
        plt.title(f"Konvergenz von est. α: Koordinate #{j_star+1}")
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(tgrid, p_t, marker='o')
        plt.axhline(0.10, linestyle=':', color='grey', linewidth=1)
        plt.ylim(0, 1.0)
        plt.xlabel("Iteration")
        plt.ylabel("URV p-Wert(t)")
        plt.title(f"URV-Test über Zeit: Koordinate #{j_star+1}")
        plt.tight_layout()
        plt.show()

        # CCDF-Subplots dynamisch (z.B. 3 Spalten)
        n = len(idxs)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows))
        axes = np.atleast_1d(axes).ravel()

        for k, idx in enumerate(idxs):
            thetas_t = [paths[r][idx] for r in range(R_runs)]
            M_t = vectorize_theta_list(thetas_t)
            x = M_t[:, j_star]
            x = x[np.isfinite(x) & (x > 0)]
            ax = axes[k]
            if x.size < 10:
                ax.set_title(f"{titles[k]} (zu wenig Daten)")
                ax.set_xlabel("x")
                if k % ncols == 0: ax.set_ylabel("CCDF")
                continue
            x = np.sort(x)
            ccdf = _emp_ccdf_sorted(x)
            ax.loglog(x, ccdf, marker='o', linestyle='None', markerfacecolor='none')
            res = run_URV_test(x, rng=rng)
            if res.get("valid", False):
                alpha_hat = float(res["alpha"])
                u = np.quantile(x, 0.90)
                c_emp = (x >= u).mean()
                xfit = x[x >= u]
                if xfit.size > 0:
                    ccdf_fit = c_emp * (xfit / u) ** (-alpha_hat)
                    ax.loglog(xfit, ccdf_fit, linewidth=2)
                ax.set_title(f"{titles[k]}: α≈{alpha_hat:.2f}, p={res['p']:.2f}")
            else:
                ax.set_title(f"{titles[k]}: Test ungültig")
            ax.set_xlabel("x")
            if k % ncols == 0:
                ax.set_ylabel("CCDF  P(T>x)")

        for k in range(n, nrows * ncols):
            fig.delaxes(axes[k])

        fig.suptitle(f"CCDF vs. Pareto-Fit über Zeit (Koordinate #{j_star+1})")
        plt.tight_layout()
        plt.show()

