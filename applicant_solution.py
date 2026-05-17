import json
import gdown

import numpy as np
from scipy.io import loadmat

from task_and_baseline import baseline, build_task_helpers

# Download the dataset
url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
downloaded_file = "challenge.mat"
gdown.download(url, downloaded_file, quiet=False)

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


def your_canceller(tx_n, rx):
    """Here is my solution"""
    
    score_filter = helpers["score_filter"]
    fit_tx = helpers["fit_tx_prediction"]

    
    n_samples, n_channels = rx.shape

    def rank1_projection(band_matrix):
        cov = band_matrix.conj().T @ band_matrix / max(1, band_matrix.shape[0])
        _, eigvecs = np.linalg.eigh(cov)
        v = eigvecs[:, -1]
        shared = band_matrix @ v
        denom = np.vdot(shared, shared) + 1e-30
        return np.column_stack(
            [
                (np.vdot(shared, band_matrix[:, ch]) / denom) * shared
                for ch in range(band_matrix.shape[1])
            ]
        )

    def band_matrix(x):
        return np.column_stack([score_filter(x[:, ch]) for ch in range(x.shape[1])])

    def covariance_blocks(mats):
        n_mats = len(mats)
        covs = np.empty((n_mats, n_mats, n_channels, n_channels), dtype=np.complex128)
        for i in range(n_mats):
            for j in range(i, n_mats):
                covs[i, j] = mats[i].conj().T @ mats[j] / mats[i].shape[0]
                covs[j, i] = (mats[i].conj().T @ mats[j] / mats[i].shape[0]).conj().T
        return covs

    def combine_covariance(blocks, coeffs):
        cov = np.zeros((n_channels, n_channels), dtype=np.complex128)
        for i, coef_i in enumerate(coeffs):
            for j, coef_j in enumerate(coeffs):
                cov += coef_i * coef_j * blocks[i, j]
        return cov

    def candidate_metric(coeffs, p0_ch, base_cov, cross_cov, band_cov, shared_cov):
        
        removed_cov = combine_covariance(band_cov, coeffs)
        cov = base_cov.copy()
        for i, coef_i in enumerate(coeffs):
            cov -= coef_i * cross_cov[i]
            cov -= coef_i * cross_cov[i].conj().T
        cov += combine_covariance(band_cov, coeffs)
        resid_cov = cov
        
        
        shared = combine_covariance(shared_cov, coeffs)

        evals, evecs = np.linalg.eigh(shared)
        lam = float(np.maximum(evals[-1].real, 0.0))
        v = evecs[:, -1]
        err_cov = shared - lam * np.outer(v, v.conj())

        removed_power = float(np.maximum(np.trace(removed_cov).real, 1e-30))
        err_power = float(np.maximum(np.trace(err_cov).real, 0.0))
        explain_ratio = 1.0 - err_power / removed_power

        resid_powers = np.maximum(np.real(np.diag(resid_cov)), 1e-30)
        err_powers = np.maximum(np.real(np.diag(err_cov)), 0.0)
        residual_guard = bool(np.all(err_powers <= 0.80 * resid_powers))

        if explain_ratio < 0.95 or not residual_guard:
            return -np.inf

        return float(np.mean(10.0 * np.log10(p0_ch / resid_powers)))

    tx_pred = fit_tx(rx)
    tx_pred_boost = band_matrix(tx_pred)
    tx_only = rx - tx_pred

    residual_band_1 = band_matrix(tx_only)
    rank1_band_1 = rank1_projection(residual_band_1)

    residual_band_2 = band_matrix(residual_band_1)
    rank1_band_2 = rank1_projection(residual_band_2)

    basis = [tx_pred, tx_pred_boost, rank1_band_1, rank1_band_2]
    rx_band = band_matrix(rx)
    p0_ch = np.maximum(np.real(np.mean(np.abs(rx_band) ** 2, axis=0)), 1e-30)
    rx_band_cov = rx_band.conj().T @ rx_band / rx_band.shape[0]

    band_basis = [band_matrix(component) for component in basis]
    band_cross = np.empty((len(basis), n_channels, n_channels), dtype=np.complex128)
    for i, band_component in enumerate(band_basis):
        band_cross[i] = rx_band.conj().T @ band_component / rx_band.shape[0]
    band_cov = covariance_blocks(band_basis)
    

    shared_basis = [band_component - fit_tx(component) for band_component, component in zip(band_basis, basis)]
    shared_cov = covariance_blocks(shared_basis)

    coarse_grids = (
        np.array([0.92, 0.96, 1.00, 1.04, 1.08]), # tx_pred scale
        np.array([0.00, 0.04, 0.08, 0.12, 0.16, 0.20]), # tx_pred_boost
        np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]), # rank1_band_1
        np.array([0.00, 0.05, 0.10, 0.15, 0.20]),  # rank1_band_2
    )

    best_coeffs = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    best_metric = candidate_metric(best_coeffs, p0_ch, rx_band_cov, band_cross, band_cov, shared_cov)

    for scale_tx in coarse_grids[0]:
        for scale_boost in coarse_grids[1]:
            for scale_rank1 in coarse_grids[2]:
                for scale_rank2 in coarse_grids[3]:
                    coeffs = np.array(
                        [scale_tx, scale_boost, scale_rank1, scale_rank2], dtype=np.float64
                    )
                    metric = candidate_metric(
                        coeffs, p0_ch, rx_band_cov, band_cross, band_cov, shared_cov
                    )
                    if metric > best_metric:
                        best_metric = metric
                        best_coeffs = coeffs

    refine_steps = np.array([0.02, 0.02, 0.05, 0.025], dtype=np.float64)
    for _ in range(2):
        refined_axes = []
        for center, step in zip(best_coeffs, refine_steps):
            refined_axes.append(np.maximum(center + step * np.array([-2, -1, 0, 1, 2]), 0.0))

        for scale_tx in refined_axes[0]:
            for scale_boost in refined_axes[1]:
                for scale_rank1 in refined_axes[2]:
                    for scale_rank2 in refined_axes[3]:
                        coeffs = np.array(
                            [scale_tx, scale_boost, scale_rank1, scale_rank2], dtype=np.float64
                        )
                        metric = candidate_metric(
                            coeffs, p0_ch, rx_band_cov, band_cross, band_cov, shared_cov
                        )
                        if metric > best_metric:
                            best_metric = metric
                            best_coeffs = coeffs

        refine_steps *= 0.5

    removed = sum(coef * component for coef, component in zip(best_coeffs, basis))
    return rx - removed

print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
