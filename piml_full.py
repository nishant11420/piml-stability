# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════╗
║   Physics-Informed Machine Learning – Stability Lobe Diagram            ║
║   Complete integrated pipeline with Excel boundary data support         ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Papers implemented:                                                    ║
║   [1] Chen et al. (2021)  – FDM + Bayesian inference (MAP + Laplace)   ║
║   [2] Karandikar et al. (2022) – Bayesian ML + E[I(MRR)] closed-loop  ║
║   [3] St. John et al. (2023)  – RF + RFE acoustic chatter classifier   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  HOW TO RUN                                                             ║
║                                                                         ║
║  Quick demo (no FDM, no GPU, just needs numpy + scikit-learn):          ║
║    python piml_full.py --mode demo                                      ║
║                                                                         ║
║  With your Excel boundary file:                                         ║
║    python piml_full.py --mode excel                                     ║
║           --boundary stability_boundary3__1_.xlsx                      ║
║                                                                         ║
║  With FDM + Excel boundary (full physics):                              ║
║    python piml_full.py --mode full                                      ║
║           --boundary stability_boundary3__1_.xlsx                      ║
║           --fdm FDM_function.py                                         ║
║                                                                         ║
║  Required packages:                                                     ║
║    pip install numpy scipy scikit-learn pandas openpyxl matplotlib      ║
║  Optional (for full physics mode):                                      ║
║    pip install torch sobol_seq librosa soundfile                        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import argparse
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.rcParams["mathtext.fontset"] = "cm"
plt.rc("font", family="serif", size=11)

# ── optional heavy dependencies ────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import RFE
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import roc_auc_score
    _SKLEARN = True
except ImportError:
    _SKLEARN = False
    warnings.warn("scikit-learn not installed. Run: pip install scikit-learn")

try:
    import sobol_seq
    _SOBOL = True
except ImportError:
    _SOBOL = False

try:
    import librosa
    _LIBROSA = True
except ImportError:
    _LIBROSA = False

# ── FDM physics module (optional, from Chen et al. repo) ──────────────────
_FDM_MODULE = None


def _try_load_fdm(path=None):
    global _FDM_MODULE
    try:
        import importlib.util
        search = [path, "FDM_function.py", "FDM_function"] if path else \
                 ["FDM_function.py", "FDM_function"]
        for p in search:
            if p and os.path.exists(str(p)):
                spec = importlib.util.spec_from_file_location("FDM_function", p)
                mod  = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _FDM_MODULE = mod
                return True
        import FDM_function as m
        _FDM_MODULE = m
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# ── SECTION 1: Excel boundary adapter ─────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

class StabilityBoundary:
    """
    Loads the Excel stability boundary file (rpm, doc) and provides:
      - oracle(ss, ap)     – True if (ss, ap) is stable
      - interp(ss)         – critical ap at spindle speed ss (mm)
      - make_experiments() – labelled stable/unstable scatter for MAP
      - plot_overlay()     – returns (ss_arr, ap_arr) for SLD overlay
    """

    def __init__(self, path: str):
        df = pd.read_excel(path)

        # accept 'rpm'/'doc' or 'ss'/'ap' or 'spindle_speed'/'depth' etc.
        col_ss = next((c for c in df.columns
                       if c.lower() in ("rpm","ss","spindle_speed","speed")), df.columns[0])
        col_ap = next((c for c in df.columns
                       if c.lower() in ("doc","ap","depth","axial_depth",
                                        "doc_m","ap_mm","ap_m")), df.columns[1])

        df = df[[col_ss, col_ap]].dropna().sort_values(col_ss)
        ss_raw = df[col_ss].values.astype(float)
        ap_raw = df[col_ap].values.astype(float)

        # auto-detect units: if median < 0.1 assume metres → convert to mm
        if np.median(ap_raw) < 0.1:
            ap_raw = ap_raw * 1000.0

        self.ss  = ss_raw
        self.ap  = ap_raw
        self._fn = interp1d(ss_raw, ap_raw, kind="linear",
                            bounds_error=False,
                            fill_value=(ap_raw[0], ap_raw[-1]))

        self.ss_min = float(ss_raw.min())
        self.ss_max = float(ss_raw.max())
        self.ap_min = float(ap_raw.min())
        self.ap_max = float(ap_raw.max())

        print(f"  Boundary loaded: {len(df)} points  "
              f"rpm {self.ss_min:.0f}–{self.ss_max:.0f}  "
              f"ap {self.ap_min:.2f}–{self.ap_max:.2f} mm")

    def critical_ap(self, ss: float) -> float:
        """Return the stability boundary depth (mm) at spindle speed ss (rpm)."""
        return float(self._fn(ss))

    def oracle(self, ss: float, ap: float) -> bool:
        """True = stable (ap is below the boundary at this spindle speed)."""
        return ap < self.critical_ap(ss)

    def make_experiments(self,
                          ss_range: tuple = None,
                          n_stable:   int = 25,
                          n_chatter:  int = 25,
                          margin_mm: float = 0.2,
                          rng: np.random.Generator = None) -> list:
        """
        Sample labelled experiment points from below (stable) and above
        (unstable) the boundary.

        Parameters
        ----------
        ss_range   : (ss_min, ss_max) rpm – defaults to file range
        n_stable   : number of stable points to generate
        n_chatter  : number of chatter points to generate
        margin_mm  : keep points at least this far from the boundary
        rng        : numpy RNG for reproducibility

        Returns
        -------
        List of dicts: {ss, ap, label}
          label 0 = stable, 1 = chatter
        """
        if rng is None:
            rng = np.random.default_rng(42)
        lo, hi = ss_range if ss_range else (self.ss_min, self.ss_max)

        experiments = []

        # stable points: ap = rand * (boundary - margin)
        for _ in range(n_stable):
            ss  = rng.uniform(lo, hi)
            bnd = self.critical_ap(ss)
            ap  = rng.uniform(self.ap_min * 0.5, max(bnd - margin_mm, self.ap_min * 0.5))
            experiments.append({"ss": float(ss), "ap": float(ap),
                                 "label": 0, "surrogate_key": None})

        # chatter points: ap = boundary + margin + rand
        for _ in range(n_chatter):
            ss  = rng.uniform(lo, hi)
            bnd = self.critical_ap(ss)
            ap  = bnd + margin_mm + rng.uniform(0.05, 0.8)
            experiments.append({"ss": float(ss), "ap": float(ap),
                                 "label": 1, "surrogate_key": None})

        n_s = sum(1 for e in experiments if e["label"] == 0)
        n_c = sum(1 for e in experiments if e["label"] == 1)
        print(f"  Generated {n_s} stable + {n_c} chatter experiment points")
        return experiments

    def plot_overlay(self, ss_grid: np.ndarray) -> tuple:
        """Return (ss_arr, ap_arr) for plotting over the SLD."""
        ap_arr = np.array([self.critical_ap(s) for s in ss_grid])
        return ss_grid, ap_arr


# ═══════════════════════════════════════════════════════════════════════════
# ── SECTION 2: FDM analytic placeholder ───────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def _fdm_placeholder(w: np.ndarray, ss_grid: np.ndarray,
                      ap_grid_m: np.ndarray) -> np.ndarray:
    """
    Fast analytic placeholder when FDM_function.py is unavailable.
    Returns (n_grid, 3) array: [ss, ap_m, spectral_radius].
    Matches the output format of FDM_function.FDM 'SampleAtAllGridPoint'.
    """
    rows = []
    nat_freq = w[0] * 2 * np.pi          # rad/s
    stiff    = w[4] * 1e6                 # N/m
    Kt       = w[6] * 1e8                 # N/m²
    for ss in ss_grid:
        tau = 60.0 / ss / 2              # tooth period (2-flute assumed)
        for ap in ap_grid_m:
            # simplified quasi-static approximation
            lam = (ap * Kt) / (stiff + 1e-12) * (nat_freq * tau / (2 * np.pi))
            lam = float(np.clip(lam * 1.5e4, 0.05, 6.0))
            rows.append([ss, ap, lam])
    return np.array(rows)


def run_fdm(w: np.ndarray, purpose: str = "SampleAtAllGridPoint") -> np.ndarray:
    """
    Compute spectral radius grid using FDM_function.py if available,
    otherwise fall back to the analytic placeholder.

    Parameters
    ----------
    w       : (8,) [wx_Hz, wy_Hz, cx, cy, kx_MN/m, ky_MN/m, Kt_100MPa, Kn_100MPa]
    purpose : 'SampleForAgent' or 'SampleAtAllGridPoint'

    Returns
    -------
    data : (n_grid, 3) [ss_rpm, ap_m, spectral_radius]
    """
    if _FDM_MODULE is not None:
        _, _, _, data = _FDM_MODULE.FDM(purpose,
            w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7])
        return np.array(data)

    # placeholder grids matching FDM defaults
    if purpose == "SampleForAgent":
        ss  = np.linspace(4800, 13200, 85)
        ap  = np.linspace(0.25e-3, 3.5e-3, 27)
    else:
        ss  = np.linspace(4800, 13200, 101)
        ap  = np.linspace(0.0, 4e-3, 51)

    return _fdm_placeholder(w, ss, ap)


# ═══════════════════════════════════════════════════════════════════════════
# ── SECTION 3: Surrogate neural network (Chen et al. §3.3.2) ──────────────
# ═══════════════════════════════════════════════════════════════════════════

def _build_surrogate_class():
    if not _TORCH:
        return None

    class _Surrogate(nn.Module):
        """MLP  8 → 20 → 40 → 40 → 1  predicting spectral radius λ."""
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(8, 20), nn.Sigmoid(),
                nn.Linear(20, 40), nn.Sigmoid(),
                nn.Linear(40, 40), nn.ReLU(),
                nn.Linear(40, 1),  nn.ReLU())
        def forward(self, x):
            return self.net(x)
        def loss(self, pred, y):
            return torch.norm(pred - y) / (y.size(0) + 1e-12)

    return _Surrogate

_SurrogateClass = _build_surrogate_class()


def _sobol_samples(mu: np.ndarray, sigma: np.ndarray, n: int,
                   rng: np.random.Generator = None) -> np.ndarray:
    """Sobol (or fallback normal) samples from N(mu, diag(σ²))."""
    if _SOBOL:
        z = sobol_seq.i4_sobol_generate_std_normal(8, n)
    else:
        rng = rng or np.random.default_rng(0)
        z   = rng.standard_normal((n, 8))
    return z * sigma[None, :] + mu[None, :]


def build_surrogate_dataset(mu: np.ndarray, sigma: np.ndarray,
                             n_samples: int = 100,
                             purpose:   str = "SampleAtAllGridPoint",
                             verbose:   bool = True) -> np.ndarray:
    """
    Sample n_samples parameter vectors from N(mu, σ²) and compute
    FDM spectral radius at every grid point.

    Returns
    -------
    all_data : (n_samples × n_grid, 11)
               w[0..7], ss_rpm, ap_m, spectral_radius
    """
    samples = _sobol_samples(mu, sigma, n_samples)
    all_rows = []
    for i, w in enumerate(samples):
        if verbose and (i % 20 == 0 or i == n_samples - 1):
            print(f"  FDM sample {i+1}/{n_samples}")
        sr_arr = run_fdm(w, purpose)          # (n_grid, 3)
        n_grid = len(sr_arr)
        w_rep  = np.tile(w, (n_grid, 1))
        all_rows.append(np.hstack([w_rep, sr_arr]))
    return np.vstack(all_rows)


def train_surrogate(data: np.ndarray, epochs: int = 1000,
                    lr: float = 0.01) -> "_SurrogateClass":
    """
    Train one surrogate MLP on (w, λ) data for a single grid point.

    Parameters
    ----------
    data   : (n_samples, 11) – columns w[0..7], ss, ap, λ
    epochs : training iterations
    lr     : Adam learning rate

    Returns
    -------
    trained model (eval mode, or None if torch unavailable)
    """
    if not _TORCH:
        return None

    X     = data[:, :8].copy()
    X[:, 0:2] /= 1000.0               # Hz → kHz for numerical stability
    y     = data[:, 10]

    X_t   = torch.tensor(X, dtype=torch.float32)
    y_t   = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    model = _SurrogateClass()
    opt   = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        pred = model(X_t)
        loss = model.loss(pred, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    return model


def train_all_surrogates(all_data: np.ndarray, n_samples: int,
                          n_grid: int, epochs: int = 1000,
                          verbose: bool = True) -> dict:
    """
    Train one surrogate per grid point. Returns dict {grid_idx: model}.
    Grid points are indexed row-major: i_ss * n_ap + i_ap.
    """
    surrogates = {}
    for i in range(n_grid):
        rows = [all_data[n_grid * j + i] for j in range(n_samples)
                if n_grid * j + i < len(all_data)]
        data_i = np.vstack(rows)
        surrogates[i] = train_surrogate(data_i, epochs=epochs, lr=0.01)
        if verbose and (i % max(1, n_grid // 10) == 0 or i == n_grid - 1):
            ss = data_i[0, 8]
            ap = data_i[0, 9] * 1000
            print(f"  Surrogate {i+1}/{n_grid}  "
                  f"Ω={ss:.0f} rpm  ap={ap:.2f} mm")
    return surrogates


# ═══════════════════════════════════════════════════════════════════════════
# ── SECTION 4: Bayesian posterior inference (Chen et al. §3.2–3.3) ────────
# ═══════════════════════════════════════════════════════════════════════════

def _chatter_prob(lam, a: float = 4.0):
    """φ(λ) = 1 / (1 + exp(-a*(λ-1)))  – logistic stability function."""
    return 1.0 / (1.0 + torch.exp(torch.tensor(-a, dtype=lam.dtype) * (lam - 1.0)))


def _log_prior(w, mu0, sig0_inv):
    """Gaussian log-prior: -0.5 * (w-μ₀)ᵀ Σ₀⁻¹ (w-μ₀)."""
    d = w - mu0
    return (-0.5 * torch.mm(d.t(), torch.mm(sig0_inv, d))).squeeze()


def _negative_log_posterior(w, mu0, sig0_inv, experiments, surrogates,
                              a: float = 4.0,
                              wt_chatter: float = 1.5):
    """Returns (neg_log_posterior, log_prior) tensors."""
    lp = _log_prior(w, mu0, sig0_inv)
    ll = torch.tensor(0.0, dtype=w.dtype)

    n  = len(experiments)
    for exp in experiments:
        key   = exp.get("surrogate_key") or exp.get("_key")
        model = surrogates.get(key) if key else None
        if model is None:
            continue
        # normalise frequencies before passing to surrogate
        w_norm    = w.clone().float()
        w_norm[:2] = w_norm[:2] / 1000.0
        lam_val   = model(w_norm.t())
        phi       = _chatter_prob(lam_val, a).clamp(1e-7, 1 - 1e-7)
        y = float(exp["label"])
        wt = wt_chatter if y == 1 else 1.0
        ll = ll + wt / n * (y * torch.log(phi) + (1 - y) * torch.log(1 - phi))

    posterior = -(lp + ll.squeeze())
    return posterior, lp


def map_inference(mu_raw: np.ndarray, sigma_raw: np.ndarray,
                   experiments: list, surrogates: dict,
                   lr: float = 0.002, n_epochs: int = 400,
                   a: float = 4.0, verbose: bool = True
                   ) -> tuple:
    """
    MAP estimation + Laplace covariance (Chen et al. §3.3).

    Parameters
    ----------
    mu_raw    : (8,) prior mean in original units [Hz, -, N/m, MPa]
    sigma_raw : (8,) prior std
    experiments : labelled cuts with 'surrogate_key' set
    surrogates  : dict {key: trained model}
    lr, n_epochs, a : optimisation hyperparameters
    verbose   : print every 50 steps

    Returns
    -------
    (w_star_norm, Sigma_star)
      w_star_norm : (8,) MAP mode in normalised units (freq in kHz)
      Sigma_star  : (8,8) posterior covariance (inflation=0.01 applied)
    """
    if not _TORCH:
        print("  PyTorch unavailable – returning prior mean as posterior.")
        w_n   = mu_raw.copy(); w_n[:2] /= 1000.0
        Sigma = np.diag((sigma_raw * 0.2) ** 2) * 0.01
        return w_n, Sigma

    # normalise
    mu_n   = mu_raw.copy();   mu_n[:2]   /= 1000.0
    sig_n  = sigma_raw.copy(); sig_n[:2] /= 1000.0

    mu0    = torch.tensor(mu_n.reshape(-1, 1),       dtype=torch.double)
    sig0   = torch.tensor(np.diag(sig_n ** 2),       dtype=torch.double)
    sig0_i = torch.tensor(np.diag(1.0 / sig_n ** 2), dtype=torch.double)

    w = torch.tensor(mu0.float(), requires_grad=True)
    opt = torch.optim.Adagrad([w], lr=lr)

    for step in range(n_epochs):
        post, prior = _negative_log_posterior(
            w, mu0.float(), sig0_i.float(), experiments, surrogates, a)
        opt.zero_grad()
        post.backward(retain_graph=True)
        opt.step()
        if verbose and step % 50 == 0:
            print(f"  MAP step {step:4d}  posterior={post.item():.4f}")

    w_star = w.detach().numpy().flatten()

    # ── Laplace covariance via auto-diff Hessian ──
    print("  Computing Laplace covariance...")
    xt = torch.tensor(w_star.reshape(-1, 1),
                      dtype=torch.double, requires_grad=True)
    F, _ = _negative_log_posterior(
        xt.float(), mu0.float(), sig0_i.float(),
        experiments, surrogates, a)
    dydx = torch.autograd.grad(F, xt, create_graph=True, retain_graph=True)[0]
    rows = []
    for g in dydx.squeeze():
        row = torch.autograd.grad(g, xt, retain_graph=True)[0]
        rows.append(row.detach().squeeze())
    H = torch.stack(rows).numpy()
    try:
        Sigma = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        Sigma = np.linalg.pinv(H)

    return w_star, 0.01 * Sigma  # variance inflation factor = 0.01


# ═══════════════════════════════════════════════════════════════════════════
# ── SECTION 5: Bayesian stability map + E[I(MRR)] (Karandikar et al.) ─────
# ═══════════════════════════════════════════════════════════════════════════

class BayesianStabilityMap:
    """
    Grid-based Bayesian probability map  p(stable | Ω, ap).

    Prior can be set from FDM spectral radius or a simple linear decay.
    Updates via Bayes' theorem after each test result.
    Spatial correlation is modelled via Gaussian likelihood kernels.
    """

    def __init__(self, ss_grid: np.ndarray, ap_grid: np.ndarray,
                 prior: np.ndarray = None,
                 bw_ss: float = 500.0, bw_ap: float = 0.5):
        self.ss = ss_grid
        self.ap = ap_grid
        self.bw_ss = bw_ss
        self.bw_ap = bw_ap
        self.SS, self.AP = np.meshgrid(ss_grid, ap_grid)

        if prior is None:
            # uninformed: p(stable) decreases linearly with depth
            p_ap  = 1.0 - 0.95 * ap_grid / (ap_grid.max() + 1e-12)
            prior = np.tile(p_ap[:, None], (1, len(ss_grid)))

        self._p = np.clip(prior, 1e-6, 1 - 1e-6).copy()
        self.history = []           # list of (ss, ap, is_stable)

    @property
    def p_stable(self):   return self._p.copy()
    @property
    def p_chatter(self):  return 1.0 - self._p

    def update(self, ss: float, ap: float, is_stable: bool):
        """Bayesian update after one test result."""
        w_ss = np.exp(-0.5 * ((self.SS - ss) / self.bw_ss) ** 2)
        w_ap = np.exp(-0.5 * ((self.AP - ap) / self.bw_ap) ** 2)
        w    = w_ss * w_ap

        if is_stable:
            monotone = np.where(self.AP <= ap, 1.0, 0.6)
            lhood    = np.clip(0.5 + 0.5 * w * monotone, 1e-6, 1 - 1e-6)
        else:
            monotone = np.where(self.AP >= ap, 0.1, 0.9)
            lhood    = np.clip(monotone + (0.5 - monotone) * w, 1e-6, 1 - 1e-6)

        lhood_not = 1.0 - lhood
        num   = lhood * self._p
        denom = num + lhood_not * (1.0 - self._p)
        self._p = np.clip(num / (denom + 1e-12), 1e-6, 1 - 1e-6)
        self.history.append((ss, ap, is_stable))

    def batch_update(self, tests: list):
        for ss, ap, stable in tests:
            self.update(ss, ap, stable)

    def physics_prior_from_fdm(self, eig_grid: np.ndarray, a: float = 4.0):
        """Overwrite prior with FDM spectral radius map."""
        phi = 1.0 / (1.0 + np.exp(-a * (eig_grid - 1.0)))
        self._p = np.clip(1.0 - phi, 1e-6, 1 - 1e-6)


class TestSelector:
    """
    E[I(MRR)] active learning strategy (Karandikar et al. 2022 §2).

    E[I(MRR)]_g = p(stable_g) × (MRR_g − MRR_prior) / MRR_prior

    With optional safety floor: if selected ap has p(stable) < threshold,
    reduce ap at that speed to the highest safe depth (§6.1 modified criterion).
    """

    def __init__(self, bsm: BayesianStabilityMap,
                 min_stability: float = 0.0,
                 convergence_ei: float = 0.05):
        self.bsm            = bsm
        self.min_stability  = min_stability
        self.convergence_ei = convergence_ei
        self._mrr_prior     = float(bsm.ss.min() * bsm.ap[0] + 1e-12)

    def expected_improvement(self) -> np.ndarray:
        SS, AP = self.bsm.SS, self.bsm.AP
        mrr    = SS * AP
        p      = self.bsm.p_stable
        ei     = p * (mrr - self._mrr_prior) / (self._mrr_prior + 1e-12)
        return np.where(mrr > self._mrr_prior, ei, 0.0)

    def next_test(self) -> tuple:
        """Returns (ss, ap, converged)."""
        ei        = self.expected_improvement()
        converged = float(np.max(ei)) < self.convergence_ei

        idx          = np.argmax(ei)
        i_ap, i_ss   = np.unravel_index(idx, ei.shape)
        ss_sel       = float(self.bsm.ss[i_ss])
        ap_sel       = float(self.bsm.ap[i_ap])

        if self.min_stability > 0:
            col  = self.bsm.p_stable[:, i_ss]
            safe = np.where(col >= self.min_stability)[0]
            ap_sel = float(self.bsm.ap[safe[-1]]) if len(safe) else float(self.bsm.ap[0])

        return ss_sel, ap_sel, converged

    def record_stable(self, ss: float, ap: float):
        mrr = ss * ap
        if mrr > self._mrr_prior:
            self._mrr_prior = mrr


def run_closed_loop(bsm: BayesianStabilityMap,
                    sel: TestSelector,
                    oracle: callable,
                    max_tests: int = 20,
                    verbose:   bool = True) -> list:
    """
    Simulate the closed-loop automated stability testing loop.

    oracle(ss, ap) → bool: True = stable.
    Returns list of result dicts.
    """
    results = []
    for t in range(1, max_tests + 1):
        ss, ap, converged = sel.next_test()
        is_stable = bool(oracle(ss, ap))

        bsm.update(ss, ap, is_stable)
        if is_stable:
            sel.record_stable(ss, ap)

        results.append({"test": t, "ss": ss, "ap": ap,
                         "stable": is_stable, "ei_max": float(np.max(sel.expected_improvement()))})

        if verbose:
            tag = "stable  " if is_stable else "CHATTER "
            print(f"  Test {t:2d}: Ω={ss:7.0f} rpm  ap={ap:.2f} mm  {tag}")

        if converged:
            if verbose:
                print(f"\n  Converged at test {t}. "
                      f"Optimal: Ω={ss:.0f} rpm, ap={ap:.2f} mm")
            break
    return results


# ═══════════════════════════════════════════════════════════════════════════
# ── SECTION 6: Monte Carlo probabilistic SLD (Chen et al. §3.4) ──────────
# ═══════════════════════════════════════════════════════════════════════════

def monte_carlo_sld(w_star:    np.ndarray,
                    Sigma:     np.ndarray,
                    ss_grid:   np.ndarray,
                    ap_grid:   np.ndarray,
                    surrogates: dict,
                    n_mc:      int = 200,
                    a:         float = 4.0,
                    rng:       np.random.Generator = None,
                    verbose:   bool = True) -> np.ndarray:
    """
    Draw n_mc samples from N(w_star, Σ) and compute P_c at every grid point.

    Uses surrogate models when available; falls back to analytic placeholder.

    Returns
    -------
    p_chatter : (n_ap, n_ss) array of chatter probabilities
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n_ap, n_ss = len(ap_grid), len(ss_grid)
    samples    = rng.multivariate_normal(w_star, Sigma, size=n_mc)
    chatter    = np.zeros((n_ap, n_ss), dtype=int)

    for k, w_k in enumerate(samples):
        if verbose and (k % max(1, n_mc // 5) == 0 or k == n_mc - 1):
            print(f"  MC sample {k+1}/{n_mc}")

        if surrogates and _TORCH:
            w_t = torch.tensor(w_k, dtype=torch.float32).unsqueeze(0)
            for i_ap in range(n_ap):
                for i_ss in range(n_ss):
                    model = surrogates.get(i_ss * n_ap + i_ap)
                    if model is None:
                        continue
                    w_norm    = w_t.clone()
                    w_norm[0, :2] /= 1000.0
                    with torch.no_grad():
                        lam = float(model(w_norm).squeeze())
                    chatter[i_ap, i_ss] += int(lam > 1.0)
        else:
            # no surrogates: use FDM / placeholder directly
            sr_arr = run_fdm(w_k, "SampleAtAllGridPoint")
            for row in sr_arr:
                ss_v, ap_v, lam_v = row[0], row[1] * 1000, row[2]
                i_ss_idx = np.argmin(np.abs(ss_grid - ss_v))
                i_ap_idx = np.argmin(np.abs(ap_grid - ap_v))
                chatter[i_ap_idx, i_ss_idx] += int(lam_v > 1.0)

    return np.clip(chatter / n_mc, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# ── SECTION 7: Acoustic RF classifier (St. John et al. 2023) ─────────────
# ═══════════════════════════════════════════════════════════════════════════

def _describe(x: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}_mean":   float(np.mean(x)),
        f"{prefix}_std":    float(np.std(x)),
        f"{prefix}_min":    float(np.min(x)),
        f"{prefix}_max":    float(np.max(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_skew":   float(skew(x)),
        f"{prefix}_kurt":   float(kurtosis(x)),
        f"{prefix}_range":  float(np.ptp(x)),
    }


def extract_features(y_audio: np.ndarray, sr: int) -> dict:
    """
    Extract 152 features from a trimmed audio segment.
    Requires librosa. Raises ImportError if not installed.
    """
    if not _LIBROSA:
        raise ImportError("librosa required: pip install librosa soundfile")

    feats = {}
    # spectral centroids + derivatives
    centroids = librosa.feature.spectral_centroid(y=y_audio, sr=sr)[0]
    feats.update(_describe(centroids,              "centroids"))
    feats.update(_describe(librosa.feature.delta(centroids), "centroids_delta"))
    feats.update(_describe(librosa.feature.delta(centroids, order=2), "centroids_accel"))
    # rolloff
    feats.update(_describe(librosa.feature.spectral_rolloff(y=y_audio, sr=sr)[0], "rolloff"))
    # bandwidths
    for o in (2, 3, 4):
        bw = librosa.feature.spectral_bandwidth(y=y_audio, sr=sr, p=o)[0]
        feats.update(_describe(bw, f"bandwidth_{o}"))
    bw2 = librosa.feature.spectral_bandwidth(y=y_audio, sr=sr, p=2)[0]
    bw4 = librosa.feature.spectral_bandwidth(y=y_audio, sr=sr, p=4)[0]
    feats["bandwidth_range_min"] = float(np.min(bw4) - np.min(bw2))
    # harmonic / percussive
    yh, yp = librosa.effects.hpss(y_audio)
    feats.update(_describe(librosa.feature.rms(y=yh)[0], "harmonics"))
    feats.update(_describe(librosa.feature.rms(y=yp)[0], "perpetual_shock"))
    _, bf = librosa.beat.beat_track(y=y_audio, sr=sr)
    bi = np.diff(bf).astype(float) if len(bf) > 1 else np.zeros(4)
    feats.update(_describe(bi, "beat_track"))
    # top-10 peaks
    from scipy.signal import find_peaks
    N    = len(y_audio)
    fft  = np.abs(np.fft.rfft(y_audio * np.hanning(N)))
    freqs = np.fft.rfftfreq(N, d=1.0 / sr)
    pidx, _ = find_peaks(fft, height=0.0, distance=int(sr * 0.01))
    pidx = pidx[np.argsort(fft[pidx])[::-1][:10]]
    while len(pidx) < 10: pidx = np.append(pidx, 0)
    pmag  = fft[pidx];  pfreq = freqs[pidx]
    pnorm = pmag / (pmag[0] + 1e-12)
    pgap  = pfreq - pfreq[0]
    for k in range(10):
        feats[f"peak_{k+1}_magnitude"]      = float(pmag[k])
        feats[f"peak_{k+1}_frequency"]      = float(pfreq[k])
        feats[f"peak_{k+1}_norm_magnitude"] = float(pnorm[k])
        feats[f"peak_{k+1}_freq_gap"]       = float(pgap[k])
    feats.update(_describe(pmag,  "peak_mag"))
    feats.update(_describe(pfreq, "peak_freq"))
    feats.update(_describe(pnorm, "peak_norm_mag"))
    feats.update(_describe(pgap,  "peak_freq_gap"))
    return feats


def prune_features(X: np.ndarray, names: list,
                   corr_thresh: float = 0.95) -> tuple:
    """Remove zero-variance and highly correlated features."""
    keep = X.std(axis=0) > 0
    X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
    corr = np.corrcoef(X.T)
    drop = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if j not in drop and abs(corr[i, j]) >= corr_thresh:
                drop.add(j)
    idx = [i for i in range(len(names)) if i not in drop]
    return X[:, idx], [names[i] for i in idx]


def _split_categories(names: list) -> dict:
    cats = {"spectral": [], "harmonic": [], "peak": []}
    for i, n in enumerate(names):
        if any(k in n for k in ("centroid","rolloff","bandwidth")):
            cats["spectral"].append(i)
        elif any(k in n for k in ("harmonic","beat_track","perpetual_shock")):
            cats["harmonic"].append(i)
        else:
            cats["peak"].append(i)
    return cats


class AcousticClassifier:
    """
    RF + RFE chatter classifier (St. John et al. 2023 §2.3).
    Performs RFE within each feature category then fits a combined RF.
    """

    def __init__(self, n_estimators: int = 100, n_top: int = 5,
                 seed: int = 42):
        if not _SKLEARN:
            raise ImportError("scikit-learn required")
        self.n_est  = n_estimators
        self.n_top  = n_top
        self.seed   = seed
        self._model = None
        self._idx   = None
        self._names = None

    def _rfe_select(self, X: np.ndarray, y: np.ndarray,
                    names: list) -> tuple:
        cats  = _split_categories(names)
        sel_i = []
        for _, idx in cats.items():
            if len(idx) <= self.n_top:
                sel_i.extend(idx)
                continue
            rf  = RandomForestClassifier(n_estimators=self.n_est,
                                          random_state=self.seed)
            rfe = RFE(rf, n_features_to_select=self.n_top, step=1)
            rfe.fit(X[:, idx], y)
            sel_i.extend([idx[k] for k, s in enumerate(rfe.support_) if s])
        return sel_i, [names[i] for i in sel_i]

    def fit(self, X: np.ndarray, y: np.ndarray, names: list):
        self._idx, self._names = self._rfe_select(X, y, names)
        self._model = RandomForestClassifier(n_estimators=self.n_est,
                                              random_state=self.seed)
        self._model.fit(X[:, self._idx], y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X[:, self._idx])[:, 1]

    def loocv_auc(self, X: np.ndarray, y: np.ndarray,
                  names: list) -> float:
        scores = []
        loo = LeaveOneOut()
        for tr, te in loo.split(X):
            idx, _ = self._rfe_select(X[tr], y[tr], names)
            rf = RandomForestClassifier(n_estimators=self.n_est,
                                         random_state=self.seed)
            rf.fit(X[tr][:, idx], y[tr])
            scores.append(float(rf.predict_proba(X[te][:, idx])[0, 1]))
        try:
            return roc_auc_score(y, scores)
        except Exception:
            return float("nan")

    @property
    def top_features(self): return list(self._names or [])


def tpf_classify(y_audio: np.ndarray, sr: int,
                  spindle_rpm: float, n_teeth: int,
                  threshold: float = 0.5) -> int:
    """
    Label a cut as stable (0) or chatter (1) using tooth-passing-frequency
    comb filtering (Karandikar et al. 2022 §5).
    """
    N    = len(y_audio)
    fft  = np.abs(np.fft.rfft(y_audio * np.hanning(N)))
    freqs = np.fft.rfftfreq(N, d=1.0 / sr)
    tpf  = spindle_rpm / 60.0 * n_teeth
    df   = freqs[1] - freqs[0]
    tol  = max(1, int(tpf * 0.05 / df))

    def peak_near(f_target):
        c  = int(f_target / df)
        lo = max(0, c - tol); hi = min(len(fft) - 1, c + tol)
        return float(np.max(fft[lo:hi + 1]))

    tpf_max = max(peak_near(k * tpf) for k in range(1, 6)) + 1e-12
    fft_f   = fft.copy()
    for k in range(1, 20):
        lo = max(0, int((k * tpf - tpf * 0.05) / df))
        hi = min(len(fft) - 1, int((k * tpf + tpf * 0.05) / df))
        fft_f[lo:hi + 1] = 0
    return int(float(np.max(fft_f)) > threshold * tpf_max)


# ═══════════════════════════════════════════════════════════════════════════
# ── SECTION 8: SLD fusion + plotting ──────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def fuse_sld(mc_p: np.ndarray, bayes_p: np.ndarray,
             alpha: float = 0.5) -> np.ndarray:
    """
    Weighted average: p_fused = α * mc_p + (1-α) * bayes_p
    """
    return np.clip(alpha * mc_p + (1 - alpha) * bayes_p, 0, 1)


def update_with_acoustic(p_chatter: np.ndarray,
                          SS: np.ndarray, AP: np.ndarray,
                          acoustic_pts: dict,
                          bw_ss: float = 300.0,
                          bw_ap: float = 0.3) -> np.ndarray:
    """
    Incorporate acoustic RF predictions into the SLD via Gaussian kernel spread.

    acoustic_pts : dict {(ss, ap): p_chatter_from_rf}
    """
    p = p_chatter.copy()
    for (ss_m, ap_m), p_ac in acoustic_pts.items():
        w = np.exp(-0.5 * ((SS - ss_m) / bw_ss) ** 2) * \
            np.exp(-0.5 * ((AP - ap_m) / bw_ap) ** 2)
        w /= w.max() + 1e-12
        p  = (1 - w) * p + w * p_ac
    return np.clip(p, 0, 1)


def plot_sld(ss_grid:    np.ndarray,
             ap_grid:    np.ndarray,
             p_chatter:  np.ndarray,
             experiments: list           = None,
             boundary:    "StabilityBoundary" = None,
             det_eig:     tuple           = None,
             cl_results:  list            = None,
             title:       str             = "Probabilistic SLD",
             save_path:   str             = None) -> plt.Figure:
    """
    Full probabilistic SLD figure with all overlays.

    Parameters
    ----------
    ss_grid, ap_grid : 1-D grid arrays
    p_chatter        : (n_ap, n_ss) chatter probability map
    experiments      : labelled experiment points [{ss, ap, label}]
    boundary         : StabilityBoundary for reference overlay
    det_eig          : (SS_det, AP_det, eig_det) from FDM for deterministic SLD
    cl_results       : closed-loop test results from run_closed_loop()
    title, save_path : figure title and optional save path
    """
    SS, AP = np.meshgrid(ss_grid, ap_grid)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # ── probability colourmap ─────────────────────────────────────────
    cs = ax.contourf(SS, AP, p_chatter,
                     levels=np.linspace(0, 1, 21),
                     cmap="GnBu", alpha=0.75)
    cb = fig.colorbar(cs, ax=ax, pad=0.02)
    cb.set_label("Probability of chatter", size=11)
    cb.set_ticks([0.2, 0.4, 0.6, 0.8, 1.0])
    cb.ax.tick_params(labelsize=10)

    # p=0.5 iso-boundary (red)
    ct = ax.contour(SS, AP, p_chatter, [0.5],
                    colors="r", linewidths=2.0)
    ax.clabel(ct, fmt="p=0.5", fontsize=9)

    # ── deterministic FDM boundary (black) ───────────────────────────
    if det_eig is not None:
        SS_d, AP_d, eig_d = det_eig
        ax.contour(SS_d, AP_d * 1000, eig_d, [1.0],
                   colors="k", linewidths=1.2, linestyles="-",
                   label="Deterministic FDM")
        # legend proxy
        from matplotlib.lines import Line2D
        proxy_det = Line2D([0], [0], color="k", lw=1.2, label="Deterministic FDM")
        ax.add_artist(ax.legend(handles=[proxy_det],
                                loc="upper right", fontsize=9))

    # ── Excel boundary reference (blue) ───────────────────────────────
    if boundary is not None:
        ss_b, ap_b = boundary.plot_overlay(ss_grid)
        valid = (ap_b >= ap_grid.min()) & (ap_b <= ap_grid.max())
        ax.plot(ss_b[valid], ap_b[valid], "b-", lw=1.8,
                label="Exp. boundary (Excel)", zorder=6)

    # ── experiment scatter ────────────────────────────────────────────
    if experiments:
        markers = {0: ("o", "k",  "Stable"),
                   1: ("x", "k",  "Chatter"),
                   2: ("^", "w",  "Marginal")}
        seen = set()
        for exp in experiments:
            lbl = int(exp.get("label", 0))
            mk, fc, lb = markers.get(lbl, ("o", "gray", ""))
            lab = lb if lbl not in seen else ""
            seen.add(lbl)
            ax.scatter(exp["ss"], exp["ap"],
                       c=fc, marker=mk, s=55,
                       edgecolors="k", linewidths=1.4,
                       label=lab, zorder=7)

    # ── closed-loop test points ───────────────────────────────────────
    if cl_results:
        for r in cl_results:
            mk = "o" if r["stable"] else "x"
            ax.scatter(r["ss"], r["ap"],
                       c="none", marker=mk, s=90,
                       edgecolors="darkorange", linewidths=1.5,
                       zorder=8)
        # legend proxy
        from matplotlib.lines import Line2D
        proxy_cl = Line2D([0], [0], marker="o", color="none",
                          markeredgecolor="darkorange",
                          markersize=8, label="Closed-loop tests")
        ax.add_artist(ax.legend(handles=[proxy_cl],
                                loc="lower right", fontsize=9))

    # ── p=0.5 boundary proxy for legend ──────────────────────────────
    from matplotlib.lines import Line2D
    proxies = [Line2D([0], [0], color="r",  lw=2.0, label="p(chatter)=0.5")]
    if boundary is not None:
        proxies.append(Line2D([0], [0], color="b", lw=1.8,
                               label="Excel boundary"))
    ax.legend(handles=proxies + (
        [Line2D([0], [0], marker="o", color="none",
                 markerfacecolor="k", markeredgecolor="k",
                 markersize=7, label="Stable (exp)"),
         Line2D([0], [0], marker="x", color="k",
                 markersize=7, label="Chatter (exp)")]
        if experiments else []),
        loc="upper left", fontsize=9, ncol=2)

    ax.set_xlabel("Spindle speed [rpm]",   size=12)
    ax.set_ylabel("Axial depth [mm]",      size=12)
    ax.set_title(title, size=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        for ext in (".png", ".svg"):
            fig.savefig(save_path.replace(".png", ext), dpi=150,
                        bbox_inches="tight")
        print(f"  Saved figure: {save_path}")

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# ── SECTION 9: Full integrated pipeline ────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

class PIMlPipeline:
    """
    End-to-end Physics-Informed ML Stability Lobe Diagram pipeline.

    Usage
    -----
    pipe = PIMlPipeline(config)
    pipe.load_boundary("stability_boundary3__1_.xlsx")   # optional
    pipe.run()
    pipe.plot(save="results/sld.png")

    All intermediate results are stored as attributes for inspection.
    """

    # ── default configuration ─────────────────────────────────────────
    DEFAULT = dict(
        # prior modal/force parameters  [wx_Hz, wy_Hz, cx, cy, kx, ky, Kt, Kn]
        # units: Hz, -, MN/m, 100 MPa
        prior_mean = [782.7, 752.8, 0.0184, 0.0186, 6.5616, 4.8852, 10.95, 1.76],
        prior_std_pct = 20.0,       # σ = 20 % of mean

        # SLD grid
        ss_min  = 4800.0,   ss_max = 13200.0, n_ss = 51,
        ap_min  = 0.0,      ap_max =    15.0, n_ap = 50,

        # surrogate training
        n_surrogate_samples = 50,    # 800 in paper; reduce for speed
        surrogate_epochs    = 500,   # 3000 in paper

        # MAP / Laplace
        map_lr      = 0.002,
        map_epochs  = 200,           # 500 in paper
        logistic_a  = 4.0,

        # Monte Carlo SLD
        n_mc_samples = 100,          # 500 in paper

        # Bayesian stability map
        bayes_bw_ss       = 500.0,
        bayes_bw_ap       = 0.5,
        min_stability     = 0.8,
        convergence_ei    = 0.05,
        max_cl_tests      = 20,

        # fusion weights
        mc_weight         = 0.6,     # weight given to MC SLD vs Bayesian map

        # acoustic
        acoustic_bw_ss    = 300.0,
        acoustic_bw_ap    = 0.3,

        output_dir        = "results",
    )

    def __init__(self, config: dict = None):
        self.cfg = {**self.DEFAULT, **(config or {})}
        os.makedirs(self.cfg["output_dir"], exist_ok=True)

        mu = np.array(self.cfg["prior_mean"], dtype=float)
        self.mu    = mu
        self.sigma = mu * self.cfg["prior_std_pct"] / 100.0

        self.ss_grid = np.linspace(self.cfg["ss_min"],
                                   self.cfg["ss_max"], self.cfg["n_ss"])
        self.ap_grid = np.linspace(self.cfg["ap_min"],
                                   self.cfg["ap_max"], self.cfg["n_ap"])
        self.SS, self.AP = np.meshgrid(self.ss_grid, self.ap_grid)

        # result containers
        self.boundary:    "StabilityBoundary" = None
        self.experiments: list                = []
        self.surrogates:  dict                = {}
        self.w_star:      np.ndarray          = None
        self.Sigma_star:  np.ndarray          = None
        self.mc_p:        np.ndarray          = None
        self.bayes_map:   BayesianStabilityMap = None
        self.cl_results:  list                = []
        self.p_final:     np.ndarray          = None
        self.acoustic_pts: dict               = {}
        self.det_sld:     tuple               = None

    # ── public helpers ────────────────────────────────────────────────

    def load_boundary(self, path: str,
                       ss_range: tuple = None,
                       n_stable: int = 30, n_chatter: int = 30):
        """Load Excel boundary and generate labelled experiments."""
        print(f"\n── Loading boundary: {path} ──")
        self.boundary = StabilityBoundary(path)
        ss_range = ss_range or (self.cfg["ss_min"], self.cfg["ss_max"])
        self.experiments = self.boundary.make_experiments(
            ss_range=ss_range, n_stable=n_stable, n_chatter=n_chatter)

    def add_experiments(self, experiments: list):
        """Add manually specified experiment list."""
        self.experiments.extend(experiments)

    def add_audio_file(self, path: str, ss: float, ap: float,
                        n_teeth: int, label: int = None):
        """
        Add a single audio file cut to the acoustic dataset.
        If label is None, uses TPF classification automatically.
        """
        if not _LIBROSA:
            warnings.warn("librosa not installed – audio file ignored.")
            return
        y_raw, sr = librosa.load(path, sr=20000, mono=True)
        if label is None:
            label = tpf_classify(y_raw, sr, ss, n_teeth)
        self.experiments.append({"ss": ss, "ap": ap, "label": label,
                                  "_audio": (y_raw, sr), "surrogate_key": None})

    # ── pipeline stages ───────────────────────────────────────────────

    def _stage_fdm_prior(self):
        print("\n── Stage A: FDM physics prior ──")
        if _FDM_MODULE is None:
            print("  FDM not available – using linear uninformed prior.")
            return None

        SS_det, AP_det, eig_det, _ = _FDM_MODULE.FDM(
            "SampleAtAllGridPoint",
            self.mu[0], self.mu[1], self.mu[2], self.mu[3],
            self.mu[4], self.mu[5], self.mu[6], self.mu[7])
        self.det_sld = (SS_det, AP_det, eig_det)
        print(f"  FDM map computed: {eig_det.shape}")
        return eig_det

    def _stage_surrogates(self):
        print("\n── Stage B: Surrogate training ──")
        n_ss   = self.cfg["n_ss"]
        n_ap   = self.cfg["n_ap"]
        n_grid = n_ss * n_ap

        if not _TORCH:
            print("  PyTorch unavailable – surrogates skipped.")
            return

        all_data = build_surrogate_dataset(
            self.mu, self.sigma,
            n_samples=self.cfg["n_surrogate_samples"],
            purpose="SampleAtAllGridPoint",
            verbose=True)

        # re-index: FDM returns (ss_steps, ap_steps) flattened
        # we need grid indexed by i_ss * n_ap + i_ap
        fdm_n_ap = len(np.linspace(0, 4e-3, 51))
        fdm_n_ss = len(np.linspace(4800, 13200, 101))
        n_fdm_grid = fdm_n_ap * fdm_n_ss

        self.surrogates = train_all_surrogates(
            all_data,
            n_samples=self.cfg["n_surrogate_samples"],
            n_grid=n_fdm_grid,
            epochs=self.cfg["surrogate_epochs"],
            verbose=True)

        # attach surrogate keys to experiments
        for exp in self.experiments:
            if exp.get("surrogate_key") is None:
                # find nearest grid point
                i_ss = int(np.argmin(np.abs(
                    np.linspace(4800, 13200, fdm_n_ss) - exp["ss"])))
                i_ap = int(np.argmin(np.abs(
                    np.linspace(0, 4e-3, fdm_n_ap) - exp["ap"] / 1000)))
                exp["surrogate_key"] = i_ss * fdm_n_ap + i_ap

        print(f"  {len(self.surrogates)} surrogates trained.")

    def _stage_posterior(self):
        print("\n── Stage C: MAP posterior inference ──")
        print(f"  Using {len(self.experiments)} labelled cuts")

        self.w_star, self.Sigma_star = map_inference(
            self.mu, self.sigma,
            self.experiments, self.surrogates,
            lr=self.cfg["map_lr"],
            n_epochs=self.cfg["map_epochs"],
            a=self.cfg["logistic_a"],
            verbose=True)

        print(f"\n  w* = {np.round(self.w_star, 4)}")

    def _stage_monte_carlo(self):
        print("\n── Stage D: Monte Carlo SLD ──")
        print(f"  {self.cfg['n_mc_samples']} posterior samples")

        self.mc_p = monte_carlo_sld(
            self.w_star, self.Sigma_star,
            self.ss_grid, self.ap_grid,
            self.surrogates,
            n_mc=self.cfg["n_mc_samples"],
            a=self.cfg["logistic_a"],
            verbose=True)

        self.mc_p = np.clip(self.mc_p, 0.0, 1.0)
        print(f"  MC SLD done. max p_chatter = {self.mc_p.max():.3f}")

    def _stage_bayesian_map(self, physics_prior_eig=None):
        print("\n── Stage E: Bayesian stability map ──")

        # build physics-informed prior if FDM was available
        if physics_prior_eig is not None:
            from scipy.interpolate import RegularGridInterpolator
            ss_fdm = np.linspace(4800, 13200, physics_prior_eig.shape[0])
            ap_fdm = np.linspace(0, 4.0, physics_prior_eig.shape[1])
            interp = RegularGridInterpolator(
                (ss_fdm, ap_fdm), physics_prior_eig,
                method="linear", bounds_error=False, fill_value=1.5)
            pts     = np.stack([self.SS.ravel(), self.AP.ravel()], axis=-1)
            eig_int = interp(pts).reshape(self.AP.shape)
            a       = self.cfg["logistic_a"]
            p_stab  = 1.0 - 1.0 / (1.0 + np.exp(-a * (eig_int - 1.0)))
            prior   = np.clip(p_stab, 1e-6, 1 - 1e-6)
        else:
            prior = None

        self.bayes_map = BayesianStabilityMap(
            self.ss_grid, self.ap_grid,
            prior=prior,
            bw_ss=self.cfg["bayes_bw_ss"],
            bw_ap=self.cfg["bayes_bw_ap"])

        # oracle for closed-loop
        if self.boundary is not None:
            oracle = self.boundary.oracle
        else:
            # oracle from experiments (approximate)
            oracle = self._experiment_oracle()

        sel = TestSelector(self.bayes_map,
                            min_stability=self.cfg["min_stability"],
                            convergence_ei=self.cfg["convergence_ei"])

        print(f"  Running closed-loop (max {self.cfg['max_cl_tests']} tests)...")
        self.cl_results = run_closed_loop(
            self.bayes_map, sel, oracle,
            max_tests=self.cfg["max_cl_tests"],
            verbose=True)

    def _experiment_oracle(self):
        """Build approximate oracle from labelled experiments via KNN."""
        from sklearn.neighbors import KNeighborsClassifier
        if not self.experiments or not _SKLEARN:
            return lambda ss, ap: ap < 1.5
        X = np.array([[e["ss"], e["ap"]] for e in self.experiments])
        y = np.array([e["label"] for e in self.experiments])
        knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)
        return lambda ss, ap: knn.predict([[ss, ap]])[0] == 0

    def _stage_acoustic(self):
        """Extract acoustic features and run RF classifier if audio data present."""
        audio_exps = [e for e in self.experiments if "_audio" in e]
        if not audio_exps or not _SKLEARN:
            return

        print(f"\n── Stage F: Acoustic classifier ({len(audio_exps)} files) ──")
        all_feats, labels = [], []
        for e in audio_exps:
            y_a, sr = e["_audio"]
            try:
                feats = extract_features(y_a, sr)
                all_feats.append(feats)
                labels.append(e["label"])
            except Exception as ex:
                warnings.warn(f"Feature extraction failed: {ex}")

        if len(all_feats) < 4:
            print("  Too few audio samples for RF – skipping.")
            return

        names = list(all_feats[0].keys())
        X     = np.array([[f.get(n, 0.0) for n in names] for f in all_feats])
        y     = np.array(labels)
        X_p, p_names = prune_features(X, names)

        clf = AcousticClassifier(n_estimators=100, n_top=5)
        clf.fit(X_p, y, p_names)
        auc = clf.loocv_auc(X_p, y, p_names)
        print(f"  RF fitted. LOOCV AUC = {auc:.3f}")
        print(f"  Top features: {clf.top_features[:5]}")

        # record acoustic predictions for fusion
        for e, feats in zip(audio_exps, all_feats):
            fv   = np.array([[feats.get(n, 0.0) for n in p_names]])
            p_ch = float(clf.predict_proba(fv)[0])
            self.acoustic_pts[(e["ss"], e["ap"])] = p_ch

    def _stage_fuse(self):
        print("\n── Stage G: Fusion ──")
        p = self.mc_p.copy()

        p = fuse_sld(p, self.bayes_map.p_chatter,
                     alpha=self.cfg["mc_weight"])
        print(f"  Fused MC + Bayesian map (α={self.cfg['mc_weight']})")

        if self.acoustic_pts:
            p = update_with_acoustic(p, self.SS, self.AP, self.acoustic_pts,
                                      self.cfg["acoustic_bw_ss"],
                                      self.cfg["acoustic_bw_ap"])
            print(f"  Incorporated {len(self.acoustic_pts)} acoustic predictions")

        self.p_final = np.clip(p, 0, 1)
        print(f"  Final p_chatter: min={self.p_final.min():.3f}  "
              f"max={self.p_final.max():.3f}")

        # save
        out = os.path.join(self.cfg["output_dir"], "probabilistic_sld.npz")
        np.savez(out, ss_grid=self.ss_grid, ap_grid=self.ap_grid,
                 p_chatter=self.p_final)
        print(f"  Saved: {out}")

    # ── main run ─────────────────────────────────────────────────────

    def run(self):
        """Execute the full pipeline end to end."""
        print("\n" + "═" * 60)
        print("  PIML Stability Lobe Diagram Pipeline")
        print(f"  Grid: {self.cfg['n_ss']} × {self.cfg['n_ap']}  "
              f"({self.cfg['ss_min']:.0f}–{self.cfg['ss_max']:.0f} rpm  "
              f"{self.cfg['ap_min']:.1f}–{self.cfg['ap_max']:.1f} mm)")
        print("═" * 60)

        eig = self._stage_fdm_prior()
        self._stage_surrogates()
        self._stage_posterior()
        self._stage_monte_carlo()
        self._stage_bayesian_map(physics_prior_eig=eig)
        self._stage_acoustic()
        self._stage_fuse()

        print("\n" + "═" * 60)
        print("  Pipeline complete.")
        print("═" * 60)

    # ── plotting ─────────────────────────────────────────────────────

    def plot(self, save: str = None, show: bool = True) -> plt.Figure:
        """Generate and optionally save the final probabilistic SLD figure."""
        if self.p_final is None:
            raise RuntimeError("Run pipeline first (pipe.run()).")

        save = save or os.path.join(self.cfg["output_dir"], "probabilistic_sld.png")

        fig = plot_sld(
            self.ss_grid, self.ap_grid, self.p_final,
            experiments = self.experiments,
            boundary    = self.boundary,
            det_eig     = self.det_sld,
            cl_results  = self.cl_results,
            title       = "Physics-Informed Probabilistic SLD",
            save_path   = save)

        if show:
            plt.show()
        return fig


# ═══════════════════════════════════════════════════════════════════════════
# ── SECTION 10: Run modes ─────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def run_demo():
    """
    Self-contained demo using analytic sinusoidal boundary.
    No FDM, no PyTorch, no audio required.
    All stages validated with synthetic data.
    """
    print("╔══════════════════════════════════════════════════════╗")
    print("║   PIML Pipeline  —  DEMO MODE                       ║")
    print("║   Synthetic sinusoidal stability boundary            ║")
    print("╚══════════════════════════════════════════════════════╝")

    rng = np.random.default_rng(0)

    # ── grid ─────────────────────────────────────────────────────────
    ss_grid = np.linspace(4800, 13200, 50)
    ap_grid = np.linspace(0.0, 15.0, 50)
    SS, AP  = np.meshgrid(ss_grid, ap_grid)

    # ── analytic boundary ─────────────────────────────────────────────
    def boundary_fn(ss):
        return 2.5 + 1.5 * np.sin(2 * np.pi * ss / 2800)

    def oracle(ss, ap):
        return ap < boundary_fn(ss)

    # ── analytic p_chatter ────────────────────────────────────────────
    bnd_2d  = boundary_fn(SS)
    p_mc    = np.clip(1.0 / (1.0 + np.exp(-4.0 * (AP - bnd_2d))), 0, 1)
    print("  MC SLD: analytic logistic model")

    # ── synthetic experiments ─────────────────────────────────────────
    experiments = []
    for _ in range(20):
        ss = rng.uniform(5000, 13000)
        bnd = boundary_fn(ss)
        experiments.append({"ss": float(ss),
                              "ap": float(rng.uniform(0.3, max(0.3, bnd - 0.3))),
                              "label": 0})
        experiments.append({"ss": float(ss),
                              "ap": float(bnd + rng.uniform(0.2, 1.0)),
                              "label": 1})
    print(f"  Generated {len(experiments)} synthetic experiments")

    # ── Bayesian stability map ────────────────────────────────────────
    bsm = BayesianStabilityMap(ss_grid, ap_grid, bw_ss=600, bw_ap=0.8)
    for e in experiments[:10]:
        bsm.update(e["ss"], e["ap"], e["label"] == 0)
    print("  Bayesian map updated with 10 test points")

    # ── closed-loop ───────────────────────────────────────────────────
    sel = TestSelector(bsm, min_stability=0.8, convergence_ei=0.05)
    cl  = run_closed_loop(bsm, sel, oracle, max_tests=12, verbose=True)
    n_s = sum(1 for r in cl if r["stable"])
    print(f"  Closed-loop: {n_s} stable, {len(cl)-n_s} chatter in {len(cl)} tests")

    # ── fuse ──────────────────────────────────────────────────────────
    p_fused = fuse_sld(p_mc, bsm.p_chatter, alpha=0.6)
    print(f"  Fused SLD: max p = {p_fused.max():.3f}")

    # ── save ─────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    np.savez("results/demo_sld.npz",
             ss_grid=ss_grid, ap_grid=ap_grid, p_chatter=p_fused)

    # ── acoustic classifier test ──────────────────────────────────────
    if _SKLEARN:
        X  = rng.standard_normal((24, 50))
        X[12:, :5] += 3.0
        y  = np.array([0] * 12 + [1] * 12)
        nm = ([f"centroids_{i}" for i in range(16)] +
              [f"bandwidth_2_{i}" for i in range(8)] +
              [f"harmonics_{i}" for i in range(8)] +
              [f"perpetual_shock_{i}" for i in range(8)] +
              [f"peak_{i}_magnitude" for i in range(10)])
        Xp, pn = prune_features(X, nm)
        clf = AcousticClassifier(n_estimators=50, n_top=4)
        clf.fit(Xp, y, pn)
        auc = clf.loocv_auc(Xp, y, pn)
        print(f"  Acoustic RF LOOCV AUC = {auc:.3f}  "
              f"top = {clf.top_features[:3]}")

    # ── plot ──────────────────────────────────────────────────────────
    # create a fake boundary object from the analytic function
    ss_arr = np.linspace(4800, 13200, 200)
    ap_arr = boundary_fn(ss_arr)
    fake_boundary_plot = lambda: (ss_arr, ap_arr)

    fig, ax = plt.subplots(figsize=(10, 6))
    cs = ax.contourf(SS, AP, p_fused, levels=np.linspace(0, 1, 21),
                     cmap="GnBu", alpha=0.75)
    cb = fig.colorbar(cs, ax=ax, pad=0.02)
    cb.set_label("Probability of chatter", size=11)
    ax.contour(SS, AP, p_fused, [0.5], colors="r", linewidths=2.0)

    ss_b, ap_b = fake_boundary_plot()
    valid = (ap_b >= ap_grid.min()) & (ap_b <= ap_grid.max())
    ax.plot(ss_b[valid], ap_b[valid], "b-", lw=1.8, label="True boundary")

    for e in experiments[:20]:
        mk = "o" if e["label"] == 0 else "x"
        fc = "k"
        ax.scatter(e["ss"], e["ap"], c=fc, marker=mk, s=40,
                   edgecolors="k", linewidths=1.2, zorder=7)
    for r in cl:
        mk = "o" if r["stable"] else "x"
        ax.scatter(r["ss"], r["ap"], c="none", marker=mk, s=90,
                   edgecolors="darkorange", linewidths=1.5, zorder=8)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color="r", lw=2.0, label="p(chatter)=0.5"),
        Line2D([0], [0], color="b", lw=1.8, label="True boundary"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="k",
               markeredgecolor="k", markersize=7, label="Experiment"),
        Line2D([0], [0], marker="o", color="none", markersize=9,
               markeredgecolor="darkorange", label="Closed-loop test"),
    ], loc="upper left", fontsize=9)

    ax.set_xlabel("Spindle speed [rpm]", size=12)
    ax.set_ylabel("Axial depth [mm]", size=12)
    ax.set_title("Physics-Informed Probabilistic SLD  (demo mode)", size=13)
    plt.tight_layout()
    plt.savefig("results/demo_sld.png", dpi=150, bbox_inches="tight")
    print("\n  Saved: results/demo_sld.png")
    plt.show()

    print("\n══ Demo complete ══")
    return p_fused


def run_excel_mode(boundary_path: str):
    """
    Pipeline driven by the Excel stability boundary file.
    Uses the boundary as:
      1. Ground-truth oracle for closed-loop simulation
      2. Source of labelled experiments for MAP inference
      3. Reference overlay on the final plot
    """
    print("╔══════════════════════════════════════════════════════╗")
    print("║   PIML Pipeline  —  EXCEL MODE                      ║")
    print(f"║   Boundary: {os.path.basename(boundary_path):<38}║")
    print("╚══════════════════════════════════════════════════════╝")

    pipe = PIMlPipeline(config={
        "ap_max":              15.0,
        "n_ap":                60,
        "n_surrogate_samples": 50,
        "surrogate_epochs":    300,
        "n_mc_samples":        100,
        "map_epochs":          200,
        "max_cl_tests":        15,
    })

    pipe.load_boundary(boundary_path, n_stable=30, n_chatter=30)
    pipe.run()
    pipe.plot(save="results/excel_sld.png", show=True)
    return pipe


def run_full_mode(boundary_path: str = None, fdm_path: str = None):
    """
    Full physics pipeline: FDM + surrogates + MAP + Monte Carlo + Bayesian.
    Requires: torch, FDM_function.py, and optionally the Excel boundary.
    """
    print("╔══════════════════════════════════════════════════════╗")
    print("║   PIML Pipeline  —  FULL PHYSICS MODE               ║")
    print("╚══════════════════════════════════════════════════════╝")

    if fdm_path:
        ok = _try_load_fdm(fdm_path)
        print(f"  FDM loaded: {ok}")

    pipe = PIMlPipeline(config={
        "n_surrogate_samples": 800,
        "surrogate_epochs":    3000,
        "n_mc_samples":        500,
        "map_epochs":          500,
    })

    if boundary_path:
        pipe.load_boundary(boundary_path, n_stable=40, n_chatter=40)

    pipe.run()
    pipe.plot(save="results/full_sld.png", show=True)
    return pipe


# ═══════════════════════════════════════════════════════════════════════════
# ── CLI entry point ────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Physics-Informed ML Stability Lobe Diagram",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python piml_full.py --mode demo
  python piml_full.py --mode excel --boundary stability_boundary3__1_.xlsx
  python piml_full.py --mode full  --boundary stability_boundary3__1_.xlsx --fdm FDM_function.py
        """)
    ap.add_argument("--mode",     default="demo",
                    choices=["demo", "excel", "full"],
                    help="Pipeline mode")
    ap.add_argument("--boundary", default=None,
                    help="Path to stability boundary Excel file (.xlsx)")
    ap.add_argument("--fdm",      default=None,
                    help="Path to FDM_function.py (optional)")
    args = ap.parse_args()

    if args.mode == "demo":
        run_demo()

    elif args.mode == "excel":
        if not args.boundary:
            print("ERROR: --boundary required for excel mode")
            sys.exit(1)
        run_excel_mode(args.boundary)

    elif args.mode == "full":
        run_full_mode(boundary_path=args.boundary, fdm_path=args.fdm)
