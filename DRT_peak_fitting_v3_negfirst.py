"""
DRT Split-Gaussian Peak Fitting + RC Deconvolution (Negative-First)
====================================================================
Version: 3.2.0
Date: 2026-03-24

Negative-first fitting:
  Phase 1: Fit negative peaks FIRST, tightly constrained by DRT zero crossings.
           Each negative segment's amplitude = actual DRT minimum,
           sigma_L/R derived from distance to zero crossings.
  Phase 2: Fit positive peaks on residual (gamma - negative_fit).
  Phase 3: Joint refinement of all peaks on full gamma.

Split Gaussian (asymmetric) model:
  gamma_k(x) = A * exp(-(x-mu)^2 / (2*sigma_L^2))   for x < mu
             = A * exp(-(x-mu)^2 / (2*sigma_R^2))   for x >= mu
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# ============================================================================
# USER CONFIGURATION
# ============================================================================

INPUT_CSV = r"C:\Users\tpham\Documents\DRT peak fitting test\2NP53\Breakin 12\PWRGEIS_400mA_#1_DRT_relaxed_lam1eneg03_2nd_L1\PWRGEIS_400mA_#1_DRT_fine.csv"
EIS_CSV = r"C:\Users\tpham\Documents\DRT peak fitting test\2NP53\Breakin 12\PWRGEIS_400mA_#1.csv"

R_OHM = None
L_H   = None
HFR_FREQ_RANGE = (8e3, 1e5)
HFR_WINDOW_PTS = 5

PEAK_PROMINENCE = 0.01
N_PEAKS_MODE = 'auto'
N_PEAKS_FIXED = 6
N_PEAKS_RANGE = (3, 0)
FREQ_RANGE = []

CELL_AREA_CM2 = 25

P1_TAU_MAX = 1e-4
P3_C_MIN = 0.5

FONT_NAME = 'Arial'
FONT_SIZE = 30
LINE_WIDTH = 2
MARKER_SIZE = 15

PARAMS_PER_PEAK = 4   # A, mu, sigma_L, sigma_R

# ============================================================================
# END USER CONFIGURATION
# ============================================================================

COLOR_CONTACT   = '#B0B0B0'
COLOR_KINETIC   = ['#FF9999', '#FFB366', '#FFEE99']
COLOR_IONIC     = '#99DD99'
COLOR_TRANSPORT = ['#99CCFF', '#B399FF']
COLOR_INDUCTIVE = ['#DD99FF', '#CC88EE']


def assign_peak_colors(rc_elements):
    """Assign colors based on physical process classification."""
    colors = [None] * len(rc_elements)
    labels = [None] * len(rc_elements)
    kinetic_idx = []
    transport_idx = []
    p3_tau = None

    for i, rc in enumerate(rc_elements):
        if rc['sign'] == 'negative':
            continue
        if rc['tau_k'] < P1_TAU_MAX:
            colors[i] = COLOR_CONTACT
            labels[i] = f"{rc['peak_label']} contact"
        elif rc['C_k'] > P3_C_MIN:
            if p3_tau is None or rc['tau_k'] < p3_tau:
                if p3_tau is not None:
                    for j in range(len(rc_elements)):
                        if labels[j] and 'ionic' in labels[j]:
                            transport_idx.append(j)
                            colors[j] = None
                            labels[j] = None
                p3_tau = rc['tau_k']
                colors[i] = COLOR_IONIC
                labels[i] = f"{rc['peak_label']} ionic"
            else:
                transport_idx.append(i)

    for i, rc in enumerate(rc_elements):
        if colors[i] is not None or rc['sign'] == 'negative':
            continue
        if rc['peak_type'] == 'capacitive':
            if p3_tau is not None and rc['tau_k'] < p3_tau:
                kinetic_idx.append(i)
            else:
                transport_idx.append(i)

    for k, idx in enumerate(sorted(kinetic_idx, key=lambda i: rc_elements[i]['tau_k'])):
        ci = k % len(COLOR_KINETIC)
        colors[idx] = COLOR_KINETIC[ci]
        sub_label = f".{k+1}" if len(kinetic_idx) > 1 else ""
        labels[idx] = f"{rc_elements[idx]['peak_label']} kinetic{sub_label}"

    for k, idx in enumerate(sorted(transport_idx, key=lambda i: rc_elements[i]['tau_k'])):
        ci = k % len(COLOR_TRANSPORT)
        colors[idx] = COLOR_TRANSPORT[ci]
        labels[idx] = f"{rc_elements[idx]['peak_label']} transport"

    neg_idx = [i for i, rc in enumerate(rc_elements) if rc['sign'] == 'negative']
    for k, idx in enumerate(sorted(neg_idx, key=lambda i: rc_elements[i]['tau_k'])):
        ci = k % len(COLOR_INDUCTIVE)
        colors[idx] = COLOR_INDUCTIVE[ci]
        labels[idx] = f"{rc_elements[idx]['peak_label']} inductive"

    return colors, labels


def setup_plot_style():
    plt.rcParams.update({
        'font.family': FONT_NAME, 'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE, 'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE, 'xtick.direction': 'in',
        'ytick.direction': 'in', 'xtick.top': False,
        'ytick.right': False, 'axes.linewidth': LINE_WIDTH,
        'lines.linewidth': LINE_WIDTH, 'lines.markersize': MARKER_SIZE,
    })


def load_fine_csv(filepath):
    df = pd.read_csv(filepath)
    tau = df['tau_s'].values
    gamma = df['gamma_Ohm'].values
    basename = os.path.splitext(os.path.basename(filepath))[0]
    if basename.endswith('_DRT_fine'):
        dataset_name = basename[:-9]
    else:
        dataset_name = basename
    return tau, gamma, dataset_name


def load_eis_csv(filepath):
    df = pd.read_csv(filepath, header=None, names=['freq_Hz', 'Z_re', 'Z_im'])
    return df['freq_Hz'].values, df['Z_re'].values, df['Z_im'].values


def auto_fit_hfr_L(freq, Z_re, Z_im, freq_range=HFR_FREQ_RANGE, window=HFR_WINDOW_PTS):
    """Estimate HFR by interpolating Z_re at the high-frequency Z_im=0 crossing.
    Estimate L from the highest-frequency point where Z_im > 0 (inductive)."""
    # Sort by frequency descending for HF-first search
    sort_idx = np.argsort(freq)[::-1]
    f_s = freq[sort_idx]
    zr_s = Z_re[sort_idx]
    zi_s = Z_im[sort_idx]

    fmin, fmax = freq_range

    # Method 1: Find HF Z_im zero crossing (transition from inductive Z_im>0 to capacitive Z_im<0)
    R_ohm = None
    for i in range(len(f_s) - 1):
        if f_s[i] < fmin:
            break
        # Z_im > 0 = inductive, Z_im < 0 = capacitive
        if zi_s[i] > 0 and zi_s[i+1] <= 0:
            # Interpolate Z_re at Z_im = 0
            frac = zi_s[i] / (zi_s[i] - zi_s[i+1])
            R_ohm = zr_s[i] + frac * (zr_s[i+1] - zr_s[i])
            break
        elif zi_s[i] <= 0 and zi_s[i+1] > 0:
            frac = -zi_s[i] / (zi_s[i+1] - zi_s[i])
            R_ohm = zr_s[i] + frac * (zr_s[i+1] - zr_s[i])
            break

    if R_ohm is None:
        # Fallback: if no crossing found, use min Z_re in HF range
        mask_hf = freq >= fmin
        if np.any(mask_hf):
            R_ohm = np.min(Z_re[mask_hf])
        else:
            return None, None

    # Estimate L from highest frequency point
    idx_max_f = np.argmax(freq)
    L = abs(Z_im[idx_max_f]) / (2 * np.pi * freq[idx_max_f])

    return R_ohm, L


# ---- Split Gaussian Model ----

def split_gaussian_single(x, A, mu, sigma_L, sigma_R):
    result = np.empty_like(x)
    left = x < mu
    result[left] = A * np.exp(-(x[left] - mu)**2 / (2 * sigma_L**2))
    result[~left] = A * np.exp(-(x[~left] - mu)**2 / (2 * sigma_R**2))
    return result


def multi_gaussian(ln_tau, *params):
    result = np.zeros_like(ln_tau)
    N = len(params) // PARAMS_PER_PEAK
    for k in range(N):
        b = k * PARAMS_PER_PEAK
        result += split_gaussian_single(ln_tau, params[b], params[b+1], params[b+2], params[b+3])
    return result


# ---- Constrained Negative Peak Fitting ----

def fit_negative_segments(ln_tau, gamma):
    """Fit negative peaks directly from DRT zero crossings.
    Each negative segment gets one or more split Gaussians (one per sub-minimum).
    Constraints:
      - A_k = actual DRT minimum (tight bounds: 0.5x to 1.5x)
      - mu_k = ln(tau) at minimum (tight bounds: +/- 0.5)
      - sigma_L/R derived from distance to adjacent zero crossing or sub-minimum
    Returns (popt_neg, N_neg, gamma_neg_fit, neg_segments_info).
    """
    signs = np.sign(gamma)
    # Handle exact zeros
    signs[signs == 0] = 1
    sign_changes = np.where(np.diff(signs))[0]
    boundaries = [0] + list(sign_changes + 1) + [len(gamma)]

    neg_segments = []
    for seg_i in range(len(boundaries) - 1):
        lo = boundaries[seg_i]
        hi = boundaries[seg_i + 1]
        seg = gamma[lo:hi]
        if len(seg) < 2:
            continue
        if np.mean(seg) < 0:
            left_zero = ln_tau[lo]
            right_zero = ln_tau[min(hi, len(ln_tau)-1)]

            # Detect sub-minima within this negative segment
            neg_seg = -seg  # flip for peak detection
            abs_max = np.max(np.abs(gamma))
            sub_peaks, _ = find_peaks(neg_seg, prominence=0.0005 * abs_max)

            if len(sub_peaks) <= 1:
                # Single minimum — original behavior
                idx_min = lo + np.argmin(seg)
                A_min = gamma[idx_min]
                mu_min = ln_tau[idx_min]
                sig_L = max((mu_min - left_zero) / 3.0, 0.05)
                sig_R = max((right_zero - mu_min) / 3.0, 0.05)
                neg_segments.append({
                    'A': A_min, 'mu': mu_min,
                    'sigma_L': sig_L, 'sigma_R': sig_R,
                    'left_zero': left_zero, 'right_zero': right_zero,
                    'lo': lo, 'hi': hi,
                })
            else:
                # Multiple sub-minima — split into separate peaks
                sub_peaks_abs = [lo + p for p in sub_peaks]
                # Build boundaries for each sub-peak: midpoints between sub-minima
                sub_bounds = [left_zero]
                for j in range(len(sub_peaks_abs) - 1):
                    # Midpoint between adjacent sub-minima in ln_tau
                    mid = (ln_tau[sub_peaks_abs[j]] + ln_tau[sub_peaks_abs[j+1]]) / 2.0
                    sub_bounds.append(mid)
                sub_bounds.append(right_zero)

                for j, sp in enumerate(sub_peaks_abs):
                    A_min = gamma[sp]
                    mu_min = ln_tau[sp]
                    left_b = sub_bounds[j]
                    right_b = sub_bounds[j+1]
                    sig_L = max((mu_min - left_b) / 3.0, 0.05)
                    sig_R = max((right_b - mu_min) / 3.0, 0.05)
                    neg_segments.append({
                        'A': A_min, 'mu': mu_min,
                        'sigma_L': sig_L, 'sigma_R': sig_R,
                        'left_zero': left_b, 'right_zero': right_b,
                        'lo': lo, 'hi': hi,
                    })

    if not neg_segments:
        return None, 0, np.zeros_like(gamma), []

    N_neg = len(neg_segments)
    p0 = []
    lower = []
    upper = []

    for seg in neg_segments:
        A = seg['A']
        mu = seg['mu']
        sL = seg['sigma_L']
        sR = seg['sigma_R']
        p0.extend([A, mu, sL, sR])
        # Tight bounds on A: between 1.5x and 0.5x of actual minimum
        # (A is negative, so lower bound is more negative)
        lower.extend([min(A * 1.5, A * 0.5), mu - 0.5, sL * 0.3, sR * 0.3])
        upper.extend([max(A * 0.5, A * 1.5), mu + 0.5, sL * 3.0, sR * 3.0])
        # Ensure upper A < 0
        upper[-4] = min(upper[-4], -1e-10)

    p0 = np.array(p0)

    try:
        popt, _ = curve_fit(
            multi_gaussian, ln_tau, np.minimum(gamma, 0.0),
            p0=p0, bounds=(lower, upper), maxfev=10000
        )
        gamma_neg_fit = multi_gaussian(ln_tau, *popt)
        return popt, N_neg, gamma_neg_fit, neg_segments
    except Exception as e:
        print(f"  WARNING: Constrained negative fit failed ({e}), using initial estimates")
        gamma_neg_fit = multi_gaussian(ln_tau, *p0)
        return p0, N_neg, gamma_neg_fit, neg_segments


# ---- Positive Peak Detection and Fitting ----

def detect_positive_peaks(tau_fine, gamma_fine, prominence_frac=0.01):
    """Detect positive peaks from gamma (or residual after negative subtraction)."""
    ln_tau = np.log(tau_fine)
    abs_max = np.max(np.abs(gamma_fine))
    min_prom = prominence_frac * abs_max
    gamma_pos = np.maximum(gamma_fine, 0.0)

    peaks_idx, _ = find_peaks(gamma_pos, prominence=min_prom)
    if len(peaks_idx) == 0 and np.max(gamma_pos) > 0:
        peaks_idx = np.array([np.argmax(gamma_pos)])

    pos_peaks = []
    for idx in peaks_idx:
        if gamma_pos[idx] <= 0:
            continue
        half_max = gamma_pos[idx] / 2.0
        sigma_est = 0.5
        sigma_L, sigma_R = sigma_est, sigma_est
        for j in range(idx - 1, -1, -1):
            if gamma_pos[j] <= half_max:
                sigma_L = max((ln_tau[idx] - ln_tau[j]) / np.sqrt(2 * np.log(2)), 0.05)
                break
        for j in range(idx + 1, len(gamma_pos)):
            if gamma_pos[j] <= half_max:
                sigma_R = max((ln_tau[j] - ln_tau[idx]) / np.sqrt(2 * np.log(2)), 0.05)
                break
        pos_peaks.append({
            'A': gamma_fine[idx],
            'mu': ln_tau[idx],
            'sigma_L': sigma_L,
            'sigma_R': sigma_R,
            'tau': tau_fine[idx],
        })

    pos_peaks.sort(key=lambda p: p['tau'])
    return pos_peaks


def generate_initial_guess(detected_peaks, N_peaks):
    sorted_peaks = sorted(detected_peaks, key=lambda p: abs(p['A']), reverse=True)
    guess = []
    for i in range(N_peaks):
        if i < len(sorted_peaks):
            pk = sorted_peaks[i]
            guess.extend([pk['A'], pk['mu'], pk['sigma_L'], pk['sigma_R']])
        else:
            ln_tau_min = min(p['mu'] for p in detected_peaks)
            ln_tau_max = max(p['mu'] for p in detected_peaks)
            frac = (i + 1) / (N_peaks + 1)
            mu = ln_tau_min + frac * (ln_tau_max - ln_tau_min)
            sign_val = 1.0 if detected_peaks[0]['A'] >= 0 else -1.0
            guess.extend([sign_val * 0.001, mu, 1.0, 1.0])
    return np.array(guess)


def fit_gaussians(ln_tau, gamma, N_peaks, initial_guess, sign='positive'):
    lower = []
    upper = []
    for _ in range(N_peaks):
        if sign == 'positive':
            lower.extend([0.0, -np.inf, 0.01, 0.01])
            upper.extend([np.inf, np.inf, np.inf, np.inf])
        else:
            lower.extend([-np.inf, -np.inf, 0.01, 0.01])
            upper.extend([0.0, np.inf, np.inf, np.inf])

    expected_len = N_peaks * PARAMS_PER_PEAK
    if len(initial_guess) > expected_len:
        initial_guess = initial_guess[:expected_len]
    elif len(initial_guess) < expected_len:
        last = initial_guess[-PARAMS_PER_PEAK:]
        while len(initial_guess) < expected_len:
            initial_guess = np.append(initial_guess, last)

    for k in range(N_peaks):
        base = k * PARAMS_PER_PEAK
        if sign == 'positive' and initial_guess[base] < 0:
            initial_guess[base] = abs(initial_guess[base])
        elif sign == 'negative' and initial_guess[base] > 0:
            initial_guess[base] = -abs(initial_guess[base])
        for s_offset in [2, 3]:
            if initial_guess[base + s_offset] < 0.01:
                initial_guess[base + s_offset] = 0.1

    try:
        popt, pcov = curve_fit(
            multi_gaussian, ln_tau, gamma, p0=initial_guess,
            bounds=(lower, upper), maxfev=10000
        )
        gamma_fit = multi_gaussian(ln_tau, *popt)
        return popt, pcov, gamma_fit
    except Exception as e:
        print(f"  WARNING: Gaussian fit failed for N={N_peaks}: {e}")
        return None, None, None


def compute_aic_bic(y_data, y_fit, k):
    n = len(y_data)
    rss = np.sum((y_data - y_fit)**2)
    if rss <= 0:
        rss = 1e-300
    aic = n * np.log(rss/n) + 2*k
    bic = n * np.log(rss/n) + k*np.log(n)
    return aic, bic


def select_n_peaks(ln_tau, gamma, detected_peaks, n_range, sign='positive', verbose=True):
    scan_results = []
    y_mean = np.mean(gamma)
    tss = np.sum((gamma - y_mean)**2)
    label = "positive" if sign == 'positive' else "negative"

    for N in range(n_range[0], n_range[1]):
        guess = generate_initial_guess(detected_peaks, N)
        popt, _, gamma_fit = fit_gaussians(ln_tau, gamma, N, guess, sign=sign)
        if popt is None:
            continue
        k = N * PARAMS_PER_PEAK
        rss = np.sum((gamma - gamma_fit)**2)
        rmse = np.sqrt(rss/len(gamma))
        r2 = 1.0 - rss/tss if tss > 0 else 0.0
        aic, bic = compute_aic_bic(gamma, gamma_fit, k)
        scan_results.append({
            'N': N, 'k': k, 'AIC': aic, 'BIC': bic,
            'RMSE': rmse, 'R2': r2, 'popt': popt, 'gamma_fit': gamma_fit,
        })

    if not scan_results:
        return None, []

    best = min(scan_results, key=lambda x: x['BIC'])
    best_N = best['N']

    if verbose:
        print(f"\n  --- {label.upper()} peaks AIC/BIC scan ---")
        print(f"  {'N':<4} {'k':<5} {'RMSE':<14} {'R2':<10} {'AIC':<14} {'BIC':<14} {'dBIC':<8}")
        print("  " + "-" * 75)
        for r in scan_results:
            d = r['BIC'] - best['BIC']
            marker = " ***" if r['N'] == best_N else ""
            print(f"  {r['N']:<4} {r['k']:<5} {r['RMSE']:.6e} {r['R2']:.6f} "
                  f"{r['AIC']:>12.2f} {r['BIC']:>12.2f} {d:>7.2f}{marker}")
        print(f"  BIC-optimal: N = {best_N} ({label})")

    return best_N, scan_results


# ---- RC Deconvolution ----

def deconvolve_rc(popt, N_peaks, ln_tau_fine, sign='positive', start_num=1):
    rc_elements = []
    for k in range(N_peaks):
        b = k * PARAMS_PER_PEAK
        A_k, mu_k, sigL, sigR = popt[b], popt[b+1], popt[b+2], popt[b+3]
        R_k = A_k * np.sqrt(np.pi/2) * (sigL + sigR)
        tau_k = np.exp(mu_k)
        f_k = 1.0/(2*np.pi*tau_k)
        tau_lo = np.exp(mu_k - 3*sigL)
        tau_hi = np.exp(mu_k + 3*sigR)
        gamma_k = split_gaussian_single(ln_tau_fine, A_k, mu_k, sigL, sigR)
        R_area = np.trapezoid(gamma_k, ln_tau_fine)
        peak_type = 'capacitive' if R_k >= 0 else 'inductive'
        C_k = tau_k/R_k if abs(R_k) > 1e-15 else np.nan
        rc_elements.append({
            'A_k': A_k, 'mu_k': mu_k, 'sigma_L': sigL, 'sigma_R': sigR,
            'R_k': R_k, 'R_area': R_area, 'C_k': C_k,
            'tau_k': tau_k, 'f_k': f_k,
            'tau_lo': tau_lo, 'tau_hi': tau_hi,
            'freq_lo': 1.0/(2*np.pi*tau_hi), 'freq_hi': 1.0/(2*np.pi*tau_lo),
            'gamma_peak': A_k, 'peak_type': peak_type, 'sign': sign,
        })
    rc_elements.sort(key=lambda x: x['tau_k'])
    for i, rc in enumerate(rc_elements):
        num = start_num + i
        if sign == 'negative':
            rc['peak_label'] = f"P-{num}"
            rc['peak_num'] = -num
        else:
            rc['peak_label'] = f"P{num}"
            rc['peak_num'] = num
    return rc_elements


def refine_joint(ln_tau, gamma, popt_pos, N_pos, popt_neg, N_neg, neg_segments=None):
    """Phase 3: Joint refinement. Positive peaks capped at local DRT max.
    Negative peaks constrained by zero-crossing boundaries."""
    N_total = N_pos + N_neg
    if N_total == 0:
        return popt_pos, popt_neg, np.zeros_like(gamma)

    # Order: [neg_params..., pos_params...] so negative peaks come first
    # But we keep [pos_params..., neg_params...] for consistency with deconvolve_rc
    p0 = []
    if popt_pos is not None:
        p0.extend(popt_pos.tolist())
    if popt_neg is not None:
        p0.extend(popt_neg.tolist())
    p0 = np.array(p0)

    gamma_pos_clip = np.maximum(gamma, 0.0)

    lower = []
    upper = []
    # Positive peak bounds
    for k in range(N_pos):
        b = k * PARAMS_PER_PEAK
        mu_k = p0[b+1]
        sig_max = max(p0[b+2], p0[b+3])
        mask = (ln_tau >= mu_k - 3*sig_max) & (ln_tau <= mu_k + 3*sig_max)
        A_max = np.max(gamma_pos_clip[mask]) * 1.05 if np.any(mask) else np.max(gamma_pos_clip) * 1.05
        A_max = max(A_max, p0[b])
        lower.extend([0.0, -np.inf, 0.01, 0.01])
        upper.extend([A_max, np.inf, np.inf, np.inf])

    # Negative peak bounds — constrained by zero crossings
    for k in range(N_neg):
        b = (N_pos + k) * PARAMS_PER_PEAK
        A_k = p0[b]
        mu_k = p0[b+1]
        sigL_k = p0[b+2]
        sigR_k = p0[b+3]

        # Tight A bounds: 50% to 150% of current value
        A_lo = min(A_k * 1.5, A_k * 0.5)
        A_hi = max(A_k * 1.5, A_k * 0.5)
        A_hi = min(A_hi, -1e-10)

        # Sigma bounds from zero crossings if available
        if neg_segments is not None and k < len(neg_segments):
            seg = neg_segments[k]
            max_sigL = max((mu_k - seg['left_zero']) / 2.0, sigL_k * 1.5)
            max_sigR = max((seg['right_zero'] - mu_k) / 2.0, sigR_k * 1.5)
        else:
            max_sigL = sigL_k * 3.0
            max_sigR = sigR_k * 3.0

        lower.extend([A_lo, mu_k - 0.5, sigL_k * 0.3, sigR_k * 0.3])
        upper.extend([A_hi, mu_k + 0.5, max_sigL, max_sigR])

    try:
        popt, _ = curve_fit(
            multi_gaussian, ln_tau, gamma, p0=p0,
            bounds=(lower, upper), maxfev=20000
        )
        gamma_fit = multi_gaussian(ln_tau, *popt)
        n_pos_params = N_pos * PARAMS_PER_PEAK
        popt_pos_r = popt[:n_pos_params] if N_pos > 0 else popt_pos
        popt_neg_r = popt[n_pos_params:] if N_neg > 0 else popt_neg

        rss_before = np.sum((gamma - multi_gaussian(ln_tau, *p0))**2)
        rss_after = np.sum((gamma - gamma_fit)**2)
        pct = (1 - rss_after/rss_before)*100 if rss_before > 0 else 0
        print(f"  Joint refinement: RSS improved {pct:.1f}%")
        return popt_pos_r, popt_neg_r, gamma_fit
    except Exception as e:
        print(f"  WARNING: Joint refinement failed ({e}), using two-phase result")
        gamma_fit = multi_gaussian(ln_tau, *p0)
        return popt_pos, popt_neg, gamma_fit


def compute_gaussian_impedance(A_k, mu_k, sigma_L, sigma_R, freq, n_quad=500):
    ln_tau_lo = mu_k - 5*sigma_L
    ln_tau_hi = mu_k + 5*sigma_R
    ln_tau_q = np.linspace(ln_tau_lo, ln_tau_hi, n_quad)
    tau_q = np.exp(ln_tau_q)
    gamma_q = split_gaussian_single(ln_tau_q, A_k, mu_k, sigma_L, sigma_R)
    d_ln_tau = ln_tau_q[1] - ln_tau_q[0]
    omega = 2*np.pi*freq
    Z_k = np.zeros(len(freq), dtype=complex)
    for q in range(n_quad):
        Z_k += gamma_q[q]/(1 + 1j*omega*tau_q[q]) * d_ln_tau
    return Z_k


def compute_total_impedance(rc_elements, freq, R_ohm, L):
    omega = 2*np.pi*freq
    Z_total = R_ohm + 1j*omega*L
    Z_individual = []
    for rc in rc_elements:
        Z_k = compute_gaussian_impedance(
            rc['A_k'], rc['mu_k'], rc['sigma_L'], rc['sigma_R'], freq)
        Z_total += Z_k
        Z_individual.append(Z_k)
    return Z_total, Z_individual


def main():
    print("=" * 60)
    print("DRT Split-Gaussian Peak Fitting (v3.2 Negative-First)")
    print("=" * 60)

    tau, gamma, dataset_name = load_fine_csv(INPUT_CSV)
    ln_tau = np.log(tau)
    print(f"  File: {os.path.basename(INPUT_CSV)}")
    print(f"  {len(tau)} points, tau = [{tau.min():.2e}, {tau.max():.2e}] s")

    save_dir = os.path.join(os.path.dirname(INPUT_CSV), f"{dataset_name}_peakfit")
    if os.name == 'nt' and not save_dir.startswith('\\\\?\\'):
        save_dir_long = '\\\\?\\' + os.path.abspath(save_dir)
    else:
        save_dir_long = save_dir
    os.makedirs(save_dir_long, exist_ok=True)

    # Load EIS and auto-fit HFR/L
    Z_exp_re, Z_exp_im, freq_eis = None, None, None
    R_ohm, L_h = R_OHM, L_H
    if EIS_CSV:
        freq_eis, Z_exp_re, Z_exp_im = load_eis_csv(EIS_CSV)
        print(f"  EIS: {len(freq_eis)} points loaded")
        if R_ohm is None or L_h is None:
            R_auto, L_auto = auto_fit_hfr_L(freq_eis, Z_exp_re, Z_exp_im)
            if R_auto is not None and R_ohm is None:
                R_ohm = R_auto
                print(f"  Auto-fit HFR: R_ohm = {R_ohm*1e3:.3f} mOhm")
            if L_auto is not None and L_h is None:
                L_h = L_auto
                print(f"  Auto-fit L:   L = {L_h*1e6:.4f} uH")
    if R_ohm is None:
        R_ohm = 0.0
    if L_h is None:
        L_h = 0.0

    # ================================================================
    # Phase 1: Fit NEGATIVE peaks first (constrained by zero crossings)
    # ================================================================
    print("\nPhase 1: Fitting NEGATIVE peaks (constrained by zero crossings)...")
    popt_neg, N_neg, gamma_neg_fit, neg_segments = fit_negative_segments(ln_tau, gamma)

    if N_neg > 0:
        print(f"  Found {N_neg} negative segment(s):")
        for k in range(N_neg):
            b = k * PARAMS_PER_PEAK
            print(f"    P-{k+1}: A={popt_neg[b]:.4e}, mu={popt_neg[b+1]:.3f}, "
                  f"sigL={popt_neg[b+2]:.3f}, sigR={popt_neg[b+3]:.3f}")
    else:
        print("  No negative segments found.")

    # ================================================================
    # Phase 2: Fit POSITIVE peaks on residual (gamma - negative_fit)
    # ================================================================
    residual_pos = gamma - gamma_neg_fit
    gamma_pos_target = np.maximum(residual_pos, 0.0)

    print("\nPhase 2: Fitting POSITIVE peaks on residual...")
    # Detect peaks from ORIGINAL gamma to count real maxima (not residual artifacts)
    pos_detected_orig = detect_positive_peaks(tau, gamma, PEAK_PROMINENCE)
    # Use residual for initial guesses (better sigma estimates after neg subtraction)
    pos_detected = detect_positive_peaks(tau, residual_pos, PEAK_PROMINENCE)
    # Cap at number of real maxima from original signal
    n_real_maxima = len(pos_detected_orig)
    print(f"  Detected {n_real_maxima} positive maxima in original DRT")

    popt_pos = None
    N_pos = 0
    pos_scan = []

    if pos_detected:
        if N_PEAKS_MODE in ('auto', 'both'):
            n_range = N_PEAKS_RANGE
            if n_range[1] == 0:
                n_range = (n_range[0], n_real_maxima + 1)
            n_range = (max(1, n_range[0]), max(n_range[0]+1, n_range[1]))
            best_N_pos, pos_scan = select_n_peaks(
                ln_tau, gamma_pos_target, pos_detected, n_range, sign='positive')
            if best_N_pos is not None:
                best_result = next(r for r in pos_scan if r['N'] == best_N_pos)
                popt_pos = best_result['popt']
                N_pos = best_N_pos
        elif N_PEAKS_MODE == 'fixed':
            N_pos = N_PEAKS_FIXED
            guess = generate_initial_guess(pos_detected, N_pos)
            popt_pos, _, _ = fit_gaussians(
                ln_tau, gamma_pos_target, N_pos, guess, sign='positive')
            if popt_pos is None:
                N_pos = 0

    print(f"  Positive fit: N = {N_pos}")

    # ================================================================
    # Phase 3: Joint refinement on full gamma
    # ================================================================
    print("\nPhase 3: Joint refinement on full gamma...")
    popt_pos, popt_neg, gamma_fit = refine_joint(
        ln_tau, gamma, popt_pos, N_pos, popt_neg, N_neg,
        neg_segments=neg_segments)
    N_total = N_pos + N_neg

    # ================================================================
    # RC Deconvolution
    # ================================================================
    rc_pos = deconvolve_rc(popt_pos, N_pos, ln_tau, sign='positive', start_num=1) if popt_pos is not None else []
    rc_neg = deconvolve_rc(popt_neg, N_neg, ln_tau, sign='negative', start_num=1) if popt_neg is not None else []
    rc_elements = rc_pos + rc_neg

    # Print summary
    print(f"\n  {'Peak':<8} {'Type':<12} {'R_k(Ohm)':<14} {'R_area(Ohm)':<14} {'C_k(F)':<14} "
          f"{'tau_k(s)':<14} {'f_k(Hz)':<14} {'A_k':<14} {'sigL':<10} {'sigR':<10}")
    print("  " + "-" * 130)
    R_total_pos = 0.0
    R_total_neg = 0.0
    for rc in rc_elements:
        C_str = f"{rc['C_k']:.4e}" if np.isfinite(rc['C_k']) else "N/A"
        print(f"  {rc['peak_label']:<8} {rc['peak_type']:<12} {rc['R_k']:.4e}    "
              f"{rc['R_area']:.4e}    {C_str:<14}{rc['tau_k']:.4e}    {rc['f_k']:.4e}    "
              f"{rc['A_k']:.4e}    {rc['sigma_L']:.4f}    {rc['sigma_R']:.4f}")
        if rc['sign'] == 'positive':
            R_total_pos += rc['R_k']
        else:
            R_total_neg += rc['R_k']

    print(f"\n  R_total (positive) = {R_total_pos:.4e} Ohm")
    if N_neg > 0:
        print(f"  R_total (negative) = {R_total_neg:.4e} Ohm")
    print(f"  R_ohm = {R_ohm:.4e} Ohm, L = {L_h:.4e} H")

    # Impedance reconstruction
    freq = freq_eis if freq_eis is not None else np.sort(1.0/tau)
    Z_total, Z_individual = compute_total_impedance(rc_elements, freq, R_ohm, L_h)

    peak_colors, peak_labels = assign_peak_colors(rc_elements)

    # ---- CSV EXPORTS ----
    peak_rows = []
    for i, rc in enumerate(rc_elements):
        Z_k = Z_individual[i]
        idx_peak = np.argmin(np.abs(freq - rc['f_k']))
        Z_k_at_peak = Z_k[idx_peak]
        peak_rows.append({
            'Peak': rc['peak_label'], 'Type': rc['peak_type'],
            'Sign': rc['sign'], 'Category': peak_labels[i],
            'A_k': rc['A_k'], 'mu_k': rc['mu_k'],
            'sigma_L': rc['sigma_L'], 'sigma_R': rc['sigma_R'],
            'R_k_Ohm': rc['R_k'], 'R_area_Ohm': rc['R_area'],
            'C_k_F': rc['C_k'], 'tau_k_s': rc['tau_k'], 'f_k_Hz': rc['f_k'],
            'gamma_peak_Ohm': rc['gamma_peak'],
            'tau_lo_s': rc['tau_lo'], 'tau_hi_s': rc['tau_hi'],
            'freq_lo_Hz': rc['freq_lo'], 'freq_hi_Hz': rc['freq_hi'],
            'Z_re_at_peak_Ohm': Z_k_at_peak.real,
            'Z_im_at_peak_Ohm': Z_k_at_peak.imag,
            'Z_mag_at_peak_Ohm': np.abs(Z_k_at_peak),
        })
    df_peaks = pd.DataFrame(peak_rows)
    csv1 = os.path.join(save_dir_long, f"{dataset_name}_gaussian_peaks.csv")
    try:
        df_peaks.to_csv(csv1, index=False, float_format='%.6e')
        print(f"\n  Saved: {csv1}")
    except (OSError, PermissionError) as e:
        print(f"\n  WARNING: Could not save peaks CSV: {e}")

    rc_data = {'freq_Hz': freq, 'Z_total_re': Z_total.real, 'Z_total_im': Z_total.imag}
    for j_rc, rc in enumerate(rc_elements):
        Z_k = Z_individual[j_rc]
        rc_data[f'Z_{rc["peak_label"]}_re'] = Z_k.real
        rc_data[f'Z_{rc["peak_label"]}_im'] = Z_k.imag
    df_z = pd.DataFrame(rc_data)
    csv2 = os.path.join(save_dir_long, f"{dataset_name}_RC_impedance.csv")
    try:
        df_z.to_csv(csv2, index=False, float_format='%.8e')
        print(f"  Saved: {csv2}")
    except (OSError, PermissionError) as e:
        print(f"  WARNING: Could not save RC impedance CSV: {e}")

    # ---- PLOTS ----
    setup_plot_style()
    print("\nGenerating plots...")

    # AIC/BIC plot
    if N_PEAKS_MODE in ('auto', 'both') and pos_scan:
        ns = [r['N'] for r in pos_scan]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ns, [r['AIC'] for r in pos_scan], 'o-', color='steelblue', linewidth=2, markersize=8, label='AIC')
        ax.plot(ns, [r['BIC'] for r in pos_scan], 's-', color='firebrick', linewidth=2, markersize=8, label='BIC')
        ax.axvline(N_pos, color='firebrick', linestyle='--', alpha=0.5, label=f'Best N={N_pos}')
        ax.set_xlabel('Number of peaks'); ax.set_ylabel('Information Criterion')
        ax.set_title(f'{dataset_name}: AIC/BIC (positive peaks)')
        ax.legend(fontsize=FONT_SIZE*0.5); ax.set_xticks(ns)
        plt.tight_layout()
        try:
            fig.savefig(os.path.join(save_dir_long, f"{dataset_name}_AIC_BIC.png"), dpi=200, bbox_inches='tight')
        except (OSError, PermissionError):
            pass
        plt.close(fig)

    # Gaussian decomposition
    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax_main, ax_res = axes

    ax_main.plot(tau, gamma, '-', color='black', linewidth=LINE_WIDTH, label='DRT')
    ax_main.plot(tau, gamma_fit, '--', color='red', linewidth=LINE_WIDTH, label='Combined fit')

    for i, rc in enumerate(rc_pos):
        idx = rc_elements.index(rc)
        c = peak_colors[idx]
        single = split_gaussian_single(ln_tau, rc['A_k'], rc['mu_k'], rc['sigma_L'], rc['sigma_R'])
        ax_main.fill_between(tau, 0, single, alpha=0.35, color=c, label=peak_labels[idx])

    for i, rc in enumerate(rc_neg):
        idx = rc_elements.index(rc)
        c = peak_colors[idx]
        single = split_gaussian_single(ln_tau, rc['A_k'], rc['mu_k'], rc['sigma_L'], rc['sigma_R'])
        ax_main.fill_between(tau, 0, single, alpha=0.25, color=c,
                             hatch='--', edgecolor=c, linewidth=0.5, label=peak_labels[idx])

    ax_main.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax_main.set_xscale('log')
    ax_main.set_ylabel(r'$\gamma$ ($\Omega$)')
    ax_main.set_title(f'{dataset_name}: Split-Gaussian (Neg-First) '
                      f'(N+={N_pos}, N-={N_neg})')
    ax_main.legend(fontsize=FONT_SIZE*0.35, loc='best')

    ax_res.plot(tau, gamma - gamma_fit, '-', color='navy', linewidth=1)
    ax_res.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax_res.set_xscale('log')
    ax_res.set_xlabel(r'$\tau$ (s)'); ax_res.set_ylabel('Residual')
    plt.tight_layout()
    try:
        fig.savefig(os.path.join(save_dir_long, f"{dataset_name}_gaussian_fit.png"), dpi=200, bbox_inches='tight')
        print(f"  Saved: ...{dataset_name}_gaussian_fit.png")
    except (OSError, PermissionError) as e:
        print(f"  WARNING: {e}")
    plt.close(fig)

    # Nyquist
    fig, ax = plt.subplots(figsize=(10, 10))
    if Z_exp_re is not None:
        ax.plot(Z_exp_re, -Z_exp_im, 'o', color='black', markersize=MARKER_SIZE,
                markeredgewidth=LINE_WIDTH, markerfacecolor='none', label='Experimental')
    ax.plot(Z_total.real, -Z_total.imag, '-', color='red', linewidth=LINE_WIDTH, label='RC reconstruction')
    for i, rc in enumerate(rc_elements):
        c = peak_colors[i]
        Z_k = Z_individual[i]
        R_offset = R_ohm + sum(rc2['R_k'] for rc2 in rc_elements[:i] if rc2['R_k'] > 0)
        ls = '--' if rc['sign'] == 'negative' else '-'
        ax.plot(Z_k.real + R_offset, -Z_k.imag, ls, color=c, linewidth=1.5,
                label=f"{peak_labels[i]}: R={rc['R_k']*1e3:.1f}m$\\Omega$")
    ax.set_xlabel(r'$Z_{real}$ ($\Omega$)'); ax.set_ylabel(r'$-Z_{imag}$ ($\Omega$)')
    ax.set_title(f'{dataset_name}: Nyquist (RC decomposition)')
    ax.set_aspect('equal'); ax.legend(fontsize=FONT_SIZE*0.35, loc='best')
    plt.tight_layout()
    try:
        fig.savefig(os.path.join(save_dir_long, f"{dataset_name}_Nyquist_RC.png"), dpi=200, bbox_inches='tight')
        print(f"  Saved: ...{dataset_name}_Nyquist_RC.png")
    except (OSError, PermissionError) as e:
        print(f"  WARNING: {e}")
    plt.close(fig)

    # Individual RC arcs
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, rc in enumerate(rc_elements):
        c = peak_colors[i]
        Z_k = Z_individual[i]
        ls = '--' if rc['sign'] == 'negative' else '-'
        ax.plot(Z_k.real, -Z_k.imag, ls, color=c, linewidth=LINE_WIDTH,
                label=f"{peak_labels[i]}: R={rc['R_k']*1e3:.2f}m$\\Omega$, $\\tau$={rc['tau_k']:.2e}s")
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel(r'$Z_{real}$ ($\Omega$)'); ax.set_ylabel(r'$-Z_{imag}$ ($\Omega$)')
    ax.set_title(f'{dataset_name}: Individual RC Elements')
    ax.legend(fontsize=FONT_SIZE*0.35, loc='best')
    plt.tight_layout()
    try:
        fig.savefig(os.path.join(save_dir_long, f"{dataset_name}_RC_individual.png"), dpi=200, bbox_inches='tight')
        print(f"  Saved: ...{dataset_name}_RC_individual.png")
    except (OSError, PermissionError) as e:
        print(f"  WARNING: {e}")
    plt.close(fig)

    print(f"\n{'='*60}")
    print(f"  N_peaks = {N_total} ({N_pos} pos + {N_neg} neg)")
    print(f"  R_total (pos) = {R_total_pos:.4e} Ohm")
    print(f"  R_ohm = {R_ohm:.4e} Ohm, L = {L_h:.4e} H")
    print(f"  Output: {save_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
