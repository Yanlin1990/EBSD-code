"""EBSD Prototype - PySide6 + DefDAP

A graphical application for Electron Backscatter Diffraction (EBSD)
data visualisation and analysis.  Built on PySide6 (Qt 6) and the
DefDAP library, it supports:

* Euler / IPF / Band-Contrast / KAM / Misorientation / Grain maps
* BC + IPF blending with multiple modes
* HAGB / LAGB boundary overlay with smoothing
* Twin-boundary detection (cubic & hexagonal systems)
* Pole-figure and inverse-pole-figure windows
* Non-indexed pixel filling and denoising
* Image export (PNG / TIFF / JPEG) and CTF export

Key improvements over the initial version
------------------------------------------
- Map is cached after first load; no redundant disk reads.
- Pixel-level loops replaced by vectorised NumPy / SciPy operations.
- QTimer debounce prevents repeated re-renders when widgets change.
- Heavy computation runs in a QThread worker so the GUI stays responsive.
- Duplicate fill logic unified into a single generic helper.
- Widget enable/disable uses an auto-registration pattern.
- Magic numbers replaced by named constants.
- Proper ``logging`` replaces bare ``traceback.print_exc()``.
- Silent ``except: pass`` in twin-axis check fixed (returns ``False``).
"""
from __future__ import annotations

import logging
import time
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set

import numpy as np
from scipy.ndimage import (
    median_filter,
    gaussian_filter,
    label as ndlabel,
    binary_dilation as _binary_dilation,
    gaussian_filter1d,
    distance_transform_edt,
)
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from skimage.measure import find_contours

from PySide6.QtCore import Qt, QTimer, QObject, Signal, QThread
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import defdap.ebsd as ebsd
from defdap.quat import Quat

logger = logging.getLogger(__name__)

__all__ = ["EbsdMainWindow"]

# ====================================================================
# Named constants (replace former magic numbers)
# ====================================================================

MAX_POLE_FIGURE_POINTS: int = 5000
"""Maximum number of orientations sampled for pole-figure scatter."""

QUATERNION_SINGULARITY_THRESHOLD: float = 1e-6
"""Below this sin(half-angle) value the rotation axis is ill-defined."""

BRIGHTNESS_CHANGE_EPSILON: float = 0.01
"""Minimum BC-brightness deviation from 1.0 to trigger adjustment."""

CONTRAST_CHANGE_EPSILON: float = 0.01
"""Minimum BC-contrast deviation from 1.0 to trigger adjustment."""

SMOOTH_CONTOUR_LENGTH_SCALE: float = 30.0
"""Contour length (px) used to scale adaptive Gaussian sigma."""

SMOOTH_CONTOUR_MIN_SIGMA_FRAC: float = 0.2
"""Minimum fraction of sigma kept when contour is very short."""

EULER_ZERO_THRESHOLD: float = 1e-10
"""Sum-of-absolute Euler angles below this → treated as non-indexed."""

MULTIPLY_BLEND_BOOST: float = 1.3
"""Brightness multiplier applied after 'multiply' BC+IPF blend."""

DEBOUNCE_MS: int = 300
"""Milliseconds to wait after the last widget change before refreshing."""

MIN_CONTOUR_POINTS: int = 4
"""Contours shorter than this are discarded."""

MIN_TWIN_CONTOUR_POINTS: int = 3
"""Twin-boundary contours shorter than this are discarded."""

MIN_LAGB_SEGMENT_POINTS: int = 3
"""LAGB segments shorter than this are discarded."""

MAX_GRAIN_SAMPLE: int = 500
"""Maximum pixels sampled per grain when computing average orientation."""


# ====================================================================
# Twin orientation-relationship definitions (孪晶取向关系定义)
# ====================================================================

TWIN_SYSTEMS: Dict[str, List[dict]] = {
    "cubic": [
        {
            "name": "Σ3 {111}<112>",
            "angle_deg": 60.0,
            "axis": np.array([1, 1, 1], dtype=float),
            "tolerance_deg": 5.0,
            "color": "red",
        },
        {
            "name": "Σ9 {114}<221>",
            "angle_deg": 38.94,
            "axis": np.array([1, 1, 0], dtype=float),
            "tolerance_deg": 5.0,
            "color": "blue",
        },
        {
            "name": "Σ27a",
            "angle_deg": 31.59,
            "axis": np.array([1, 1, 0], dtype=float),
            "tolerance_deg": 5.0,
            "color": "green",
        },
    ],
    "hexagonal": [
        {
            "name": "{10-12} tension",
            "angle_deg": 86.3,
            "axis": np.array([1, -2, 1, 0], dtype=float),
            "tolerance_deg": 5.0,
            "color": "red",
        },
        {
            "name": "{10-11} compression",
            "angle_deg": 56.2,
            "axis": np.array([1, -2, 1, 0], dtype=float),
            "tolerance_deg": 5.0,
            "color": "blue",
        },
        {
            "name": "{10-13}",
            "angle_deg": 64.0,
            "axis": np.array([1, -2, 1, 0], dtype=float),
            "tolerance_deg": 5.0,
            "color": "green",
        },
        {
            "name": "{11-21}",
            "angle_deg": 34.8,
            "axis": np.array([1, 0, -1, 0], dtype=float),
            "tolerance_deg": 5.0,
            "color": "magenta",
        },
        {
            "name": "{11-22}",
            "angle_deg": 64.4,
            "axis": np.array([1, 0, -1, 0], dtype=float),
            "tolerance_deg": 5.0,
            "color": "cyan",
        },
    ],
}


# ====================================================================
# Crystallographic helpers
# ====================================================================


def _mb_to_miller_dir(v4: np.ndarray) -> np.ndarray:
    """Convert 4-index Miller-Bravais direction to 3-index Miller."""
    u, v, t, w = v4
    return np.array([u - t, v - t, w], dtype=float)


def _normalize(v: np.ndarray) -> np.ndarray:
    """Return unit vector; returns *v* unchanged if norm is zero."""
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def check_twin_relation(
    q1: Quat,
    q2: Quat,
    sym_group: str,
    twin_def: dict,
) -> bool:
    """Check twin relationship by misorientation angle only."""
    mis_ori_cos = q1.mis_ori(q2, sym_group, return_quat=0)
    mis_ori_cos = min(mis_ori_cos, 1.0)
    mis_angle_deg = 2.0 * np.arccos(mis_ori_cos) * 180.0 / np.pi
    return abs(mis_angle_deg - twin_def["angle_deg"]) <= twin_def["tolerance_deg"]


def check_twin_with_axis(
    q1: Quat,
    q2: Quat,
    sym_group: str,
    twin_def: dict,
) -> bool:
    """Check twin relationship by both misorientation angle and axis."""
    mis_ori_cos, mis_quat = q1.mis_ori(q2, sym_group, return_quat=2)
    mis_ori_cos = min(mis_ori_cos, 1.0)
    mis_angle_deg = 2.0 * np.arccos(mis_ori_cos) * 180.0 / np.pi
    if abs(mis_angle_deg - twin_def["angle_deg"]) > twin_def["tolerance_deg"]:
        return False
    try:
        dq = mis_quat * q1.conjugate
        dq_coef = dq.quat_coef
        sin_half = np.sqrt(1.0 - dq_coef[0] ** 2)
        if sin_half < QUATERNION_SINGULARITY_THRESHOLD:
            return True
        axis_m = _normalize(dq_coef[1:4] / sin_half)
        axis_t = twin_def["axis"].copy()
        if len(axis_t) == 4:
            axis_t = _mb_to_miller_dir(axis_t)
        axis_t = _normalize(axis_t)
        if abs(np.dot(axis_m, axis_t)) > np.cos(
            np.radians(twin_def["tolerance_deg"])
        ):
            return True
        return False
    except Exception:
        logger.warning(
            "Twin axis check failed for %s — treating as non-twin",
            twin_def["name"],
            exc_info=True,
        )
        return False


# ====================================================================
# Vectorised pixel-fill helpers
# ====================================================================


def _vectorised_neighbor_fill_2d(
    img: np.ndarray,
    mask: np.ndarray,
    max_iter: int,
) -> np.ndarray:
    """Iterative 8-neighbour mean fill for a 2-D array (vectorised).

    Each iteration, every masked pixel that has at least one valid
    neighbour is replaced by the mean of its valid neighbours.
    """
    img = img.astype(np.float64)
    remaining = mask.copy()
    h, w = img.shape

    for _ in range(max_iter):
        if not np.any(remaining):
            break
        padded = np.pad(img, 1, mode="edge")
        valid_pad = np.pad(~remaining, 1, constant_values=False)
        neighbour_sum = np.zeros_like(img)
        neighbour_cnt = np.zeros_like(img)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                sl = padded[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w]
                vm = valid_pad[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w]
                neighbour_sum += np.where(vm, sl, 0.0)
                neighbour_cnt += vm.astype(np.float64)
        fillable = remaining & (neighbour_cnt > 0)
        if not np.any(fillable):
            break
        img[fillable] = neighbour_sum[fillable] / neighbour_cnt[fillable]
        remaining[fillable] = False

    # Remaining isolated pixels — nearest valid
    if np.any(remaining) and np.any(~remaining):
        _, nearest = distance_transform_edt(
            remaining, return_distances=True, return_indices=True,
        )
        img[remaining] = img[nearest[0][remaining], nearest[1][remaining]]
    return img


def _vectorised_neighbor_fill_rgb(
    img: np.ndarray,
    mask: np.ndarray,
    max_iter: int,
) -> np.ndarray:
    """Iterative 8-neighbour mean fill for an (H, W, C) RGB array."""
    img = img.astype(np.float64)
    remaining = mask.copy()
    h, w = mask.shape
    n_ch = img.shape[2]

    for _ in range(max_iter):
        if not np.any(remaining):
            break
        padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
        valid_pad = np.pad(~remaining, 1, constant_values=False)
        neighbour_sum = np.zeros_like(img)
        neighbour_cnt = np.zeros((h, w), dtype=np.float64)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                sl = padded[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w, :]
                vm = valid_pad[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w]
                neighbour_sum += sl * vm[..., np.newaxis]
                neighbour_cnt += vm.astype(np.float64)
        fillable = remaining & (neighbour_cnt > 0)
        if not np.any(fillable):
            break
        for ch in range(n_ch):
            img[:, :, ch][fillable] = (
                neighbour_sum[:, :, ch][fillable] / neighbour_cnt[fillable]
            )
        remaining[fillable] = False

    if np.any(remaining) and np.any(~remaining):
        _, nearest = distance_transform_edt(
            remaining, return_distances=True, return_indices=True,
        )
        for ch in range(img.shape[2]):
            img[:, :, ch][remaining] = img[:, :, ch][
                nearest[0][remaining], nearest[1][remaining]
            ]
    return img


def _vectorised_median_fill_2d(
    img: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Fill masked pixels using a full-image 3x3 median filter (vectorised)."""
    img = img.copy().astype(np.float32)
    if not np.any(mask):
        return img
    filtered = median_filter(img, size=3)
    img[mask] = filtered[mask]
    return img


def _vectorised_median_fill_rgb(
    img: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Fill masked pixels with channel-wise 3x3 median (vectorised, RGB)."""
    img = img.copy().astype(np.float32)
    if not np.any(mask):
        return img
    for ch in range(img.shape[2]):
        filtered = median_filter(img[:, :, ch], size=3)
        img[mask, ch] = filtered[mask]
    return img


def _find_neighbor_pairs_vectorised(gm: np.ndarray) -> Set[Tuple[int, int]]:
    """Find unique adjacent grain-ID pairs using vectorised operations."""
    valid = gm > 0
    # Horizontal neighbours
    h_mask = valid[:, :-1] & valid[:, 1:]
    h_left = gm[:, :-1][h_mask]
    h_right = gm[:, 1:][h_mask]
    # Vertical neighbours
    v_mask = valid[:-1, :] & valid[1:, :]
    v_top = gm[:-1, :][v_mask]
    v_bot = gm[1:, :][v_mask]
    # Combine
    all_left = np.concatenate([h_left, v_top])
    all_right = np.concatenate([h_right, v_bot])
    # Remove same-grain
    diff = all_left != all_right
    all_left = all_left[diff]
    all_right = all_right[diff]
    # Canonical ordering (min, max)
    lo = np.minimum(all_left, all_right)
    hi = np.maximum(all_left, all_right)
    pairs = np.unique(np.stack([lo, hi], axis=1), axis=0)
    return set(map(tuple, pairs))


# ====================================================================
# Quaternion average (eigenvalue method)
# ====================================================================


def _avg_quaternion_eigenvalue(quats_array: Any) -> np.ndarray:
    """Compute mean quaternion via the eigenvalue method.

    Builds M = sum(qi * qi^T) and returns the eigenvector for the
    largest eigenvalue -- the globally optimal L2 mean quaternion.
    """
    M = np.zeros((4, 4), dtype=np.float64)
    n = 0
    for q in quats_array:
        coef = np.asarray(q.quat_coef, dtype=np.float64).ravel()[:4]
        M += np.outer(coef, coef)
        n += 1
    if n == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    _, eigvecs = np.linalg.eigh(M)
    return eigvecs[:, -1]


# ====================================================================
# Pure computation helpers (thread-safe -- no Qt widget access)
# ====================================================================


def _get_non_indexed_mask_pure(m: Any) -> np.ndarray:
    """Detect non-indexed pixels. Returns bool (H, W), True = bad."""
    h = w = None
    mask = None
    try:
        phase = np.asarray(m.data.phase)
        h, w = phase.shape
        mask = phase == 0
    except Exception:
        logger.debug("No phase data for non-indexed detection")
    try:
        bc = np.asarray(m.data.band_contrast)
        if h is None:
            h, w = bc.shape
        bc_bad = bc <= 0
        mask = bc_bad if mask is None else (mask | bc_bad)
    except Exception:
        logger.debug("No band-contrast data for non-indexed detection")
    try:
        ea = np.asarray(m.data.euler_angle)
        ea_sum = np.abs(ea[0]) + np.abs(ea[1]) + np.abs(ea[2])
        ea_bad = ea_sum < EULER_ZERO_THRESHOLD
        if h is None:
            h, w = ea_bad.shape
        mask = ea_bad if mask is None else (mask | ea_bad)
    except Exception:
        logger.debug("No Euler-angle data for non-indexed detection")
    if mask is None:
        if h is None:
            return np.zeros((1, 1), dtype=bool)
        return np.zeros((h, w), dtype=bool)
    return mask


def _get_bc_array_pure(
    m: Any,
    brightness: float,
    contrast: float,
) -> np.ndarray:
    """Compute normalised band-contrast array with brightness/contrast."""
    bc = np.asarray(m.data.band_contrast).astype(np.float32)
    lo, hi = float(bc.min()), float(bc.max())
    if hi > lo:
        bc = (bc - lo) / (hi - lo)
    else:
        bc = np.zeros_like(bc)
    if abs(brightness - 1.0) > BRIGHTNESS_CHANGE_EPSILON:
        bc = bc * brightness
    if abs(contrast - 1.0) > CONTRAST_CHANGE_EPSILON:
        bc = (bc - 0.5) * contrast + 0.5
    return np.clip(bc, 0.0, 1.0)


def _normalize_rgb_pure(a: np.ndarray) -> np.ndarray:
    """Normalize an RGB image to float32 in [0, 1]."""
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    elif a.dtype != np.float32:
        a = a.astype(np.float32)
    mx = float(a.max())
    if mx > 1.5 and mx > 0:
        a = a / mx
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[:, :, :3]
    return np.clip(a, 0.0, 1.0)


def _compute_ipf_rgb_pure(m: Any, direction: np.ndarray) -> np.ndarray:
    """Compute IPF colour map for given reference direction."""
    q = np.array(m.data["orientation"])
    fq = q.ravel()
    rgb = Quat.calc_ipf_colours(fq, direction, m.crystal_sym)
    return rgb.T.reshape(q.shape + (3,)).astype(np.float32)


def _fill_non_indexed_pure(
    img: np.ndarray,
    mask: np.ndarray,
    method: str,
    max_iter: int,
) -> np.ndarray:
    """Fill non-indexed pixels in a 2-D or (H, W, C) array."""
    if method == "leave as-is" or not np.any(mask):
        return img
    img = img.copy()
    is_rgb = img.ndim == 3
    if method == "black":
        img[mask] = 0.0
        return img
    if method == "white":
        if is_rgb:
            img[mask] = 1.0
        else:
            mx = float(img[~mask].max()) if np.any(~mask) else 1.0
            img[mask] = mx
        return img
    if method == "fill (median 3\u00d73)":
        if is_rgb:
            return _vectorised_median_fill_rgb(img, mask).astype(img.dtype)
        return _vectorised_median_fill_2d(img, mask).astype(img.dtype)
    # "fill (neighbor)"
    if is_rgb:
        return _vectorised_neighbor_fill_rgb(img, mask, max_iter).astype(img.dtype)
    return _vectorised_neighbor_fill_2d(img, mask, max_iter).astype(img.dtype)


def _denoise_2d_pure(
    a: np.ndarray,
    method: str,
    median_kernel: int,
    gauss_sigma: float,
) -> np.ndarray:
    """Apply denoising to a 2-D array."""
    if method == "Median":
        k = median_kernel
        if k < 2:
            return a
        k = k if k % 2 == 1 else k + 1
        return median_filter(a.copy(), size=k)
    if method == "Gaussian":
        if gauss_sigma < 0.01:
            return a
        return gaussian_filter(
            a.copy().astype(np.float64), sigma=gauss_sigma,
        ).astype(a.dtype)
    return a


def _denoise_rgb_pure(
    a: np.ndarray,
    method: str,
    median_kernel: int,
    gauss_sigma: float,
) -> np.ndarray:
    """Apply denoising to an (H, W, C) RGB array."""
    if method == "Median":
        k = median_kernel
        if k < 2:
            return a
        k = k if k % 2 == 1 else k + 1
        a = a.copy()
        return np.dstack([
            median_filter(a[:, :, i], size=k)
            for i in range(a.shape[2])
        ])
    if method == "Gaussian":
        if gauss_sigma < 0.01:
            return a
        a = a.copy().astype(np.float64)
        for i in range(a.shape[2]):
            a[:, :, i] = gaussian_filter(a[:, :, i], sigma=gauss_sigma)
        return a.astype(np.float32)
    return a


def _blend_bc_ipf_pure(
    bc: np.ndarray,
    ipf_rgb: np.ndarray,
    alpha: float,
    mode: str,
) -> np.ndarray:
    """Blend band-contrast and IPF images."""
    bc3 = np.dstack([bc, bc, bc])
    if mode == "multiply":
        mult = bc3 * ipf_rgb
        result = bc3 * (1.0 - alpha) + mult * alpha
        result *= MULTIPLY_BLEND_BOOST
    elif mode == "soft_light":
        m_ = ipf_rgb <= 0.5
        soft = np.where(
            m_,
            bc3 - (1 - 2 * ipf_rgb) * bc3 * (1 - bc3),
            bc3 + (2 * ipf_rgb - 1) * (
                np.sqrt(np.maximum(bc3, 0)) - bc3),
        )
        result = bc3 * (1.0 - alpha) + soft * alpha
    else:  # overlay
        result = bc3 * (1.0 - alpha) + ipf_rgb * alpha
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _build_clean_grain_map_pure(
    m: Any,
    gb_angle: float,
    min_grain: int,
    hole_fill: int,
    frag_merge: int,
) -> np.ndarray:
    """Build hole-filled, fragment-merged grain ID map."""
    gm = np.asarray(m.data["grains"]).copy()
    if hole_fill > 0:
        inv = gm <= 0
        if np.any(inv):
            lh, nh = ndlabel(inv)
            for hid in range(1, nh + 1):
                hm = lh == hid
                if hm.sum() > hole_fill:
                    continue
                d = _binary_dilation(hm, iterations=1)
                b = d & ~hm
                ni = gm[b]
                ni = ni[ni > 0]
                if ni.size > 0:
                    gm[hm] = int(np.bincount(ni).argmax())
    if frag_merge > 0:
        for gid in np.unique(gm):
            if gid <= 0:
                continue
            grain_mask = gm == gid
            if grain_mask.sum() >= frag_merge:
                continue
            d = _binary_dilation(grain_mask, iterations=1)
            b = d & ~grain_mask
            ni = gm[b]
            ni = ni[(ni > 0) & (ni != gid)]
            if ni.size > 0:
                gm[grain_mask] = int(np.bincount(ni).argmax())
            else:
                gm[grain_mask] = 0
    return gm


def _smooth_contour_pure(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth a contour with adaptive Gaussian."""
    n = len(x)
    if sigma < 0.05 or n < 5:
        return x, y
    asig = max(
        sigma * min(1.0, n / SMOOTH_CONTOUR_LENGTH_SCALE),
        sigma * SMOOTH_CONTOUR_MIN_SIGMA_FRAC,
    )
    closed = abs(x[0] - x[-1]) < 1.0 and abs(y[0] - y[-1]) < 1.0
    mode = "wrap" if closed else "nearest"
    return (
        gaussian_filter1d(x, sigma=asig, mode=mode),
        gaussian_filter1d(y, sigma=asig, mode=mode),
    )


def _contours_from_grain_map_pure(
    grain_map: np.ndarray,
    sigma: float,
) -> List[np.ndarray]:
    """Extract and smooth grain boundary contours."""
    uids = np.unique(grain_map)
    uids = uids[uids > 0]
    all_c: List[np.ndarray] = []
    for gid in uids:
        mask_f = (grain_map == gid).astype(np.float32)
        for c in find_contours(mask_f, level=0.5):
            if c.shape[0] < MIN_CONTOUR_POINTS:
                continue
            y, x = c[:, 0], c[:, 1]
            x, y = _smooth_contour_pure(x, y, sigma)
            all_c.append(np.column_stack([x, y]))
    return all_c


def _contiguous_segments_pure(
    mask_1d: np.ndarray,
    min_length: int,
) -> List[Tuple[int, int]]:
    """Return (start, end) pairs of contiguous True runs in *mask_1d*."""
    segs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, val in enumerate(mask_1d):
        if val:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= min_length:
                    segs.append((start, i))
                start = None
    if start is not None and len(mask_1d) - start >= min_length:
        segs.append((start, len(mask_1d)))
    return segs


def _detect_twin_boundaries_pure(
    params: dict,
    m: Any,
    clean_gm: np.ndarray,
    neighbor_pairs: Any,
) -> dict:
    """Detect twin boundaries using eigenvalue-averaged grain orientations."""
    sym = params["crystal_sym"]
    twin_defs = params["twin_defs"]
    if not twin_defs:
        return {}

    quats = np.array(m.data["orientation"])
    strict = params["strict_axis"]
    check_fn = check_twin_with_axis if strict else check_twin_relation

    grain_ids = np.unique(clean_gm)
    grain_ids = grain_ids[grain_ids > 0]
    grain_avg_ori: Dict[int, Any] = {}

    for gid in grain_ids:
        ys, xs = np.where(clean_gm == gid)
        n_px = len(ys)
        if n_px > MAX_GRAIN_SAMPLE:
            idx = np.linspace(0, n_px - 1, MAX_GRAIN_SAMPLE).astype(int)
            ys_s, xs_s = ys[idx], xs[idx]
        else:
            ys_s, xs_s = ys, xs
        grain_quats_obj = quats[ys_s, xs_s]
        avg_coef = _avg_quaternion_eigenvalue(grain_quats_obj)
        grain_avg_ori[gid] = Quat(avg_coef)

    twin_pairs: Dict[str, set] = {td["name"]: set() for td in twin_defs}
    for g_a, g_b in neighbor_pairs:
        if g_a not in grain_avg_ori or g_b not in grain_avg_ori:
            continue
        for td in twin_defs:
            if check_fn(grain_avg_ori[g_a], grain_avg_ori[g_b], sym, td):
                twin_pairs[td["name"]].add((g_a, g_b))
                break

    result: dict = {}
    sigma = params["gb_smooth"]
    for td in twin_defs:
        pairs = twin_pairs[td["name"]]
        if not pairs:
            continue
        contours: List[np.ndarray] = []
        for g_a, g_b in pairs:
            boundary = (
                _binary_dilation(clean_gm == g_a, iterations=1)
                & (clean_gm == g_b)
            )
            if boundary.sum() < 2:
                continue
            for c in find_contours(boundary.astype(np.float32), level=0.5):
                if c.shape[0] < MIN_TWIN_CONTOUR_POINTS:
                    continue
                y_c, x_c = c[:, 0], c[:, 1]
                if len(x_c) > 4:
                    x_c, y_c = _smooth_contour_pure(x_c, y_c, sigma)
                contours.append(np.column_stack([x_c, y_c]))
        if contours:
            result[td["name"]] = {
                "contours": contours,
                "color": td["color"],
            }
    return result


def _extract_lagb_contours_pure(
    params: dict,
    m: Any,
    lagb_sub_gm: np.ndarray,
    ha_gm: np.ndarray,
) -> List[np.ndarray]:
    """Extract LAGB contours clipped to HAGB grain interiors."""
    sigma = params["lagb_smooth"]
    h_map, w_map = ha_gm.shape
    all_c: List[np.ndarray] = []
    for ha_id in np.unique(ha_gm):
        if ha_id <= 0:
            continue
        ha_mask = ha_gm == ha_id
        sub_ids = np.unique(lagb_sub_gm[ha_mask])
        sub_ids = sub_ids[sub_ids > 0]
        if len(sub_ids) <= 1:
            continue
        for sid in sub_ids:
            sm = (lagb_sub_gm == sid) & ha_mask
            if sm.sum() < 3:
                continue
            for c in find_contours(sm.astype(np.float32), level=0.5):
                if c.shape[0] < MIN_CONTOUR_POINTS:
                    continue
                y, x = c[:, 0], c[:, 1]
                ix = np.clip(np.round(x).astype(int), 0, w_map - 1)
                iy = np.clip(np.round(y).astype(int), 0, h_map - 1)
                inside = ha_gm[iy, ix] == ha_id
                segs = _contiguous_segments_pure(inside, MIN_LAGB_SEGMENT_POINTS)
                for s, e in segs:
                    sx, sy = x[s:e], y[s:e]
                    if len(sx) > 4:
                        sx, sy = _smooth_contour_pure(sx, sy, sigma)
                    all_c.append(np.column_stack([sx, sy]))
    return all_c


def _extract_image_array_static(ax: Any) -> Optional[np.ndarray]:
    """Extract the first image array from a Matplotlib axes."""
    if not ax.images:
        return None
    return np.asarray(ax.images[0].get_array()).copy()





class _RenderWorker(QObject):
    """Runs heavy map-rendering work off the main thread."""

    finished = Signal(object)  # emits the result dict
    error = Signal(str)

    def __init__(self, fn: Any, args: tuple = (), kwargs: dict | None = None):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs or {}

    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.finished.emit(result)
        except Exception as exc:
            logger.error("Worker failed: %s", exc, exc_info=True)
            self.error.emit(str(exc))


# ====================================================================
# GUI helper — Matplotlib view widget
# ====================================================================


class MplView(QWidget):
    """Embeds a Matplotlib figure + navigation toolbar in a QWidget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)


# ====================================================================
# Main window
# ====================================================================


class EbsdMainWindow(QMainWindow):
    """Main application window for EBSD analysis."""

    _SPACE_GROUP_TO_STRUCTURE: Dict[int, str] = {
        225: "FCC",
        227: "FCC",
        229: "BCC",
        194: "HCP",
        186: "HCP",
    }
    _LAUE_TO_SYSTEM: Dict[int, str] = {9: "Hexagonal", 11: "Cubic"}

    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EBSD Prototype - PySide6 + DefDAP")
        self.resize(1560, 960)

        # State
        self.current_path: Optional[Path] = None
        self.current_map: Optional[ebsd.Map] = None
        self.current_fig_ready: bool = False
        self._twin_boundary_cache: Optional[dict] = None
        self._cached_raw_map: Optional[ebsd.Map] = None
        self._cached_lagb_map: Optional[ebsd.Map] = None

        # Async rendering state
        self._render_busy: bool = False
        self._pending_params: Optional[dict] = None
        self._worker_thread: Optional[QThread] = None
        self._current_worker: Optional[_RenderWorker] = None

        # Multi-level caches -- stored as (key, value) tuples
        self._cache_mask: Tuple[Any, Any] = (None, None)
        self._cache_bc: Tuple[Any, Any] = (None, None)
        self._cache_ipf: Tuple[Any, Any] = (None, None)
        self._cache_clean_gm: Tuple[Any, Any] = (None, None)
        self._cache_neighbor_pairs: Tuple[Any, Any] = (None, None)
        self._cache_twin: Tuple[Any, Any] = (None, None)
        self._cache_lagb_grains: Tuple[Any, Any] = (None, None)

        # Widget registration list (auto enable/disable)
        self._data_widgets: List[QWidget] = []

        # Build UI
        self._build_ui()
        self._connect_signals()
        self._update_control_states(False)

        # Debounce timer
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(DEBOUNCE_MS)
        self._refresh_timer.timeout.connect(self._do_refresh)

    # ----------------------------------------------------------------
    # Widget factory helpers
    # ----------------------------------------------------------------

    def _register(self, widget: QWidget) -> QWidget:
        """Register *widget* so it is auto-enabled / disabled with data."""
        self._data_widgets.append(widget)
        return widget

    def _dspin(
        self,
        lo: float,
        hi: float,
        val: float,
        step: float = 0.1,
        dec: int = 1,
        suf: str = "",
    ) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSingleStep(step)
        s.setDecimals(dec)
        if suf:
            s.setSuffix(suf)
        s.setFixedWidth(110)
        return self._register(s)

    def _ispin(
        self,
        lo: int,
        hi: int,
        val: int,
        step: int = 1,
        suf: str = "",
    ) -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSingleStep(step)
        if suf:
            s.setSuffix(suf)
        s.setFixedWidth(110)
        return self._register(s)

    # ================================================================
    # UI layout
    # ================================================================

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        sp = QSplitter(Qt.Horizontal)
        self.control_panel = self._build_controls()
        self.viewer = MplView()
        self.info_panel = self._build_info_panel()
        sp.addWidget(self.control_panel)
        sp.addWidget(self.viewer)
        sp.addWidget(self.info_panel)
        sp.setStretchFactor(0, 0)
        sp.setStretchFactor(1, 1)
        sp.setStretchFactor(2, 0)
        sp.setSizes([380, 880, 300])
        lay = QVBoxLayout(central)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(sp)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

    # ----------------------------------------------------------------
    # Controls panel
    # ----------------------------------------------------------------

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        vbox = QVBoxLayout(panel)

        # ── File group ──
        fg = QGroupBox("File")
        fl = QGridLayout(fg)
        self.btn_open = QPushButton("Open CPR/CRC")
        self.btn_save_img = self._register(QPushButton("Save Image"))
        self.btn_export_ctf = self._register(QPushButton("Export CTF"))
        fl.addWidget(self.btn_open, 0, 0, 1, 2)
        fl.addWidget(self.btn_save_img, 1, 0)
        fl.addWidget(self.btn_export_ctf, 1, 1)
        vbox.addWidget(fg)

        self.tabs = QTabWidget()

        # ──────── Tab 1: Map / GB ────────
        tab1 = QWidget()
        form = QFormLayout(tab1)

        self.combo_map = self._register(QComboBox())
        self.combo_map.addItems([
            "Euler", "IPF-X", "IPF-Y", "IPF-Z",
            "BC+IPF-X", "BC+IPF-Y", "BC+IPF-Z",
            "Band Contrast",
            "KAM", "Boundary", "Grain", "Misorientation",
        ])
        self.combo_map.setCurrentText("IPF-X")
        form.addRow("Map type", self.combo_map)

        # BC+IPF blend
        self.spin_bc_ipf_alpha = self._dspin(0.0, 1.0, 0.7, 0.05, 2)
        self.spin_bc_ipf_alpha.setToolTip(
            "IPF opacity over BC (0 = pure BC, 1 = pure IPF)")
        form.addRow("BC+IPF blend", self.spin_bc_ipf_alpha)

        self.combo_bc_ipf_mode = self._register(QComboBox())
        self.combo_bc_ipf_mode.addItems(["overlay", "multiply", "soft_light"])
        self.combo_bc_ipf_mode.setCurrentText("overlay")
        form.addRow("Blend mode", self.combo_bc_ipf_mode)

        self.spin_bc_brightness = self._dspin(0.5, 2.0, 1.0, 0.05, 2)
        form.addRow("BC brightness", self.spin_bc_brightness)
        self.spin_bc_contrast = self._dspin(0.5, 3.0, 1.0, 0.1, 1)
        form.addRow("BC contrast", self.spin_bc_contrast)

        # Grain detection
        self.spin_gb_angle = self._dspin(1.0, 60.0, 15.0, 0.5, 1, "°")
        form.addRow("HAGB threshold", self.spin_gb_angle)
        self.spin_min_grain = self._ispin(1, 1000, 10, 1, " px")
        form.addRow("Min grain", self.spin_min_grain)
        self.spin_kam_max = self._dspin(0.1, 20.0, 5.0, 0.5, 1, "°")
        form.addRow("KAM max", self.spin_kam_max)
        self.spin_misori_max = self._dspin(0.1, 20.0, 5.0, 0.5, 1, "°")
        form.addRow("Misori max", self.spin_misori_max)

        # Non-indexed & denoise
        form.addRow(QLabel("── Non-indexed / Denoise ──"))

        self.combo_noindex_method = self._register(QComboBox())
        self.combo_noindex_method.addItems([
            "fill (neighbor)",
            "fill (median 3×3)",
            "black",
            "white",
            "leave as-is",
        ])
        self.combo_noindex_method.setCurrentText("fill (neighbor)")
        self.combo_noindex_method.setToolTip(
            "fill (neighbor): iteratively fill with nearest valid pixel\n"
            "fill (median 3×3): fill with local median\n"
            "black / white: paint solid\n"
            "leave as-is: no treatment")
        form.addRow("Non-indexed", self.combo_noindex_method)

        self.spin_noindex_iter = self._ispin(1, 50, 5, 1)
        self.spin_noindex_iter.setToolTip(
            "Max iterations for neighbor fill (more = fill larger gaps)")
        form.addRow("Fill iterations", self.spin_noindex_iter)

        self.combo_denoise_method = self._register(QComboBox())
        self.combo_denoise_method.addItems(["None", "Median", "Gaussian"])
        form.addRow("Denoise", self.combo_denoise_method)
        self.spin_median_kernel = self._ispin(1, 21, 1, 2)
        form.addRow("Median kernel", self.spin_median_kernel)
        self.spin_gauss_sigma = self._dspin(0.0, 10.0, 0.0, 0.1, 1)
        form.addRow("Gauss sigma", self.spin_gauss_sigma)

        # HAGB
        form.addRow(QLabel("── HAGB ──"))
        self.combo_gb_color = self._register(QComboBox())
        self.combo_gb_color.addItems(["black", "white", "red", "yellow", "blue"])
        form.addRow("HAGB color", self.combo_gb_color)
        self.spin_gb_width = self._dspin(0.1, 5.0, 0.3, 0.1, 1, " pt")
        self.spin_gb_alpha = self._dspin(0.0, 1.0, 1.0, 0.1, 1)
        self.spin_gb_smooth = self._dspin(0.0, 5.0, 0.6, 0.1, 1)
        form.addRow("HAGB width", self.spin_gb_width)
        form.addRow("HAGB alpha", self.spin_gb_alpha)
        form.addRow("HAGB smooth", self.spin_gb_smooth)
        self.spin_hole_fill = self._ispin(0, 500, 10, 1, " px")
        self.spin_frag_merge = self._ispin(0, 200, 5, 1, " px")
        form.addRow("Fill holes ≤", self.spin_hole_fill)
        form.addRow("Merge frags ≤", self.spin_frag_merge)
        self.chk_plot_gbs = self._register(QCheckBox("Show HAGB"))
        self.chk_plot_gbs.setChecked(True)
        form.addRow(self.chk_plot_gbs)

        # LAGB
        form.addRow(QLabel("── LAGB (2°~15°) ──"))
        self.chk_plot_lagb = self._register(QCheckBox("Show LAGB"))
        self.chk_plot_lagb.setChecked(False)
        form.addRow(self.chk_plot_lagb)
        self.spin_lagb_min = self._dspin(0.5, 15.0, 2.0, 0.5, 1, "°")
        self.spin_lagb_max = self._dspin(1.0, 60.0, 15.0, 0.5, 1, "°")
        form.addRow("LAGB min", self.spin_lagb_min)
        form.addRow("LAGB max", self.spin_lagb_max)
        self.combo_lagb_color = self._register(QComboBox())
        self.combo_lagb_color.addItems([
            "red", "blue", "green", "magenta", "cyan",
            "yellow", "white", "black",
        ])
        self.combo_lagb_color.setCurrentText("red")
        form.addRow("LAGB color", self.combo_lagb_color)
        self.spin_lagb_width = self._dspin(0.1, 5.0, 0.2, 0.1, 1, " pt")
        self.spin_lagb_alpha = self._dspin(0.0, 1.0, 0.8, 0.1, 1)
        self.spin_lagb_smooth = self._dspin(0.0, 5.0, 0.4, 0.1, 1)
        form.addRow("LAGB width", self.spin_lagb_width)
        form.addRow("LAGB alpha", self.spin_lagb_alpha)
        form.addRow("LAGB smooth", self.spin_lagb_smooth)
        self.combo_lagb_style = self._register(QComboBox())
        self.combo_lagb_style.addItems(["solid", "dashed", "dotted", "dashdot"])
        self.combo_lagb_style.setCurrentText("dashed")
        form.addRow("LAGB style", self.combo_lagb_style)

        # Display
        form.addRow(QLabel("── Display ──"))
        self.combo_scalar_cmap = self._register(QComboBox())
        self.combo_scalar_cmap.addItems([
            "viridis", "inferno", "plasma", "magma", "turbo", "jet", "gray",
        ])
        form.addRow("Scalar cmap", self.combo_scalar_cmap)
        self.chk_show_colorbar = self._register(QCheckBox("Show colorbar"))
        self.chk_show_colorbar.setChecked(True)
        form.addRow(self.chk_show_colorbar)
        self.spin_scalebar_frac = self._dspin(0.05, 0.50, 0.18, 0.01, 2)
        form.addRow("Scale length", self.spin_scalebar_frac)
        self.combo_scalebar_loc = self._register(QComboBox())
        self.combo_scalebar_loc.addItems([
            "lower right", "upper right", "lower left", "upper left",
        ])
        self.combo_scalebar_loc.setCurrentText("lower left")
        form.addRow("Scale bar pos", self.combo_scalebar_loc)
        self.chk_scalebar = self._register(QCheckBox("Show scale bar"))
        self.chk_scalebar.setChecked(True)
        form.addRow(self.chk_scalebar)
        self.btn_refresh = self._register(QPushButton("Refresh"))
        form.addRow(self.btn_refresh)

        self.tabs.addTab(tab1, "Map / GB")

        # ──────── Tab 2: Twins ────────
        tab2 = QWidget()
        tf = QFormLayout(tab2)
        self.lbl_crystal_sys = QLabel("(load file first)")
        tf.addRow("Crystal system", self.lbl_crystal_sys)
        self.chk_show_twins = self._register(
            QCheckBox("Show twin boundaries"))
        self.chk_show_twins.setChecked(False)
        tf.addRow(self.chk_show_twins)
        self.spin_twin_tol = self._dspin(1.0, 15.0, 5.0, 0.5, 1, "°")
        tf.addRow("Twin tolerance", self.spin_twin_tol)
        self.spin_twin_width = self._dspin(0.1, 5.0, 0.5, 0.1, 1, " pt")
        self.spin_twin_alpha = self._dspin(0.0, 1.0, 1.0, 0.1, 1)
        tf.addRow("Twin line width", self.spin_twin_width)
        tf.addRow("Twin line alpha", self.spin_twin_alpha)
        self.chk_strict_axis = self._register(
            QCheckBox("Strict axis check"))
        self.chk_strict_axis.setChecked(False)
        tf.addRow(self.chk_strict_axis)

        tf.addRow(QLabel("── Twin systems ──"))
        self.twin_checks: List[QCheckBox] = []
        tf.addRow(QLabel("Cubic:"))
        for td in TWIN_SYSTEMS["cubic"]:
            cb = self._register(
                QCheckBox(f'{td["name"]} ({td["angle_deg"]:.1f}°)'))
            cb.setChecked(td["name"] == "Σ3 {111}<112>")
            cb.setProperty("twin_def", td)
            cb.setProperty("sym", "cubic")
            self.twin_checks.append(cb)
            tf.addRow(cb)
        tf.addRow(QLabel("Hexagonal:"))
        for td in TWIN_SYSTEMS["hexagonal"]:
            cb = self._register(
                QCheckBox(f'{td["name"]} ({td["angle_deg"]:.1f}°)'))
            cb.setChecked(td["name"] == "{10-12} tension")
            cb.setProperty("twin_def", td)
            cb.setProperty("sym", "hexagonal")
            self.twin_checks.append(cb)
            tf.addRow(cb)
        self.lbl_twin_stats = QLabel("-")
        self.lbl_twin_stats.setWordWrap(True)
        tf.addRow("Twin stats", self.lbl_twin_stats)
        self.tabs.addTab(tab2, "Twins")

        # ──────── Tab 3: Pole Figure ────────
        tab3 = QWidget()
        pf = QFormLayout(tab3)
        self.combo_pf_direction = self._register(QComboBox())
        self.combo_pf_direction.addItems(["X (RD)", "Y (TD)", "Z (ND)"])
        self.combo_pf_direction.setCurrentText("Z (ND)")
        pf.addRow("Sample direction", self.combo_pf_direction)
        self.combo_pf_projection = self._register(QComboBox())
        self.combo_pf_projection.addItems(["stereographic", "lambert"])
        pf.addRow("Projection", self.combo_pf_projection)
        self.spin_pf_marker_size = self._dspin(0.5, 50.0, 3.0, 0.5, 1)
        pf.addRow("Marker size", self.spin_pf_marker_size)
        self.combo_pf_marker = self._register(QComboBox())
        self.combo_pf_marker.addItems([".", "+", "o", "x", "^"])
        pf.addRow("Marker", self.combo_pf_marker)
        self.spin_pf_alpha = self._dspin(0.01, 1.0, 0.3, 0.05, 2)
        pf.addRow("Alpha", self.spin_pf_alpha)
        self.chk_pf_color_ipf = self._register(QCheckBox("Color by IPF"))
        self.chk_pf_color_ipf.setChecked(True)
        pf.addRow(self.chk_pf_color_ipf)
        self.btn_plot_pf = self._register(QPushButton("Plot Pole Figure"))
        pf.addRow(self.btn_plot_pf)
        self.btn_plot_ipf = self._register(QPushButton("Plot IPF"))
        pf.addRow(self.btn_plot_ipf)
        self.tabs.addTab(tab3, "Pole Figure")

        vbox.addWidget(self.tabs)

        # ── Output group ──
        og = QGroupBox("Output")
        of_ = QFormLayout(og)
        self.edit_save_name = self._register(QLineEdit("ebsd_preview"))
        of_.addRow("File name", self.edit_save_name)
        self.combo_save_fmt = self._register(QComboBox())
        self.combo_save_fmt.addItems(["PNG", "TIFF", "JPEG"])
        of_.addRow("Format", self.combo_save_fmt)
        self.spin_save_dpi = self._ispin(72, 1200, 300, 50, " dpi")
        of_.addRow("DPI", self.spin_save_dpi)
        vbox.addWidget(og)
        vbox.addStretch(1)
        return panel

    # ----------------------------------------------------------------
    # Info panel
    # ----------------------------------------------------------------

    def _build_info_panel(self) -> QWidget:
        panel = QWidget()
        outer = QVBoxLayout(panel)

        fg = QGroupBox("File / Map info")
        ff = QFormLayout(fg)
        self.lbl_path = QLabel("-")
        self.lbl_path.setWordWrap(True)
        self.lbl_size = QLabel("-")
        self.lbl_step = QLabel("-")
        self.lbl_num_phases = QLabel("-")
        ff.addRow("File", self.lbl_path)
        ff.addRow("Pixels", self.lbl_size)
        ff.addRow("Step", self.lbl_step)
        ff.addRow("Phases", self.lbl_num_phases)

        ff.addRow(QLabel("── Phase details ──"))
        self.lbl_phase_name = QLabel("-")
        self.lbl_crystal_type = QLabel("-")
        self.lbl_structure = QLabel("-")
        self.lbl_laue_group = QLabel("-")
        self.lbl_space_group = QLabel("-")
        self.lbl_lattice = QLabel("-")
        self.lbl_lattice.setWordWrap(True)
        self.lbl_c_over_a = QLabel("-")
        ff.addRow("Phase name", self.lbl_phase_name)
        ff.addRow("Crystal type", self.lbl_crystal_type)
        ff.addRow("Structure", self.lbl_structure)
        ff.addRow("Laue group", self.lbl_laue_group)
        ff.addRow("Space group", self.lbl_space_group)
        ff.addRow("Lattice", self.lbl_lattice)
        ff.addRow("c/a ratio", self.lbl_c_over_a)

        gg = QGroupBox("Selected grain")
        gf = QFormLayout(gg)
        self.lbl_click_xy = QLabel("-")
        self.lbl_grain_id = QLabel("-")
        self.lbl_grain_pixels = QLabel("-")
        self.lbl_grain_area = QLabel("-")
        self.lbl_grain_eqd = QLabel("-")
        self.lbl_grain_phase = QLabel("-")
        self.lbl_grain_avg_mis = QLabel("-")
        self.lbl_grain_ref_ori = QLabel("-")
        self.lbl_grain_ref_ori.setWordWrap(True)
        self.lbl_grain_twin = QLabel("-")
        self.lbl_grain_twin.setWordWrap(True)
        gf.addRow("Click (x, y)", self.lbl_click_xy)
        gf.addRow("Grain ID", self.lbl_grain_id)
        gf.addRow("Pixels", self.lbl_grain_pixels)
        gf.addRow("Area", self.lbl_grain_area)
        gf.addRow("Eq. diameter", self.lbl_grain_eqd)
        gf.addRow("Phase ID", self.lbl_grain_phase)
        gf.addRow("Avg misori", self.lbl_grain_avg_mis)
        gf.addRow("Ref ori", self.lbl_grain_ref_ori)
        gf.addRow("Twin info", self.lbl_grain_twin)

        outer.addWidget(fg)
        outer.addWidget(gg)
        outer.addStretch(1)
        return panel

    # ================================================================
    # Signal wiring
    # ================================================================

    def _connect_signals(self) -> None:
        self.btn_open.clicked.connect(self.open_file)
        self.btn_save_img.clicked.connect(self.save_image)
        self.btn_export_ctf.clicked.connect(self.export_ctf)
        self.btn_refresh.clicked.connect(self.refresh_plot)
        self.btn_plot_pf.clicked.connect(self.plot_pole_figure)
        self.btn_plot_ipf.clicked.connect(self.plot_inverse_pole_figure)
        self.viewer.canvas.mpl_connect(
            "button_press_event", self.on_canvas_click)
        self.spin_gb_angle.valueChanged.connect(self._sync_lagb_max)

        # Combo boxes → debounced refresh
        for w in [
            self.combo_map, self.combo_denoise_method,
            self.combo_noindex_method,
            self.combo_gb_color, self.combo_scalar_cmap,
            self.combo_scalebar_loc, self.combo_lagb_color,
            self.combo_lagb_style, self.combo_bc_ipf_mode,
        ]:
            w.currentIndexChanged.connect(self._schedule_refresh)

        # Check boxes → debounced refresh
        for w in [
            self.chk_show_colorbar, self.chk_scalebar,
            self.chk_plot_gbs, self.chk_plot_lagb,
            self.chk_show_twins,
        ]:
            w.toggled.connect(self._schedule_refresh)

        # Spin boxes → debounced refresh
        for w in [
            self.spin_gb_angle, self.spin_min_grain,
            self.spin_kam_max, self.spin_misori_max,
            self.spin_median_kernel, self.spin_gauss_sigma,
            self.spin_scalebar_frac,
            self.spin_gb_width, self.spin_gb_alpha,
            self.spin_gb_smooth, self.spin_hole_fill,
            self.spin_frag_merge,
            self.spin_lagb_min, self.spin_lagb_max,
            self.spin_lagb_width, self.spin_lagb_alpha,
            self.spin_lagb_smooth,
            self.spin_twin_tol, self.spin_twin_width,
            self.spin_twin_alpha,
            self.spin_bc_ipf_alpha, self.spin_bc_brightness,
            self.spin_bc_contrast, self.spin_noindex_iter,
        ]:
            w.valueChanged.connect(self._schedule_refresh)

        for cb in self.twin_checks:
            cb.toggled.connect(self._schedule_refresh)

    # ----------------------------------------------------------------
    # Debounce helpers
    # ----------------------------------------------------------------

    def _schedule_refresh(self) -> None:
        """(Re)start the debounce timer — actual refresh fires after DEBOUNCE_MS."""
        self._refresh_timer.start()

    def _do_refresh(self) -> None:
        """Called once the debounce timer expires."""
        self.refresh_plot()

    # ----------------------------------------------------------------
    # Misc UI helpers
    # ----------------------------------------------------------------

    def _sync_lagb_max(self, val: float) -> None:
        self.spin_lagb_max.blockSignals(True)
        self.spin_lagb_max.setValue(val)
        self.spin_lagb_max.blockSignals(False)

    def _clear_grain_info(self) -> None:
        for lbl in [
            self.lbl_click_xy, self.lbl_grain_id,
            self.lbl_grain_pixels, self.lbl_grain_area,
            self.lbl_grain_eqd, self.lbl_grain_phase,
            self.lbl_grain_avg_mis, self.lbl_grain_ref_ori,
            self.lbl_grain_twin,
        ]:
            lbl.setText("-")

    def _update_info(self, m: ebsd.Map) -> None:
        if getattr(m.data, "euler_angle", None) is not None:
            _, ny, nx = m.data.euler_angle.shape
            self.lbl_size.setText(f"{nx} × {ny}")
        else:
            self.lbl_size.setText("-")
        self.lbl_step.setText(f"{m.step_size:.3f} µm")
        self.lbl_num_phases.setText(str(m.num_phases))
        self.lbl_path.setText(
            str(self.current_path) if self.current_path else "-")
        try:
            phase = m.primary_phase
            sym = m.crystal_sym
            self.lbl_phase_name.setText(phase.name or "-")
            self.lbl_crystal_type.setText(sym.capitalize())
            sg = phase.spaceGroup
            structure = self._SPACE_GROUP_TO_STRUCTURE.get(sg, None)
            if structure is None:
                lg = phase.laue_group
                if lg == 9:
                    structure = "HCP"
                elif lg == 11:
                    structure = (
                        "FCC" if sg in (225, 227)
                        else ("BCC" if sg == 229 else "Cubic")
                    )
                else:
                    structure = "-"
            self.lbl_structure.setText(structure)
            lg = phase.laue_group
            self.lbl_laue_group.setText(
                f"{lg} ({self._LAUE_TO_SYSTEM.get(lg, '?')})")
            self.lbl_space_group.setText(str(sg))
            lp = phase.lattice_params
            if lp and len(lp) >= 6:
                self.lbl_lattice.setText(
                    f"a={lp[0]:.4f}  b={lp[1]:.4f}  c={lp[2]:.4f} Å\n"
                    f"α={np.degrees(lp[3]):.1f}°  "
                    f"β={np.degrees(lp[4]):.1f}°  "
                    f"γ={np.degrees(lp[5]):.1f}°")
            else:
                self.lbl_lattice.setText("-")
            ca = m.c_over_a
            self.lbl_c_over_a.setText(f"{ca:.4f}" if ca else "N/A")
            self.lbl_crystal_sys.setText(f"{structure} ({sym})")
        except Exception:
            logger.warning("Failed to read phase info", exc_info=True)
            for lbl in [
                self.lbl_phase_name, self.lbl_crystal_type,
                self.lbl_structure, self.lbl_laue_group,
                self.lbl_space_group, self.lbl_lattice,
                self.lbl_c_over_a,
            ]:
                lbl.setText("-")

    def _update_control_states(self, enabled: bool) -> None:
        for w in self._data_widgets:
            w.setEnabled(enabled)

    # ================================================================
    # Map loading (cached)
    # ================================================================

    def _load_map(self, force_reload: bool = False) -> ebsd.Map:
        if self.current_path is None:
            raise RuntimeError("No file loaded.")
        if force_reload or self._cached_raw_map is None:
            self._cached_raw_map = ebsd.Map(self.current_path)
        return self._cached_raw_map

    # ================================================================
    # Cache management
    # ================================================================

    def _invalidate_all_caches(self) -> None:
        """Reset all caches. Call when a new file is loaded."""
        self._cached_raw_map = None
        self._cached_lagb_map = None
        self._cache_mask = (None, None)
        self._cache_bc = (None, None)
        self._cache_ipf = (None, None)
        self._cache_clean_gm = (None, None)
        self._cache_neighbor_pairs = (None, None)
        self._cache_twin = (None, None)
        self._cache_lagb_grains = (None, None)

    # ================================================================
    # Twin helpers (UI read -- main thread only)
    # ================================================================

    def _get_non_indexed_mask(self, m: ebsd.Map) -> np.ndarray:
        """Detect non-indexed pixels.  Returns bool (H, W), True = bad."""
        h = w = None
        mask = None

        try:
            phase = np.asarray(m.data.phase)
            h, w = phase.shape
            mask = phase == 0
        except Exception:
            logger.debug("No phase data for non-indexed detection")

        try:
            bc = np.asarray(m.data.band_contrast)
            if h is None:
                h, w = bc.shape
            bc_bad = bc <= 0
            mask = bc_bad if mask is None else (mask | bc_bad)
        except Exception:
            logger.debug("No band-contrast data for non-indexed detection")

        try:
            ea = np.asarray(m.data.euler_angle)  # (3, H, W)
            ea_sum = np.abs(ea[0]) + np.abs(ea[1]) + np.abs(ea[2])
            ea_bad = ea_sum < EULER_ZERO_THRESHOLD
            if h is None:
                h, w = ea_bad.shape
            mask = ea_bad if mask is None else (mask | ea_bad)
        except Exception:
            logger.debug("No Euler-angle data for non-indexed detection")

        if mask is None:
            if h is None:
                return np.zeros((1, 1), dtype=bool)
            return np.zeros((h, w), dtype=bool)
        return mask

    # ================================================================
    # Unified non-indexed pixel filling
    # ================================================================

    def _fill_non_indexed(
        self,
        img: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Fill non-indexed pixels in a 2-D or (H, W, C) array."""
        method = self.combo_noindex_method.currentText()
        if method == "leave as-is" or not np.any(mask):
            return img

        img = img.copy()
        is_rgb = img.ndim == 3

        if method == "black":
            img[mask] = 0.0
            return img
        if method == "white":
            if is_rgb:
                img[mask] = 1.0
            else:
                mx = float(img[~mask].max()) if np.any(~mask) else 1.0
                img[mask] = mx
            return img

        max_iter = self.spin_noindex_iter.value()

        if method == "fill (median 3×3)":
            if is_rgb:
                return _vectorised_median_fill_rgb(img, mask).astype(img.dtype)
            return _vectorised_median_fill_2d(img, mask).astype(img.dtype)

        # "fill (neighbor)" — vectorised iterative 8-neighbour fill
        if is_rgb:
            return _vectorised_neighbor_fill_rgb(
                img, mask, max_iter).astype(img.dtype)
        return _vectorised_neighbor_fill_2d(
            img, mask, max_iter).astype(img.dtype)

    # ================================================================
    # Denoise (applied AFTER non-indexed fill)
    # ================================================================

    def _denoise_2d(self, a: np.ndarray) -> np.ndarray:
        method = self.combo_denoise_method.currentText()
        if method == "Median":
            k = self.spin_median_kernel.value()
            if k < 2:
                return a
            k = k if k % 2 == 1 else k + 1
            return median_filter(a.copy(), size=k)
        if method == "Gaussian":
            s = self.spin_gauss_sigma.value()
            if s < 0.01:
                return a
            return gaussian_filter(
                a.copy().astype(np.float64), sigma=s,
            ).astype(a.dtype)
        return a

    def _denoise_rgb(self, a: np.ndarray) -> np.ndarray:
        method = self.combo_denoise_method.currentText()
        if method == "Median":
            k = self.spin_median_kernel.value()
            if k < 2:
                return a
            k = k if k % 2 == 1 else k + 1
            a = a.copy()
            return np.dstack([
                median_filter(a[:, :, i], size=k)
                for i in range(a.shape[2])
            ])
        if method == "Gaussian":
            s = self.spin_gauss_sigma.value()
            if s < 0.01:
                return a
            a = a.copy().astype(np.float64)
            for i in range(a.shape[2]):
                a[:, :, i] = gaussian_filter(a[:, :, i], sigma=s)
            return a.astype(np.float32)
        return a

    def _normalize_rgb(self, a: np.ndarray) -> np.ndarray:
        if a.dtype == np.uint8:
            a = a.astype(np.float32) / 255.0
        elif a.dtype != np.float32:
            a = a.astype(np.float32)
        mx = float(a.max())
        if mx > 1.5 and mx > 0:
            a = a / mx
        if a.ndim == 3 and a.shape[2] == 4:
            a = a[:, :, :3]
        return np.clip(a, 0.0, 1.0)

    # ================================================================
    # BC + IPF blending
    # ================================================================

    def _get_bc_array(self, m: ebsd.Map) -> np.ndarray:
        bc = np.asarray(m.data.band_contrast).astype(np.float32)
        lo, hi = float(bc.min()), float(bc.max())
        if hi > lo:
            bc = (bc - lo) / (hi - lo)
        else:
            bc = np.zeros_like(bc)
        br = self.spin_bc_brightness.value()
        if abs(br - 1.0) > BRIGHTNESS_CHANGE_EPSILON:
            bc = bc * br
        ct = self.spin_bc_contrast.value()
        if abs(ct - 1.0) > CONTRAST_CHANGE_EPSILON:
            bc = (bc - 0.5) * ct + 0.5
        return np.clip(bc, 0.0, 1.0)

    def _compute_ipf_rgb(
        self,
        m: ebsd.Map,
        direction: np.ndarray,
    ) -> np.ndarray:
        q = np.array(m.data["orientation"])
        fq = q.ravel()
        rgb = Quat.calc_ipf_colours(fq, direction, m.crystal_sym)
        return rgb.T.reshape(q.shape + (3,)).astype(np.float32)

    def _blend_bc_ipf(
        self,
        bc: np.ndarray,
        ipf_rgb: np.ndarray,
    ) -> np.ndarray:
        alpha = self.spin_bc_ipf_alpha.value()
        mode = self.combo_bc_ipf_mode.currentText()
        bc3 = np.dstack([bc, bc, bc])
        if mode == "multiply":
            mult = bc3 * ipf_rgb
            result = bc3 * (1.0 - alpha) + mult * alpha
            result *= MULTIPLY_BLEND_BOOST
        elif mode == "soft_light":
            m_ = ipf_rgb <= 0.5
            soft = np.where(
                m_,
                bc3 - (1 - 2 * ipf_rgb) * bc3 * (1 - bc3),
                bc3 + (2 * ipf_rgb - 1) * (
                    np.sqrt(np.maximum(bc3, 0)) - bc3),
            )
            result = bc3 * (1.0 - alpha) + soft * alpha
        else:  # overlay
            result = bc3 * (1.0 - alpha) + ipf_rgb * alpha
        return np.clip(result, 0.0, 1.0).astype(np.float32)

    # ================================================================
    # Grain map cleaning
    # ================================================================

    def _build_clean_grain_map(self, m: ebsd.Map) -> np.ndarray:
        gm = np.asarray(m.data["grains"]).copy()
        hf = self.spin_hole_fill.value()
        fm = self.spin_frag_merge.value()
        if hf > 0:
            inv = gm <= 0
            if np.any(inv):
                lh, nh = ndlabel(inv)
                for hid in range(1, nh + 1):
                    hm = lh == hid
                    if hm.sum() > hf:
                        continue
                    d = _binary_dilation(hm, iterations=1)
                    b = d & ~hm
                    ni = gm[b]
                    ni = ni[ni > 0]
                    if ni.size > 0:
                        gm[hm] = int(np.bincount(ni).argmax())
        if fm > 0:
            for gid in np.unique(gm):
                if gid <= 0:
                    continue
                grain_mask = gm == gid
                if grain_mask.sum() >= fm:
                    continue
                d = _binary_dilation(grain_mask, iterations=1)
                b = d & ~grain_mask
                ni = gm[b]
                ni = ni[(ni > 0) & (ni != gid)]
                if ni.size > 0:
                    gm[grain_mask] = int(np.bincount(ni).argmax())
                else:
                    gm[grain_mask] = 0
        return gm

    # ================================================================
    # Contours
    # ================================================================

    def _smooth_contour(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sigma: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(x)
        if sigma < 0.05 or n < 5:
            return x, y
        asig = max(
            sigma * min(1.0, n / SMOOTH_CONTOUR_LENGTH_SCALE),
            sigma * SMOOTH_CONTOUR_MIN_SIGMA_FRAC,
        )
        closed = abs(x[0] - x[-1]) < 1.0 and abs(y[0] - y[-1]) < 1.0
        mode = "wrap" if closed else "nearest"
        return (
            gaussian_filter1d(x, sigma=asig, mode=mode),
            gaussian_filter1d(y, sigma=asig, mode=mode),
        )

    def _contours_from_grain_map(
        self,
        grain_map: np.ndarray,
        sigma: float,
    ) -> List[np.ndarray]:
        uids = np.unique(grain_map)
        uids = uids[uids > 0]
        all_c: List[np.ndarray] = []
        for gid in uids:
            mask_f = (grain_map == gid).astype(np.float64)
            for c in find_contours(mask_f, level=0.5):
                if c.shape[0] < MIN_CONTOUR_POINTS:
                    continue
                y, x = c[:, 0], c[:, 1]
                x, y = self._smooth_contour(x, y, sigma)
                all_c.append(np.column_stack([x, y]))
        return all_c

    # ================================================================
    # Boundary overlays  (logic moved to _compute_render_result)
    # ================================================================

    # ================================================================
    # Twin helpers (UI read -- main thread only)
    # ================================================================

    def _get_active_twin_defs(self, sym_group: str) -> List[dict]:
        tol = self.spin_twin_tol.value()
        defs: List[dict] = []
        for cb in self.twin_checks:
            if not cb.isChecked():
                continue
            if cb.property("sym") != sym_group:
                continue
            td = cb.property("twin_def").copy()
            td["tolerance_deg"] = tol
            defs.append(td)
        return defs

    # ================================================================
    # Async render pipeline
    # ================================================================

    def _collect_render_params(self, m: ebsd.Map) -> dict:
        """Snapshot all UI parameters into a plain dict (main thread only)."""
        mt = self.combo_map.currentText()
        if "X" in mt:
            direction = (1, 0, 0)
        elif "Y" in mt:
            direction = (0, 1, 0)
        else:
            direction = (0, 0, 1)

        sym = m.crystal_sym
        active_twin_defs = self._get_active_twin_defs(sym)
        twin_defs_hash = tuple(
            (td["name"], td["angle_deg"], td["tolerance_deg"])
            for td in active_twin_defs
        )

        return {
            # Identity
            "path": str(self.current_path),
            "step_size": float(m.step_size),
            "crystal_sym": sym,
            # Map
            "map_type": mt,
            "direction": direction,
            "gb_angle": self.spin_gb_angle.value(),
            "min_grain": self.spin_min_grain.value(),
            "kam_max": self.spin_kam_max.value(),
            "misori_max": self.spin_misori_max.value(),
            # Non-indexed / denoise
            "noindex_method": self.combo_noindex_method.currentText(),
            "noindex_iter": self.spin_noindex_iter.value(),
            "denoise_method": self.combo_denoise_method.currentText(),
            "median_kernel": self.spin_median_kernel.value(),
            "gauss_sigma": self.spin_gauss_sigma.value(),
            # BC / IPF
            "bc_brightness": self.spin_bc_brightness.value(),
            "bc_contrast": self.spin_bc_contrast.value(),
            "bc_ipf_alpha": self.spin_bc_ipf_alpha.value(),
            "bc_ipf_mode": self.combo_bc_ipf_mode.currentText(),
            # HAGB
            "show_hagb": self.chk_plot_gbs.isChecked(),
            "gb_color": self.combo_gb_color.currentText(),
            "gb_width": self.spin_gb_width.value(),
            "gb_alpha": self.spin_gb_alpha.value(),
            "gb_smooth": self.spin_gb_smooth.value(),
            "hole_fill": self.spin_hole_fill.value(),
            "frag_merge": self.spin_frag_merge.value(),
            # LAGB
            "show_lagb": self.chk_plot_lagb.isChecked(),
            "lagb_min": self.spin_lagb_min.value(),
            "lagb_max": self.spin_lagb_max.value(),
            "lagb_color": self.combo_lagb_color.currentText(),
            "lagb_width": self.spin_lagb_width.value(),
            "lagb_alpha": self.spin_lagb_alpha.value(),
            "lagb_smooth": self.spin_lagb_smooth.value(),
            "lagb_style": self.combo_lagb_style.currentText(),
            # Twins
            "show_twins": self.chk_show_twins.isChecked(),
            "twin_defs": active_twin_defs,
            "twin_defs_hash": twin_defs_hash,
            "twin_tol": self.spin_twin_tol.value(),
            "twin_width": self.spin_twin_width.value(),
            "twin_alpha": self.spin_twin_alpha.value(),
            "strict_axis": self.chk_strict_axis.isChecked(),
            # Display
            "scalar_cmap": self.combo_scalar_cmap.currentText(),
            "show_colorbar": self.chk_show_colorbar.isChecked(),
            "show_scalebar": self.chk_scalebar.isChecked(),
            "scalebar_frac": self.spin_scalebar_frac.value(),
            "scalebar_loc": self.combo_scalebar_loc.currentText(),
            # Cache snapshots passed to worker
            "cache_mask": self._cache_mask,
            "cache_bc": self._cache_bc,
            "cache_ipf": self._cache_ipf,
            "cache_clean_gm": self._cache_clean_gm,
            "cache_neighbor_pairs": self._cache_neighbor_pairs,
            "cache_twin": self._cache_twin,
            "cache_lagb_grains": self._cache_lagb_grains,
            # LAGB independent map (may be None on first use)
            "cached_lagb_map": self._cached_lagb_map,
        }

    def _start_render(self, params: dict) -> None:
        """Dispatch rendering to a background worker thread."""
        self._render_busy = True
        worker = _RenderWorker(
            EbsdMainWindow._compute_render_result, (params, self._cached_raw_map)
        )
        thread = QThread(self)
        self._worker_thread = thread
        self._current_worker = worker
        worker.moveToThread(thread)
        worker.finished.connect(self._on_worker_finished)
        worker.error.connect(self._on_worker_error)
        thread.started.connect(worker.run)
        thread.start()

    def _on_worker_finished(self, result: dict) -> None:
        """Called in the main thread when the worker completes."""
        thread = self._worker_thread
        self._worker_thread = None
        self._current_worker = None
        if thread is not None:
            thread.quit()
            thread.wait(2000)

        # Update caches with any newly computed values
        _cache_map = {
            "new_cache_mask": "_cache_mask",
            "new_cache_bc": "_cache_bc",
            "new_cache_ipf": "_cache_ipf",
            "new_cache_clean_gm": "_cache_clean_gm",
            "new_cache_neighbor_pairs": "_cache_neighbor_pairs",
            "new_cache_twin": "_cache_twin",
            "new_cache_lagb_grains": "_cache_lagb_grains",
        }
        for new_key, attr in _cache_map.items():
            val = result.get(new_key)
            if val is not None:
                setattr(self, attr, val)
        if result.get("new_cached_lagb_map") is not None:
            self._cached_lagb_map = result["new_cached_lagb_map"]

        # Draw result in the main thread
        try:
            self._draw_render_result(result)
            self._clear_grain_info()
        except Exception as exc:
            logger.error("Draw failed", exc_info=True)
            self.current_fig_ready = False
            QMessageBox.critical(self, "Render error", str(exc))
            self.statusBar().showMessage("Render failed")

        QApplication.restoreOverrideCursor()
        self._render_busy = False

        # Process any pending render request queued while busy
        if self._pending_params is not None:
            p = self._pending_params
            self._pending_params = None
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage("Rendering...")
            self._start_render(p)

    def _on_worker_error(self, msg: str) -> None:
        """Called in the main thread when the worker raises an exception."""
        thread = self._worker_thread
        self._worker_thread = None
        self._current_worker = None
        if thread is not None:
            thread.quit()
            thread.wait(2000)
        self.current_fig_ready = False
        QApplication.restoreOverrideCursor()
        self._render_busy = False
        self.statusBar().showMessage(f"Render failed: {msg}")
        QMessageBox.critical(self, "Render error", msg)

    @staticmethod
    def _compute_render_result(params: dict, m: ebsd.Map) -> dict:
        """Heavy computation executed in the worker thread.

        No Qt widget access is permitted here.
        """
        t0 = time.perf_counter()

        mt = params["map_type"]
        is_bc_ipf = mt.startswith("BC+IPF")
        path = params["path"]

        # ------ Non-indexed mask ------
        cache_mask_key, cache_mask_val = params["cache_mask"]
        if cache_mask_key == path and cache_mask_val is not None:
            mask = cache_mask_val
            new_cache_mask = None
        else:
            mask = _get_non_indexed_mask_pure(m)
            new_cache_mask = (path, mask)

        noindex_method = params["noindex_method"]
        noindex_iter = int(params["noindex_iter"])
        denoise_method = params["denoise_method"]
        median_kernel = int(params["median_kernel"])
        gauss_sigma = float(params["gauss_sigma"])

        image: Optional[np.ndarray] = None
        is_scalar = False
        scalar_label: Optional[str] = None
        vmin = vmax = None
        new_cache_bc = None
        new_cache_ipf = None

        # ------ BC + IPF ------
        if is_bc_ipf:
            bc_key = (path, params["bc_brightness"], params["bc_contrast"])
            ck, cv = params["cache_bc"]
            if ck == bc_key and cv is not None:
                bc = cv
            else:
                bc = _get_bc_array_pure(m, params["bc_brightness"], params["bc_contrast"])
                new_cache_bc = (bc_key, bc)

            direction = np.array(params["direction"], dtype=float)
            ipf_key = (path, params["direction"])
            ik, iv = params["cache_ipf"]
            if ik == ipf_key and iv is not None:
                ipf_rgb = iv
            else:
                ipf_rgb = _compute_ipf_rgb_pure(m, direction)
                ipf_rgb = _normalize_rgb_pure(ipf_rgb)
                new_cache_ipf = (ipf_key, ipf_rgb)

            bc_f = _fill_non_indexed_pure(bc, mask, noindex_method, noindex_iter)
            bc_f = _denoise_2d_pure(bc_f, denoise_method, median_kernel, gauss_sigma)
            ipf_f = _fill_non_indexed_pure(ipf_rgb, mask, noindex_method, noindex_iter)
            ipf_f = _denoise_rgb_pure(ipf_f, denoise_method, median_kernel, gauss_sigma)
            image = _blend_bc_ipf_pure(
                bc_f, ipf_f, params["bc_ipf_alpha"], params["bc_ipf_mode"]
            )

        # ------ Band Contrast ------
        elif mt == "Band Contrast":
            bc_key = (path, params["bc_brightness"], params["bc_contrast"])
            ck, cv = params["cache_bc"]
            if ck == bc_key and cv is not None:
                bc = cv
            else:
                bc = _get_bc_array_pure(m, params["bc_brightness"], params["bc_contrast"])
                new_cache_bc = (bc_key, bc)
            bc_f = _fill_non_indexed_pure(bc, mask, noindex_method, noindex_iter)
            bc_f = _denoise_2d_pure(bc_f, denoise_method, median_kernel, gauss_sigma)
            image = bc_f
            is_scalar = True
            scalar_label = "Band Contrast"
            vmin, vmax = 0.0, 1.0

        # ------ Euler ------
        elif mt == "Euler":
            fig_tmp = Figure()
            ax_tmp = fig_tmp.add_subplot(111)
            try:
                m.plot_map("euler_angle", "all_euler", fig=fig_tmp, ax=ax_tmp,
                           plot_scale_bar=False, plot_colour_bar=False)
                raw = _extract_image_array_static(ax_tmp)
            except Exception:
                logger.debug("Euler plot_map failed", exc_info=True)
                raw = None
            if raw is not None:
                a = _normalize_rgb_pure(raw)
                a = _fill_non_indexed_pure(a, mask, noindex_method, noindex_iter)
                a = _denoise_rgb_pure(a, denoise_method, median_kernel, gauss_sigma)
                image = a

        # ------ IPF maps ------
        elif mt in ("IPF-X", "IPF-Y", "IPF-Z"):
            ax_key = mt[-1]
            dir_map: Dict[str, Tuple[int, int, int]] = {
                "X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1),
            }
            direction = np.array(dir_map[ax_key], dtype=float)
            ipf_key = (path, dir_map[ax_key])
            ik, iv = params["cache_ipf"]
            if ik == ipf_key and iv is not None:
                ipf_rgb = iv
                new_cache_ipf = None
            else:
                ipf_rgb = None
                try:
                    fig_tmp = Figure()
                    ax_tmp = fig_tmp.add_subplot(111)
                    m.plot_map(
                        "orientation", f"IPF_{ax_key.lower()}",
                        fig=fig_tmp, ax=ax_tmp,
                        plot_scale_bar=False, plot_colour_bar=False,
                    )
                    raw = _extract_image_array_static(ax_tmp)
                    if raw is not None:
                        ipf_rgb = _normalize_rgb_pure(raw)
                except Exception:
                    logger.debug("IPF_%s plot_map failed, using manual", ax_key)
                if ipf_rgb is None:
                    ipf_rgb = _compute_ipf_rgb_pure(m, direction)
                    ipf_rgb = _normalize_rgb_pure(ipf_rgb)
                new_cache_ipf = (ipf_key, ipf_rgb)

            a = _fill_non_indexed_pure(ipf_rgb, mask, noindex_method, noindex_iter)
            a = _denoise_rgb_pure(a, denoise_method, median_kernel, gauss_sigma)
            image = a

        # ------ KAM ------
        elif mt == "KAM":
            km = float(params["kam_max"])
            try:
                m.calc_kam()
                a2 = np.asarray(m.data["KAM"]).astype(np.float32)
            except Exception:
                logger.debug("KAM extraction fallback", exc_info=True)
                try:
                    fig_tmp = Figure()
                    ax_tmp = fig_tmp.add_subplot(111)
                    m.calc_kam()
                    m.plot_map("KAM", vmin=0, vmax=km, fig=fig_tmp, ax=ax_tmp,
                               plot_scale_bar=False, plot_colour_bar=False)
                    raw = _extract_image_array_static(ax_tmp)
                    a2 = (np.mean(raw[..., :3].astype(np.float32), axis=2)
                          if raw is not None
                          else np.zeros((1, 1), dtype=np.float32))
                except Exception:
                    a2 = np.zeros((1, 1), dtype=np.float32)
            a2 = _fill_non_indexed_pure(a2, mask, noindex_method, noindex_iter)
            a2 = _denoise_2d_pure(a2, denoise_method, median_kernel, gauss_sigma)
            image = a2
            is_scalar = True
            scalar_label = "KAM (\u00b0)"
            vmin, vmax = 0.0, km

        # ------ Boundary ------
        elif mt == "Boundary":
            gb = float(params["gb_angle"])
            m.data.generate("grain_boundaries", misori_tol=gb)
            fig_tmp = Figure()
            ax_tmp = fig_tmp.add_subplot(111)
            try:
                m.plot_boundary_map(fig=fig_tmp, ax=ax_tmp,
                                    plot_scale_bar=False, plot_colour_bar=False)
                raw = _extract_image_array_static(ax_tmp)
                if raw is not None:
                    image = _normalize_rgb_pure(raw)
            except Exception:
                logger.debug("Boundary plot_map failed", exc_info=True)

        # ------ Grain ------
        elif mt == "Grain":
            gb = float(params["gb_angle"])
            mg_ = int(params["min_grain"])
            m.data.generate("grain_boundaries", misori_tol=gb)
            m.data.generate("grains", min_grain_size=mg_)
            fig_tmp = Figure()
            ax_tmp = fig_tmp.add_subplot(111)
            try:
                m.plot_grain_map(fig=fig_tmp, ax=ax_tmp,
                                 plot_scale_bar=False, plot_colour_bar=False)
                raw = _extract_image_array_static(ax_tmp)
                if raw is not None:
                    image = _normalize_rgb_pure(raw)
            except Exception:
                logger.debug("Grain plot_map failed", exc_info=True)

        # ------ Misorientation ------
        elif mt == "Misorientation":
            gb = float(params["gb_angle"])
            mg_ = int(params["min_grain"])
            mm = float(params["misori_max"])
            m.data.generate("grain_boundaries", misori_tol=gb)
            m.data.generate("grains", min_grain_size=mg_)
            try:
                m.calc_grain_mis_ori()
                a2 = np.asarray(m.data["mis_ori"]).astype(np.float32)
            except Exception:
                logger.debug("Misorientation extraction fallback", exc_info=True)
                try:
                    fig_tmp = Figure()
                    ax_tmp = fig_tmp.add_subplot(111)
                    m.calc_grain_mis_ori()
                    m.plot_mis_ori_map(
                        fig=fig_tmp, ax=ax_tmp, vmin=0, vmax=mm,
                        plot_gbs=False, plot_scale_bar=False,
                        plot_colour_bar=False,
                    )
                    raw = _extract_image_array_static(ax_tmp)
                    a2 = (np.mean(raw[..., :3].astype(np.float32), axis=2)
                          if raw is not None
                          else np.zeros((1, 1), dtype=np.float32))
                except Exception:
                    a2 = np.zeros((1, 1), dtype=np.float32)
            a2 = _fill_non_indexed_pure(a2, mask, noindex_method, noindex_iter)
            a2 = _denoise_2d_pure(a2, denoise_method, median_kernel, gauss_sigma)
            image = a2
            is_scalar = True
            scalar_label = "Misorientation (\u00b0)"
            vmin, vmax = 0.0, mm

        if image is None:
            image = np.zeros((4, 4, 3), dtype=np.float32)

        # ------ Overlays ------
        hagb_contours: List[np.ndarray] = []
        lagb_contours: List[np.ndarray] = []
        twin_data: dict = {}
        twin_stats: str = ""
        new_cache_clean_gm = None
        new_cache_neighbor_pairs = None
        new_cache_twin = None
        new_cache_lagb_grains = None
        new_cached_lagb_map = None

        need_overlays = mt != "Boundary"
        need_grains = need_overlays and (
            params["show_hagb"] or params["show_lagb"] or params["show_twins"]
        )

        clean_gm: Optional[np.ndarray] = None
        clean_gm_key = (
            path,
            params["gb_angle"],
            params["min_grain"],
            params["hole_fill"],
            params["frag_merge"],
        )

        if need_grains:
            m.data.generate(
                "grain_boundaries", misori_tol=params["gb_angle"]
            )
            m.data.generate(
                "grains", min_grain_size=params["min_grain"]
            )
            ck, cv = params["cache_clean_gm"]
            if ck == clean_gm_key and cv is not None:
                clean_gm = cv
            else:
                logger.debug("Rebuilding clean grain map")
                clean_gm = _build_clean_grain_map_pure(
                    m,
                    float(params["gb_angle"]),
                    int(params["min_grain"]),
                    int(params["hole_fill"]),
                    int(params["frag_merge"]),
                )
                new_cache_clean_gm = (clean_gm_key, clean_gm)

        if need_overlays and params["show_hagb"] and clean_gm is not None:
            hagb_contours = _contours_from_grain_map_pure(
                clean_gm, float(params["gb_smooth"])
            )

        if need_overlays and params["show_twins"] and clean_gm is not None:
            np_key = clean_gm_key
            nk, nv = params["cache_neighbor_pairs"]
            if nk == np_key and nv is not None:
                neighbor_pairs = nv
            else:
                neighbor_pairs = _find_neighbor_pairs_vectorised(clean_gm)
                new_cache_neighbor_pairs = (np_key, neighbor_pairs)

            twin_key = clean_gm_key + (
                params["twin_tol"],
                params["twin_defs_hash"],
                params["strict_axis"],
            )
            tk, tv = params["cache_twin"]
            if tk == twin_key and tv is not None:
                twin_data = tv
            else:
                logger.debug("Computing twin boundaries")
                twin_data = _detect_twin_boundaries_pure(
                    params, m, clean_gm, neighbor_pairs
                )
                new_cache_twin = (twin_key, twin_data)

            parts = [
                f"{name}: {len(data['contours'])}"
                for name, data in twin_data.items()
            ]
            twin_stats = "\n".join(parts)

        if need_overlays and params["show_lagb"] and clean_gm is not None:
            lagb_grains_key = (path, params["lagb_min"], params["min_grain"])
            lk, lv = params["cache_lagb_grains"]
            if lk == lagb_grains_key and lv is not None:
                lagb_sub_gm = lv
            else:
                logger.debug("Generating LAGB sub-grain map")
                lagb_map = params.get("cached_lagb_map")
                if lagb_map is None:
                    lagb_map = ebsd.Map(path)
                    new_cached_lagb_map = lagb_map
                lagb_map.data.generate(
                    "grain_boundaries",
                    misori_tol=params["lagb_min"],
                )
                lagb_map.data.generate(
                    "grains",
                    min_grain_size=max(int(params["min_grain"]) // 2, 2),
                )
                lagb_sub_gm = np.asarray(lagb_map.data["grains"]).copy()
                new_cache_lagb_grains = (lagb_grains_key, lagb_sub_gm)
                if new_cached_lagb_map is None:
                    new_cached_lagb_map = lagb_map

            lagb_contours = _extract_lagb_contours_pure(
                params, m, lagb_sub_gm, clean_gm
            )

        logger.debug(
            "_compute_render_result completed in %.3fs",
            time.perf_counter() - t0,
        )

        return {
            "image": image,
            "is_scalar": is_scalar,
            "scalar_label": scalar_label,
            "vmin": vmin,
            "vmax": vmax,
            "cmap": params["scalar_cmap"],
            "hagb_contours": hagb_contours,
            "hagb_color": params["gb_color"],
            "hagb_width": params["gb_width"],
            "hagb_alpha": params["gb_alpha"],
            "lagb_contours": lagb_contours,
            "lagb_color": params["lagb_color"],
            "lagb_width": params["lagb_width"],
            "lagb_alpha": params["lagb_alpha"],
            "lagb_style": params["lagb_style"],
            "twin_data": twin_data,
            "twin_stats": twin_stats,
            "twin_width": params["twin_width"],
            "twin_alpha": params["twin_alpha"],
            "show_colorbar": params["show_colorbar"],
            "show_scalebar": params["show_scalebar"],
            "scalebar_frac": params["scalebar_frac"],
            "scalebar_loc": params["scalebar_loc"],
            "step_size": params["step_size"],
            # Cache updates (None means no change)
            "new_cache_mask": new_cache_mask,
            "new_cache_bc": new_cache_bc,
            "new_cache_ipf": new_cache_ipf,
            "new_cache_clean_gm": new_cache_clean_gm,
            "new_cache_neighbor_pairs": new_cache_neighbor_pairs,
            "new_cache_twin": new_cache_twin,
            "new_cache_lagb_grains": new_cache_lagb_grains,
            "new_cached_lagb_map": new_cached_lagb_map,
        }

    def _draw_render_result(self, result: dict) -> None:
        """Draw pre-computed result onto the viewer canvas (main thread)."""
        fig = self.viewer.figure
        fig.clear()
        ax = fig.add_subplot(111)

        image = result["image"]
        sh, sw = image.shape[:2]

        if result["is_scalar"]:
            im = ax.imshow(
                image, origin="upper", interpolation="none",
                extent=[-0.5, sw - 0.5, sh - 0.5, -0.5],
                aspect="equal",
                cmap=result["cmap"],
                vmin=result["vmin"], vmax=result["vmax"],
            )
            if result["show_colorbar"] and result["scalar_label"]:
                fig.colorbar(
                    im, ax=ax, fraction=0.046, pad=0.04,
                    label=result["scalar_label"],
                )
        else:
            ax.imshow(
                image, origin="upper", interpolation="none",
                extent=[-0.5, sw - 0.5, sh - 0.5, -0.5],
                aspect="equal",
            )

        ax.set_xlim(-0.5, sw - 0.5)
        ax.set_ylim(sh - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])

        # HAGB overlay
        if result["hagb_contours"]:
            ax.add_collection(LineCollection(
                result["hagb_contours"],
                colors=result["hagb_color"],
                linewidths=result["hagb_width"],
                alpha=result["hagb_alpha"],
                antialiaseds=True,
                capstyle="round",
                joinstyle="round",
                zorder=10,
            ))

        # LAGB overlay
        if result["lagb_contours"]:
            _stm = {
                "solid": "solid",
                "dashed": (0, (5, 3)),
                "dotted": (0, (1, 2)),
                "dashdot": "dashdot",
            }
            ax.add_collection(LineCollection(
                result["lagb_contours"],
                colors=result["lagb_color"],
                linewidths=result["lagb_width"],
                alpha=result["lagb_alpha"],
                linestyles=_stm.get(result["lagb_style"], "dashed"),
                antialiaseds=True,
                capstyle="round",
                joinstyle="round",
                zorder=11,
            ))

        # Twin boundary overlay
        for name, data in result["twin_data"].items():
            ax.add_collection(LineCollection(
                data["contours"],
                colors=data["color"],
                linewidths=result["twin_width"],
                alpha=result["twin_alpha"],
                antialiaseds=True,
                capstyle="round",
                joinstyle="round",
                zorder=12,
            ))

        # Update twin stats label
        ts = result.get("twin_stats", "")
        self.lbl_twin_stats.setText(ts if ts else "-")

        # Scale bar
        if result["show_scalebar"]:
            ax.add_artist(ScaleBar(
                dx=result["step_size"],
                units="um",
                dimension="si-length",
                location=result["scalebar_loc"],
                length_fraction=result["scalebar_frac"],
                box_alpha=0.6,
                frameon=True,
                color="black",
            ))

        fig.tight_layout()
        self.viewer.canvas.draw_idle()
        self.current_fig_ready = True
        self.statusBar().showMessage("Ready")

    # ================================================================
    # Pole figure / IPF windows
    # ================================================================

    def _get_pf_direction(self) -> np.ndarray:
        txt = self.combo_pf_direction.currentText()
        if "X" in txt:
            return np.array([1, 0, 0])
        if "Y" in txt:
            return np.array([0, 1, 0])
        return np.array([0, 0, 1])

    def plot_pole_figure(self) -> None:
        if self.current_map is None:
            QMessageBox.information(self, "No data", "Load a file first.")
            return
        m = self.current_map
        sym = m.crystal_sym
        quats = np.array(m.data["orientation"]).ravel()
        direction = self._get_pf_direction()
        projection = self.combo_pf_projection.currentText()
        marker = self.combo_pf_marker.currentText()
        ms = self.spin_pf_marker_size.value()
        alpha = self.spin_pf_alpha.value()

        step = max(1, len(quats) // MAX_POLE_FIGURE_POINTS)
        quats_s = quats[::step]

        if sym == "cubic":
            pole_dirs = [
                (np.array([1, 0, 0]), "{100}"),
                (np.array([1, 1, 0]), "{110}"),
                (np.array([1, 1, 1]), "{111}"),
            ]
        elif sym == "hexagonal":
            pole_dirs = [
                (np.array([0, 0, 1]), "{0001}"),
                (np.array([1, 0, 0]), "{10-10}"),
                (np.array([1, 1, 0]), "{11-20}"),
            ]
        else:
            pole_dirs = [(np.array([0, 0, 1]), "{001}")]

        n_pf = len(pole_dirs)
        fig_pf = Figure(figsize=(6 * n_pf, 6), tight_layout=True)

        for idx, (pd, pl) in enumerate(pole_dirs):
            ax_pf = fig_pf.add_subplot(1, n_pf, idx + 1)
            ax_pf.set_aspect("equal")
            ax_pf.set_title(
                f"{pl} PF — {self.combo_pf_direction.currentText()}")
            ax_pf.axis("off")
            th = np.linspace(0, 2 * np.pi, 200)
            r = np.sqrt(2)
            ax_pf.plot(r * np.cos(th), r * np.sin(th), "k-", lw=1.5)

            xp: List[float] = []
            yp: List[float] = []
            cols: List[Any] = []
            for q in quats_s:
                ps = q.conjugate.transform_vector(pd)
                if ps[2] < 0:
                    ps = -ps
                if projection == "lambert":
                    rp = np.sqrt(2 * (1 - ps[2]))
                else:
                    rp = np.tan(np.arccos(min(ps[2], 1.0)) / 2)
                phi = np.arctan2(ps[1], ps[0])
                xp.append(rp * np.cos(phi))
                yp.append(rp * np.sin(phi))
                if self.chk_pf_color_ipf.isChecked():
                    cols.append(
                        Quat.calc_ipf_colours(
                            np.array([q]), direction, sym)[:, 0])
                else:
                    cols.append([0, 0, 1])

            c = (
                np.clip(np.array(cols), 0, 1)
                if self.chk_pf_color_ipf.isChecked()
                else "blue"
            )
            ax_pf.scatter(
                xp, yp, s=ms, c=c, marker=marker,
                alpha=alpha, edgecolors="none",
            )

        win = QMainWindow(self)
        win.setWindowTitle("Pole Figure")
        win.resize(600 * n_pf, 650)
        canvas = FigureCanvas(fig_pf)
        toolbar = NavigationToolbar(canvas, win)
        cont = QWidget()
        lay = QVBoxLayout(cont)
        lay.addWidget(toolbar)
        lay.addWidget(canvas)
        win.setCentralWidget(cont)
        win.show()

    def plot_inverse_pole_figure(self) -> None:
        if self.current_map is None:
            QMessageBox.information(self, "No data", "Load a file first.")
            return
        m = self.current_map
        sym = m.crystal_sym
        quats = np.array(m.data["orientation"]).ravel()
        direction = self._get_pf_direction()
        projection = self.combo_pf_projection.currentText()
        step = max(1, len(quats) // MAX_POLE_FIGURE_POINTS)
        quats_s = list(quats[::step])
        marker = self.combo_pf_marker.currentText()
        ms = self.spin_pf_marker_size.value()
        alpha = self.spin_pf_alpha.value()

        fig_ipf = Figure(figsize=(7, 7), tight_layout=True)
        ax_ipf = fig_ipf.add_subplot(111)
        try:
            Quat.plot_ipf(
                quats_s, direction, sym,
                projection=projection, fig=fig_ipf, ax=ax_ipf,
                marker=marker, s=ms, alpha=alpha,
            )
        except Exception:
            logger.warning("plot_ipf failed; using manual fallback",
                           exc_info=True)
            alpha_f, beta_f = Quat.calc_fund_dirs(
                np.array(quats_s), direction, sym)
            ac = np.tan(alpha_f / 2)
            xp = ac * np.cos(beta_f - np.pi / 2)
            yp = ac * np.sin(beta_f - np.pi / 2)
            rgb = Quat.calc_ipf_colours(
                np.array(quats_s), direction, sym)
            ax_ipf.scatter(
                xp, yp, s=ms,
                c=np.clip(rgb.T, 0, 1),
                marker=marker, alpha=alpha,
                edgecolors="none",
            )
            ax_ipf.set_aspect("equal")
            ax_ipf.axis("off")
            ax_ipf.set_title(
                f"IPF — {self.combo_pf_direction.currentText()}")

        win = QMainWindow(self)
        win.setWindowTitle("Inverse Pole Figure")
        win.resize(700, 700)
        canvas = FigureCanvas(fig_ipf)
        toolbar = NavigationToolbar(canvas, win)
        cont = QWidget()
        lay = QVBoxLayout(cont)
        lay.addWidget(toolbar)
        lay.addWidget(canvas)
        win.setCentralWidget(cont)
        win.show()

    # ================================================================
    # Canvas click
    # ================================================================

    def on_canvas_click(self, event: Any) -> None:
        if self._render_busy:
            self.statusBar().showMessage("Rendering in progress, click later...")
            return
        if self.current_map is None or event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        try:
            m = self.current_map
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            gm = np.asarray(m.data["grains"])
            if y < 0 or y >= gm.shape[0] or x < 0 or x >= gm.shape[1]:
                return
            self.lbl_click_xy.setText(f"({x}, {y})")
            gv = int(gm[y, x])
            if gv <= 0:
                self.lbl_grain_id.setText("None")
                for lbl in [
                    self.lbl_grain_pixels, self.lbl_grain_area,
                    self.lbl_grain_eqd, self.lbl_grain_phase,
                    self.lbl_grain_avg_mis, self.lbl_grain_ref_ori,
                    self.lbl_grain_twin,
                ]:
                    lbl.setText("-")
                self.statusBar().showMessage("Boundary / non-grain")
                return
            gid = gv - 1
            grain = m[gid]
            grain_mask = gm == gv
            px = int(grain_mask.sum())
            area = px * (m.step_size ** 2)
            eqd = float(np.sqrt(4.0 * area / np.pi))

            # Phase ID
            pid = -1
            try:
                pm = np.asarray(m.data["phase"])
                pv = pm[grain_mask]
                pv = pv[pv > 0]
                if pv.size:
                    pid = int(np.bincount(pv).argmax())
            except Exception:
                logger.debug("Could not determine phase for grain %d", gid)

            # Reference orientation
            try:
                grain.calc_average_ori()
                ro = getattr(grain, "ref_ori", None)
                rot = str(ro) if ro else "-"
            except Exception:
                logger.debug("Could not compute ref_ori for grain %d", gid)
                rot = "-"

            # Average misorientation
            try:
                grain.build_mis_ori_list()
                am = getattr(grain, "average_mis_ori", None)
                amt = "-" if am is None else f"{float(am):.3f}°"
            except Exception:
                logger.debug("Could not compute avg misori for grain %d", gid)
                amt = "-"

            self.lbl_grain_id.setText(str(gid))
            self.lbl_grain_pixels.setText(str(px))
            self.lbl_grain_area.setText(f"{area:.2f} µm²")
            self.lbl_grain_eqd.setText(f"{eqd:.2f} µm")
            self.lbl_grain_phase.setText(str(pid))
            self.lbl_grain_avg_mis.setText(amt)
            self.lbl_grain_ref_ori.setText(rot)
            self.lbl_grain_twin.setText("-")
            self.statusBar().showMessage(f"Selected grain {gid}")
        except Exception as exc:
            logger.error("Click handler failed", exc_info=True)
            self.statusBar().showMessage(f"Click failed: {exc}")

    # ================================================================
    # File actions
    # ================================================================

    def open_file(self) -> None:
        ps, _ = QFileDialog.getOpenFileName(
            self, "Open CPR file", "", "Oxford CPR files (*.cpr)")
        if not ps:
            return
        p = Path(ps)
        if not p.with_suffix(".crc").exists():
            QMessageBox.warning(
                self, "Missing CRC",
                f"No CRC found:\n{p.with_suffix('.crc')}")
            return
        self.current_path = p
        self._invalidate_all_caches()
        self.statusBar().showMessage(f"Loaded: {p}")
        self._update_control_states(True)
        self.refresh_plot()

    def refresh_plot(self) -> None:
        if self.current_path is None:
            return
        try:
            m = self._load_map()
            self.current_map = m
            self._update_info(m)
        except Exception as exc:
            logger.error("Map load failed", exc_info=True)
            QMessageBox.critical(self, "Load error", str(exc))
            return
        params = self._collect_render_params(m)
        if self._render_busy:
            self._pending_params = params
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage("Rendering...")
        self._start_render(params)

    def save_image(self) -> None:
        if not self.current_fig_ready:
            QMessageBox.information(
                self, "No figure", "Render a map first.")
            return
        fmt = self.combo_save_fmt.currentText()
        fm = {
            "PNG": (".png", "PNG (*.png)"),
            "TIFF": (".tif", "TIFF (*.tif *.tiff)"),
            "JPEG": (".jpg", "JPEG (*.jpg *.jpeg)"),
        }
        ext, _flt = fm[fmt]
        bn = self.edit_save_name.text().strip() or "ebsd_preview"
        for e in [".png", ".tif", ".tiff", ".jpg", ".jpeg"]:
            if bn.lower().endswith(e):
                bn = bn[: -len(e)]
                break
        af = ";;".join([fm["PNG"][1], fm["TIFF"][1], fm["JPEG"][1]])
        sd = (
            str(self.current_path.parent / (bn + ext))
            if self.current_path
            else bn + ext
        )
        ps, _ = QFileDialog.getSaveFileName(self, "Save Image", sd, af)
        if not ps:
            return
        out = Path(ps)
        dpi = self.spin_save_dpi.value()
        kw: Dict[str, Any] = {"dpi": dpi, "bbox_inches": "tight"}
        s = out.suffix.lower()
        if s in (".jpg", ".jpeg"):
            kw["facecolor"] = "white"
            kw["pil_kwargs"] = {"quality": 95}
        elif s in (".tif", ".tiff"):
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        try:
            self.viewer.figure.savefig(str(out), **kw)
            self.statusBar().showMessage(f"Saved: {out} ({dpi} dpi)")
        except Exception as exc:
            logger.error("Save failed", exc_info=True)
            QMessageBox.critical(self, "Save error", f"{exc}")

    def export_ctf(self) -> None:
        if self.current_path is None:
            QMessageBox.information(self, "No file", "Load first.")
            return
        ps, _ = QFileDialog.getSaveFileName(
            self, "Export CTF",
            str(self.current_path.with_suffix(".ctf")),
            "Oxford Text (*.ctf)")
        if not ps:
            return
        out = Path(ps)
        if out.exists():
            r = QMessageBox.question(
                self, "Overwrite?", f"Overwrite?\n{out}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if r != QMessageBox.StandardButton.Yes:
                return
            out.unlink()
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage("Exporting CTF...")
            m2 = self._load_map()
            m2.save(
                file_name=out.stem,
                data_type="OxfordText",
                file_dir=str(out.parent),
            )
            self.statusBar().showMessage(f"Exported: {out}")
        except Exception as exc:
            logger.error("CTF export failed", exc_info=True)
            QMessageBox.critical(self, "Export error", f"{exc}")
        finally:
            QApplication.restoreOverrideCursor()


# ====================================================================
# Entry point
# ====================================================================


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = QApplication(sys.argv)
    w = EbsdMainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())