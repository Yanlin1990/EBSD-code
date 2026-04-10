"""EBSD Prototype - PySide6 + DefDAP (Architecture v2)

A graphical application for Electron Backscatter Diffraction (EBSD)
data visualisation and analysis.  Built on PySide6 (Qt 6) and the
DefDAP library, it supports:

* Euler / IPF / Band-Contrast / KAM / Misorientation / Grain / Boundary maps
* BC + IPF blending with multiple modes (overlay / multiply / soft_light)
* HAGB / LAGB boundary overlay with smoothing
* Twin-boundary detection (cubic & hexagonal systems)
* Pole-figure and inverse-pole-figure windows
* Non-indexed pixel filling and denoising
* Image export (PNG / TIFF / JPEG) and CTF export

Architecture (v2)
-----------------
The application uses a strict four-layer design to ensure the GUI never
blocks during computation and all computation functions are independently
testable:

  RenderParams (frozen dataclass)
      |  immutable snapshot of UI parameters
  EbsdEngine (pure Python, zero Qt dependency)
      |  result dict (numpy arrays + metadata)
  RenderWorker (QObject, runs in QThread)
      |  Signal[dict] on completion
  EbsdMainWindow (pure GUI layer)

Key improvements over the previous version
------------------------------------------
- Strict separation: EbsdEngine has zero Qt imports / widget access.
- Parameter-driven cache: each computation step keyed by its input
  parameters; automatically invalidated when parameters change.
- True background rendering: QThread + QObject; GUI never freezes.
- Pending-request anti-re-entrancy: if a render is requested while one
  is already running, the latest request is stored and dispatched when
  the current one completes.
- Separate Map instances for HAGB and LAGB to prevent state collisions.
- Truly vectorised median fill using scipy.ndimage.median_filter.
- Proper eigenvalue-based grain average orientation for twin detection.
- Zero QApplication.processEvents() calls.
- Fixed defaults: Median kernel=3, Gaussian sigma=1.0.
- Correct direction matching: mt.endswith("-X") not "X" in mt.
- Reliable RGB normalisation threshold: mx > 1.001.
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.ndimage import (
    binary_dilation as _binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    gaussian_filter1d,
    label as ndlabel,
    median_filter,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib_scalebar.scalebar import ScaleBar
from skimage.measure import find_contours

from PySide6.QtCore import Qt, QObject, QThread, QTimer, Signal
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

__all__ = ["EbsdMainWindow"]

logger = logging.getLogger(__name__)

# ====================================================================
# Named constants
# ====================================================================

MAX_POLE_FIGURE_POINTS: int = 5000
"""Maximum number of orientations sampled for pole-figure scatter."""

QUATERNION_SINGULARITY_THRESHOLD: float = 1e-6
"""Below this sin(half-angle) the rotation axis is ill-defined."""

BRIGHTNESS_CHANGE_EPSILON: float = 0.01
"""Minimum BC-brightness deviation from 1.0 to trigger adjustment."""

CONTRAST_CHANGE_EPSILON: float = 0.01
"""Minimum BC-contrast deviation from 1.0 to trigger adjustment."""

SMOOTH_CONTOUR_LENGTH_SCALE: float = 30.0
"""Contour length (px) used to scale adaptive Gaussian sigma."""

SMOOTH_CONTOUR_MIN_SIGMA_FRAC: float = 0.2
"""Minimum fraction of sigma kept when contour is very short."""

EULER_ZERO_THRESHOLD: float = 1e-10
"""Sum-of-absolute Euler angles below this -- treated as non-indexed."""

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

MAX_AVG_ORI_SAMPLES: int = 500
"""Maximum pixels sampled per grain for eigenvalue average orientation."""


# ====================================================================
# Twin orientation-relationship definitions
# ====================================================================

TWIN_SYSTEMS: Dict[str, List[dict]] = {
    "cubic": [
        {
            "name": "\u03a33 {111}<112>",
            "angle_deg": 60.0,
            "axis": np.array([1, 1, 1], dtype=float),
            "tolerance_deg": 5.0,
            "color": "red",
        },
        {
            "name": "\u03a39 {114}<221>",
            "angle_deg": 38.94,
            "axis": np.array([1, 1, 0], dtype=float),
            "tolerance_deg": 5.0,
            "color": "blue",
        },
        {
            "name": "\u03a327a",
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
        sin_half = np.sqrt(max(0.0, 1.0 - dq_coef[0] ** 2))
        if sin_half < QUATERNION_SINGULARITY_THRESHOLD:
            return True
        axis_m = _normalize(dq_coef[1:4] / sin_half)
        axis_t = twin_def["axis"].copy()
        if len(axis_t) == 4:
            axis_t = _mb_to_miller_dir(axis_t)
        axis_t = _normalize(axis_t)
        return abs(np.dot(axis_m, axis_t)) > np.cos(
            np.radians(twin_def["tolerance_deg"])
        )
    except Exception:
        logger.warning(
            "Twin axis check failed for %s -- treating as non-twin",
            twin_def["name"],
            exc_info=True,
        )
        return False


# ====================================================================
# Vectorised pixel-fill helpers
# ====================================================================


def _neighbor_fill_2d(
    img: np.ndarray,
    mask: np.ndarray,
    max_iter: int,
) -> np.ndarray:
    """Iterative 8-neighbour mean fill for a 2-D array (vectorised)."""
    img = img.astype(np.float32)
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
                neighbour_cnt += vm.astype(np.float32)
        fillable = remaining & (neighbour_cnt > 0)
        if not np.any(fillable):
            break
        img[fillable] = neighbour_sum[fillable] / neighbour_cnt[fillable]
        remaining[fillable] = False
    if np.any(remaining) and np.any(~remaining):
        _, nearest = distance_transform_edt(
            remaining, return_distances=True, return_indices=True,
        )
        img[remaining] = img[nearest[0][remaining], nearest[1][remaining]]
    return img


def _neighbor_fill_rgb(
    img: np.ndarray,
    mask: np.ndarray,
    max_iter: int,
) -> np.ndarray:
    """Iterative 8-neighbour mean fill for an (H, W, C) RGB array."""
    img = img.astype(np.float32)
    remaining = mask.copy()
    h, w = mask.shape
    n_ch = img.shape[2]
    for _ in range(max_iter):
        if not np.any(remaining):
            break
        padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
        valid_pad = np.pad(~remaining, 1, constant_values=False)
        neighbour_sum = np.zeros_like(img)
        neighbour_cnt = np.zeros((h, w), dtype=np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                sl = padded[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w, :]
                vm = valid_pad[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w]
                neighbour_sum += sl * vm[..., np.newaxis]
                neighbour_cnt += vm.astype(np.float32)
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


def _median_fill_2d(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """True vectorised median fill using scipy.ndimage.median_filter."""
    if not np.any(mask):
        return img
    filtered = median_filter(img.astype(np.float32), size=3)
    result = img.copy().astype(np.float32)
    result[mask] = filtered[mask]
    return result


def _median_fill_rgb(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """True vectorised median fill for (H, W, C) using scipy."""
    if not np.any(mask):
        return img
    img = img.astype(np.float32)
    result = img.copy()
    for ch in range(img.shape[2]):
        filtered = median_filter(img[:, :, ch], size=3)
        result[:, :, ch][mask] = filtered[mask]
    return result


def find_neighbor_pairs(gm: np.ndarray) -> Set[Tuple[int, int]]:
    """Find unique adjacent grain-ID pairs using vectorised operations."""
    valid = gm > 0
    h_mask = valid[:, :-1] & valid[:, 1:]
    h_left = gm[:, :-1][h_mask]
    h_right = gm[:, 1:][h_mask]
    v_mask = valid[:-1, :] & valid[1:, :]
    v_top = gm[:-1, :][v_mask]
    v_bot = gm[1:, :][v_mask]
    all_left = np.concatenate([h_left, v_top])
    all_right = np.concatenate([h_right, v_bot])
    diff = all_left != all_right
    all_left = all_left[diff]
    all_right = all_right[diff]
    lo = np.minimum(all_left, all_right)
    hi = np.maximum(all_left, all_right)
    pairs = np.unique(np.stack([lo, hi], axis=1), axis=0)
    return set(map(tuple, pairs))


# ====================================================================
# RenderParams -- immutable parameter snapshot
# ====================================================================


@dataclass(frozen=True)
class RenderParams:
    """Frozen snapshot of all UI rendering parameters.

    Created in the main thread by ``EbsdMainWindow._snapshot_params()``
    and passed into ``RenderWorker`` / ``EbsdEngine.compute()``.  Because
    it is frozen and contains only primitive types, it is safe to share
    across threads without copying.
    """

    path: str
    map_type: str
    gb_angle: float
    min_grain: int
    kam_max: float
    misori_max: float
    noindex_method: str
    noindex_iter: int
    denoise_method: str
    median_kernel: int
    gauss_sigma: float
    bc_brightness: float
    bc_contrast: float
    bc_ipf_alpha: float
    bc_ipf_mode: str
    gb_color: str
    gb_width: float
    gb_alpha: float
    gb_smooth: float
    hole_fill: int
    frag_merge: int
    show_hagb: bool
    show_lagb: bool
    lagb_min: float
    lagb_max: float
    lagb_color: str
    lagb_width: float
    lagb_alpha: float
    lagb_smooth: float
    lagb_style: str
    show_twins: bool
    twin_tol: float
    twin_width: float
    twin_alpha: float
    strict_axis: bool
    # Tuple of (name, angle_deg, axis_tuple, sym, tol_deg, color) tuples
    active_twin_defs: Tuple
    scalar_cmap: str
    show_colorbar: bool
    show_scalebar: bool
    scalebar_frac: float
    scalebar_loc: str


# ====================================================================
# EbsdEngine -- pure computation, zero Qt dependency
# ====================================================================


class EbsdEngine:
    """Pure-Python computation engine.

    Has no imports from Qt and accesses no Qt widgets.  All public
    methods receive their inputs as explicit arguments (no global GUI
    state).  Internally maintains a parameter-driven cache to avoid
    redundant computation when parameters have not changed.

    Two ``ebsd.Map`` instances are kept -- one for HAGB analysis
    (``_main_map``) and one for LAGB analysis (``_lagb_map``).  They
    are loaded from the same file but operated on independently, so
    calling ``generate("grain_boundaries", misori_tol=lagb_min)`` on
    the LAGB instance does not overwrite the HAGB grain-boundary data.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[Any, Any]] = {}
        self._main_map: Optional[ebsd.Map] = None
        self._lagb_map: Optional[ebsd.Map] = None

    # ------------------------------------------------------------------
    # Map lifecycle
    # ------------------------------------------------------------------

    def load(self, path: Path) -> None:
        """Load a new EBSD map file (invalidates all cache entries)."""
        t0 = time.perf_counter()
        self._main_map = ebsd.Map(path)
        self._lagb_map = ebsd.Map(path)
        self.invalidate_all()
        logger.debug("EbsdEngine.load() took %.3fs", time.perf_counter() - t0)

    def invalidate_all(self) -> None:
        """Clear all cached computation results."""
        self._cache.clear()

    @property
    def main_map(self) -> Optional[ebsd.Map]:
        return self._main_map

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cached(self, name: str, key: Any, compute_fn: Callable) -> Any:
        """Return cached result if key matches, else compute and cache."""
        if name in self._cache:
            cached_key, cached_val = self._cache[name]
            if cached_key == key:
                return cached_val
        t0 = time.perf_counter()
        val = compute_fn()
        logger.debug(
            "Cache miss '%s': computed in %.3fs", name, time.perf_counter() - t0
        )
        self._cache[name] = (key, val)
        return val

    # ------------------------------------------------------------------
    # Grain generation (tracks parameters to detect changes)
    # ------------------------------------------------------------------

    def _ensure_grains(
        self,
        m: ebsd.Map,
        gb_angle: float,
        min_grain: int,
        cache_prefix: str = "",
    ) -> None:
        """Generate grain boundaries + grains on *m* if params changed."""
        param_key = (gb_angle, min_grain)
        cache_name = cache_prefix + "_grain_gen_params"
        if cache_name in self._cache:
            cached_key, _ = self._cache[cache_name]
            if cached_key == param_key:
                return  # Already generated with these params
        t0 = time.perf_counter()
        m.data.generate("grain_boundaries", misori_tol=gb_angle)
        m.data.generate("grains", min_grain_size=min_grain)
        logger.debug(
            "_ensure_grains(%s) took %.3fs",
            cache_prefix or "main",
            time.perf_counter() - t0,
        )
        self._cache[cache_name] = (param_key, True)

    # ------------------------------------------------------------------
    # Static computation helpers (pure functions, no Qt, no self state)
    # ------------------------------------------------------------------

    @staticmethod
    def get_non_indexed_mask(m: ebsd.Map) -> np.ndarray:
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
            return np.zeros((h or 1, w or 1), dtype=bool)
        return mask

    @staticmethod
    def get_bc_array(
        m: ebsd.Map,
        brightness: float,
        contrast: float,
    ) -> np.ndarray:
        """Return normalised, brightness/contrast-adjusted BC array."""
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

    @staticmethod
    def compute_ipf_rgb(m: ebsd.Map, direction: np.ndarray) -> np.ndarray:
        """Compute IPF colourmap.  Returns (H, W, 3) float32.

        Derives the map shape from ``euler_angle`` (shape (3, H, W)) to
        avoid the reshape bug that occurs when ``m.data['orientation']``
        is stored as a flat 1-D array of Quat objects.
        """
        try:
            ea = np.asarray(m.data.euler_angle)
            H, W = int(ea.shape[1]), int(ea.shape[2])
        except Exception:
            try:
                bc = np.asarray(m.data.band_contrast)
                H, W = bc.shape
            except Exception:
                raise RuntimeError("Cannot determine map dimensions for IPF")
        quats = np.asarray(m.data["orientation"]).ravel()
        rgb = Quat.calc_ipf_colours(quats, direction, m.crystal_sym)  # (3, N)
        return rgb.T.reshape(H, W, 3).astype(np.float32)

    @staticmethod
    def get_euler_rgb(m: ebsd.Map) -> np.ndarray:
        """Map Euler angles phi1/Phi/phi2 to R/G/B.  Returns (H, W, 3) float32."""
        ea = np.asarray(m.data.euler_angle).astype(np.float32)  # (3, H, W)
        r = np.clip(ea[0] / (2.0 * np.pi), 0.0, 1.0)
        g = np.clip(ea[1] / np.pi, 0.0, 1.0)
        b = np.clip(ea[2] / (2.0 * np.pi), 0.0, 1.0)
        return np.dstack([r, g, b]).astype(np.float32)

    @staticmethod
    def normalize_rgb(a: np.ndarray) -> np.ndarray:
        """Normalise an RGB array to [0, 1] float32."""
        if a.dtype == np.uint8:
            a = a.astype(np.float32) / 255.0
        elif a.dtype != np.float32:
            a = a.astype(np.float32)
        mx = float(a.max())
        if mx > 1.001:
            a = a / mx
        if a.ndim == 3 and a.shape[2] == 4:
            a = a[:, :, :3]
        return np.clip(a, 0.0, 1.0)

    @staticmethod
    def blend_bc_ipf(
        bc: np.ndarray,
        ipf_rgb: np.ndarray,
        alpha: float,
        mode: str,
    ) -> np.ndarray:
        """Blend a greyscale BC map with an IPF RGB map."""
        bc3 = np.dstack([bc, bc, bc]).astype(np.float32)
        if mode == "multiply":
            mult = bc3 * ipf_rgb
            result = bc3 * (1.0 - alpha) + mult * alpha
            result = result * MULTIPLY_BLEND_BOOST
        elif mode == "soft_light":
            m_ = ipf_rgb <= 0.5
            soft = np.where(
                m_,
                bc3 - (1 - 2 * ipf_rgb) * bc3 * (1 - bc3),
                bc3 + (2 * ipf_rgb - 1) * (np.sqrt(np.maximum(bc3, 0)) - bc3),
            )
            result = bc3 * (1.0 - alpha) + soft * alpha
        else:  # overlay (default)
            result = bc3 * (1.0 - alpha) + ipf_rgb * alpha
        return np.clip(result, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def fill_non_indexed(
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
                return _median_fill_rgb(img, mask)
            return _median_fill_2d(img, mask)
        # "fill (neighbor)" -- vectorised iterative 8-neighbour fill
        if is_rgb:
            return _neighbor_fill_rgb(img, mask, max_iter)
        return _neighbor_fill_2d(img, mask, max_iter)

    @staticmethod
    def denoise_2d(
        a: np.ndarray,
        method: str,
        kernel: int,
        sigma: float,
    ) -> np.ndarray:
        """Apply denoise filter to a 2-D array."""
        if method == "Median":
            k = kernel
            if k < 2:
                return a
            k = k if k % 2 == 1 else k + 1
            return median_filter(a.astype(np.float32), size=k)
        if method == "Gaussian":
            if sigma < 0.01:
                return a
            return gaussian_filter(
                a.astype(np.float64), sigma=sigma
            ).astype(np.float32)
        return a

    @staticmethod
    def denoise_rgb(
        a: np.ndarray,
        method: str,
        kernel: int,
        sigma: float,
    ) -> np.ndarray:
        """Apply denoise filter to an (H, W, C) array."""
        if method == "Median":
            k = kernel
            if k < 2:
                return a
            k = k if k % 2 == 1 else k + 1
            return np.dstack([
                median_filter(a[:, :, i].astype(np.float32), size=k)
                for i in range(a.shape[2])
            ]).astype(np.float32)
        if method == "Gaussian":
            if sigma < 0.01:
                return a
            out = a.astype(np.float64)
            for i in range(a.shape[2]):
                out[:, :, i] = gaussian_filter(out[:, :, i], sigma=sigma)
            return out.astype(np.float32)
        return a

    @staticmethod
    def build_clean_grain_map(
        m: ebsd.Map,
        hole_fill: int,
        frag_merge: int,
    ) -> np.ndarray:
        """Build a cleaned grain-ID map by filling holes and merging fragments."""
        gm = np.asarray(m.data["grains"]).copy().astype(np.int32)
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

    @staticmethod
    def _smooth_contour(
        x: np.ndarray,
        y: np.ndarray,
        sigma: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Smooth a contour path using adaptive Gaussian filter."""
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

    @staticmethod
    def contours_from_grain_map(
        grain_map: np.ndarray,
        sigma: float,
    ) -> List[np.ndarray]:
        """Extract smoothed boundary contours from a grain-ID map."""
        uids = np.unique(grain_map)
        uids = uids[uids > 0]
        all_c: List[np.ndarray] = []
        for gid in uids:
            mask_f = (grain_map == gid).astype(np.float32)
            for c in find_contours(mask_f, level=0.5):
                if c.shape[0] < MIN_CONTOUR_POINTS:
                    continue
                y, x = c[:, 0], c[:, 1]
                x, y = EbsdEngine._smooth_contour(x, y, sigma)
                all_c.append(np.column_stack([x, y]))
        return all_c

    @staticmethod
    def _contiguous_segments(
        mask_1d: np.ndarray,
        min_length: int,
    ) -> List[Tuple[int, int]]:
        """Return (start, end) pairs of contiguous True runs in mask_1d."""
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

    @staticmethod
    def extract_lagb_contours(
        sub_gm: np.ndarray,
        main_gm: np.ndarray,
        sigma: float,
    ) -> List[np.ndarray]:
        """Extract LAGB contours as segments within each HAGB grain."""
        h_map, w_map = main_gm.shape
        all_c: List[np.ndarray] = []
        for ha_id in np.unique(main_gm):
            if ha_id <= 0:
                continue
            ha_mask = main_gm == ha_id
            sub_ids = np.unique(sub_gm[ha_mask])
            sub_ids = sub_ids[sub_ids > 0]
            if len(sub_ids) <= 1:
                continue
            for sid in sub_ids:
                sm = (sub_gm == sid) & ha_mask
                if sm.sum() < 3:
                    continue
                for c in find_contours(sm.astype(np.float32), level=0.5):
                    if c.shape[0] < MIN_CONTOUR_POINTS:
                        continue
                    y, x = c[:, 0], c[:, 1]
                    ix = np.clip(np.round(x).astype(int), 0, w_map - 1)
                    iy = np.clip(np.round(y).astype(int), 0, h_map - 1)
                    inside = main_gm[iy, ix] == ha_id
                    segs = EbsdEngine._contiguous_segments(
                        inside, MIN_LAGB_SEGMENT_POINTS
                    )
                    for s, e in segs:
                        sx, sy = x[s:e], y[s:e]
                        if len(sx) > 4:
                            sx, sy = EbsdEngine._smooth_contour(sx, sy, sigma)
                        all_c.append(np.column_stack([sx, sy]))
        return all_c

    @staticmethod
    def avg_quaternion_eigenvalue(quats: List[Quat]) -> Quat:
        """Compute mean orientation via eigenvalue decomposition."""
        if len(quats) == 1:
            return quats[0]
        M = sum(np.outer(q.quat_coef, q.quat_coef) for q in quats)
        _eigvals, eigvecs = np.linalg.eigh(M)
        return Quat(eigvecs[:, -1])

    @staticmethod
    def grain_ids_to_rgb(gm: np.ndarray) -> np.ndarray:
        """Convert grain-ID map to (H, W, 3) float32 with deterministic colours."""
        h, w = gm.shape
        rgb = np.ones((h, w, 3), dtype=np.float32)
        uids = np.unique(gm)
        uids = uids[uids > 0]
        if uids.size == 0:
            return rgb
        rng = np.random.default_rng(42)
        colors = rng.uniform(0.2, 1.0, size=(len(uids), 3)).astype(np.float32)
        for i, gid in enumerate(uids):
            rgb[gm == gid] = colors[i]
        return rgb

    @staticmethod
    def _bc_ipf_direction(mt: str) -> np.ndarray:
        if mt.endswith("-X"):
            return np.array([1, 0, 0], dtype=float)
        if mt.endswith("-Y"):
            return np.array([0, 1, 0], dtype=float)
        return np.array([0, 0, 1], dtype=float)

    # ------------------------------------------------------------------
    # LAGB preparation (uses separate _lagb_map)
    # ------------------------------------------------------------------

    def _prepare_lagb_data(
        self,
        params: RenderParams,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate LAGB sub-grain map and main clean grain map.

        Uses the independent ``_lagb_map`` instance so that calling
        ``generate("grain_boundaries", misori_tol=lagb_min)`` does NOT
        overwrite the HAGB grain-boundary data on ``_main_map``.
        """
        lagb_min_grain = max(params.min_grain // 2, 2)
        self._lagb_map.data.generate(
            "grain_boundaries", misori_tol=params.lagb_min
        )
        self._lagb_map.data.generate(
            "grains", min_grain_size=lagb_min_grain
        )
        sub_gm = np.asarray(self._lagb_map.data["grains"]).copy()
        # Main grain map -- ensure grains generated with current HAGB params
        self._ensure_grains(
            self._main_map, params.gb_angle, params.min_grain
        )
        main_clean_gm = self.build_clean_grain_map(
            self._main_map, params.hole_fill, params.frag_merge
        )
        return sub_gm, main_clean_gm

    # ------------------------------------------------------------------
    # Twin detection
    # ------------------------------------------------------------------

    def detect_twin_boundaries(
        self,
        m: ebsd.Map,
        clean_gm: np.ndarray,
        params: RenderParams,
    ) -> dict:
        """Detect twin boundaries and return contour data per twin type."""
        sym = m.crystal_sym
        # Reconstruct twin_def dicts from params (axis stored as hashable tuple)
        twin_defs = []
        for name, angle_deg, axis_t, sym_req, tol_deg, color in params.active_twin_defs:
            if sym_req != sym:
                continue
            twin_defs.append({
                "name": name,
                "angle_deg": angle_deg,
                "axis": np.array(axis_t, dtype=float),
                "tolerance_deg": tol_deg,
                "color": color,
            })
        if not twin_defs:
            return {}

        quats = np.asarray(m.data["orientation"])
        h, w = clean_gm.shape
        strict = params.strict_axis
        check_fn = check_twin_with_axis if strict else check_twin_relation

        # Build grain average orientations using eigenvalue method
        grain_ids = np.unique(clean_gm)
        grain_ids = grain_ids[grain_ids > 0]
        grain_avg_ori: Dict[int, Quat] = {}
        rng = np.random.default_rng(0)
        for gid in grain_ids:
            ys, xs = np.where(clean_gm == gid)
            if len(ys) > MAX_AVG_ORI_SAMPLES:
                idx = rng.choice(len(ys), MAX_AVG_ORI_SAMPLES, replace=False)
                ys, xs = ys[idx], xs[idx]
            q_list = [quats[int(y), int(x)] for y, x in zip(ys, xs)]
            try:
                grain_avg_ori[gid] = self.avg_quaternion_eigenvalue(q_list)
            except Exception:
                grain_avg_ori[gid] = q_list[0]

        neighbor_pairs = find_neighbor_pairs(clean_gm)
        twin_pairs: Dict[str, set] = {td["name"]: set() for td in twin_defs}
        for g_a, g_b in neighbor_pairs:
            if g_a not in grain_avg_ori or g_b not in grain_avg_ori:
                continue
            for td in twin_defs:
                try:
                    if check_fn(grain_avg_ori[g_a], grain_avg_ori[g_b], sym, td):
                        twin_pairs[td["name"]].add((g_a, g_b))
                        break
                except Exception:
                    logger.debug(
                        "Twin check error for pair (%d, %d)", g_a, g_b,
                        exc_info=True,
                    )

        result: dict = {}
        sigma = params.gb_smooth
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
                        x_c, y_c = self._smooth_contour(x_c, y_c, sigma)
                    contours.append(np.column_stack([x_c, y_c]))
            if contours:
                result[td["name"]] = {
                    "contours": contours,
                    "color": td["color"],
                }
        return result

    # ------------------------------------------------------------------
    # Main computation entry point
    # ------------------------------------------------------------------

    def compute(self, params: RenderParams) -> dict:
        """Compute all rendering data based on *params*.

        Returns a result dict containing everything the main thread needs
        to render the matplotlib figure.  No Qt widgets are accessed.
        """
        m = self._main_map
        if m is None:
            raise RuntimeError("No map loaded -- call EbsdEngine.load() first")

        t0 = time.perf_counter()
        path = params.path
        mt = params.map_type

        # Non-indexed mask (cached by path only -- does not change per session)
        mask = self._cached(
            "mask", path, lambda: self.get_non_indexed_mask(m)
        )

        result: dict = {
            "map_type": mt,
            "step_size": m.step_size,
            "show_scalebar": params.show_scalebar,
            "scalebar_loc": params.scalebar_loc,
            "scalebar_frac": params.scalebar_frac,
            "show_colorbar": params.show_colorbar,
            "scalar_cmap": params.scalar_cmap,
            "hagb_contours": None,
            "lagb_contours": None,
            "twin_data": None,
            "twin_width": params.twin_width,
            "twin_alpha": params.twin_alpha,
            "gb_color": params.gb_color,
            "gb_width": params.gb_width,
            "gb_alpha": params.gb_alpha,
            "lagb_color": params.lagb_color,
            "lagb_width": params.lagb_width,
            "lagb_alpha": params.lagb_alpha,
            "lagb_style": params.lagb_style,
            "vmin": None,
            "vmax": None,
            "scalar_label": None,
            "image_type": "rgb",
        }

        # ----------------------------------------------------------------
        # Compute the main image
        # ----------------------------------------------------------------

        if mt.startswith("BC+IPF"):
            direction = self._bc_ipf_direction(mt)
            dir_t = tuple(direction.tolist())

            bc_key = (path, params.bc_brightness, params.bc_contrast)
            bc = self._cached(
                "bc", bc_key,
                lambda: self.get_bc_array(m, params.bc_brightness, params.bc_contrast),
            )
            bc = self.fill_non_indexed(
                bc, mask, params.noindex_method, params.noindex_iter
            )
            bc = self.denoise_2d(
                bc, params.denoise_method, params.median_kernel, params.gauss_sigma
            )

            ipf_key = (path, dir_t)
            ipf_rgb = self._cached(
                "ipf_" + mt[-1], ipf_key,
                lambda: self.compute_ipf_rgb(m, direction),
            )
            ipf_rgb = self.normalize_rgb(ipf_rgb)
            ipf_rgb = self.fill_non_indexed(
                ipf_rgb, mask, params.noindex_method, params.noindex_iter
            )
            ipf_rgb = self.denoise_rgb(
                ipf_rgb, params.denoise_method, params.median_kernel, params.gauss_sigma
            )
            image = self.blend_bc_ipf(
                bc, ipf_rgb, params.bc_ipf_alpha, params.bc_ipf_mode
            )
            result["image"] = image
            result["image_type"] = "rgb"

        elif mt == "Band Contrast":
            bc_key = (path, params.bc_brightness, params.bc_contrast)
            bc = self._cached(
                "bc", bc_key,
                lambda: self.get_bc_array(m, params.bc_brightness, params.bc_contrast),
            )
            bc = self.fill_non_indexed(
                bc, mask, params.noindex_method, params.noindex_iter
            )
            bc = self.denoise_2d(
                bc, params.denoise_method, params.median_kernel, params.gauss_sigma
            )
            result["image"] = bc
            result["image_type"] = "scalar"
            result["scalar_label"] = "Band Contrast"
            result["vmin"] = 0.0
            result["vmax"] = 1.0

        elif mt == "Euler":
            euler_rgb = self._cached(
                "euler", path, lambda: self.get_euler_rgb(m)
            )
            euler_rgb = self.normalize_rgb(euler_rgb)
            euler_rgb = self.fill_non_indexed(
                euler_rgb, mask, params.noindex_method, params.noindex_iter
            )
            euler_rgb = self.denoise_rgb(
                euler_rgb, params.denoise_method, params.median_kernel, params.gauss_sigma
            )
            result["image"] = euler_rgb
            result["image_type"] = "rgb"

        elif mt in ("IPF-X", "IPF-Y", "IPF-Z"):
            dir_map: Dict[str, np.ndarray] = {
                "IPF-X": np.array([1, 0, 0], dtype=float),
                "IPF-Y": np.array([0, 1, 0], dtype=float),
                "IPF-Z": np.array([0, 0, 1], dtype=float),
            }
            direction = dir_map[mt]
            dir_t = tuple(direction.tolist())
            ipf_key = (path, dir_t)
            ipf_rgb = self._cached(
                "ipf_" + mt[-1], ipf_key,
                lambda: self.compute_ipf_rgb(m, direction),
            )
            ipf_rgb = self.normalize_rgb(ipf_rgb)
            ipf_rgb = self.fill_non_indexed(
                ipf_rgb, mask, params.noindex_method, params.noindex_iter
            )
            ipf_rgb = self.denoise_rgb(
                ipf_rgb, params.denoise_method, params.median_kernel, params.gauss_sigma
            )
            result["image"] = ipf_rgb
            result["image_type"] = "rgb"

        elif mt == "KAM":
            self._ensure_grains(m, params.gb_angle, params.min_grain)
            m.calc_kam()
            kam = np.asarray(m.data["KAM"]).astype(np.float32)
            kam = self.fill_non_indexed(
                kam, mask, params.noindex_method, params.noindex_iter
            )
            kam = self.denoise_2d(
                kam, params.denoise_method, params.median_kernel, params.gauss_sigma
            )
            result["image"] = kam
            result["image_type"] = "scalar"
            result["scalar_label"] = "KAM (\u00b0)"
            result["vmin"] = 0.0
            result["vmax"] = params.kam_max

        elif mt == "Misorientation":
            self._ensure_grains(m, params.gb_angle, params.min_grain)
            m.calc_grain_mis_ori()
            mis = np.asarray(m.data["mis_ori"]).astype(np.float32)
            mis = self.fill_non_indexed(
                mis, mask, params.noindex_method, params.noindex_iter
            )
            mis = self.denoise_2d(
                mis, params.denoise_method, params.median_kernel, params.gauss_sigma
            )
            result["image"] = mis
            result["image_type"] = "scalar"
            result["scalar_label"] = "Misorientation (\u00b0)"
            result["vmin"] = 0.0
            result["vmax"] = params.misori_max

        elif mt == "Grain":
            self._ensure_grains(m, params.gb_angle, params.min_grain)
            gm_key = (
                path, params.gb_angle, params.min_grain,
                params.hole_fill, params.frag_merge,
            )
            clean_gm = self._cached(
                "clean_gm", gm_key,
                lambda: self.build_clean_grain_map(
                    m, params.hole_fill, params.frag_merge
                ),
            )
            result["image"] = self.grain_ids_to_rgb(clean_gm)
            result["image_type"] = "rgb"

        elif mt == "Boundary":
            self._ensure_grains(m, params.gb_angle, params.min_grain)
            gb_arr = np.asarray(m.data["grain_boundaries"]).astype(np.float32)
            result["image"] = gb_arr
            result["image_type"] = "scalar"
            result["scalar_label"] = "Grain Boundary"
            result["vmin"] = 0.0
            result["vmax"] = 1.0

        else:
            raise ValueError(f"Unknown map type: {mt!r}")

        # ----------------------------------------------------------------
        # HAGB contours
        # ----------------------------------------------------------------
        if params.show_hagb and mt != "Boundary":
            self._ensure_grains(m, params.gb_angle, params.min_grain)
            gm_key = (
                path, params.gb_angle, params.min_grain,
                params.hole_fill, params.frag_merge,
            )
            clean_gm = self._cached(
                "clean_gm", gm_key,
                lambda: self.build_clean_grain_map(
                    m, params.hole_fill, params.frag_merge
                ),
            )
            result["hagb_contours"] = self.contours_from_grain_map(
                clean_gm, params.gb_smooth
            )

        # ----------------------------------------------------------------
        # LAGB contours (uses separate _lagb_map to avoid state collision)
        # ----------------------------------------------------------------
        if params.show_lagb and mt != "Boundary":
            lagb_key = (
                path, params.lagb_min, params.min_grain,
                params.gb_angle, params.hole_fill, params.frag_merge,
            )
            sub_gm, main_clean_gm = self._cached(
                "lagb", lagb_key,
                lambda: self._prepare_lagb_data(params),
            )
            result["lagb_contours"] = self.extract_lagb_contours(
                sub_gm, main_clean_gm, params.lagb_smooth
            )

        # ----------------------------------------------------------------
        # Twin boundaries
        # ----------------------------------------------------------------
        if params.show_twins and params.active_twin_defs and mt != "Boundary":
            self._ensure_grains(m, params.gb_angle, params.min_grain)
            gm_key = (
                path, params.gb_angle, params.min_grain,
                params.hole_fill, params.frag_merge,
            )
            clean_gm = self._cached(
                "clean_gm", gm_key,
                lambda: self.build_clean_grain_map(
                    m, params.hole_fill, params.frag_merge
                ),
            )
            twin_key = gm_key + (
                params.twin_tol, params.active_twin_defs, params.strict_axis
            )
            result["twin_data"] = self._cached(
                "twin", twin_key,
                lambda: self.detect_twin_boundaries(m, clean_gm, params),
            )

        logger.debug(
            "EbsdEngine.compute('%s') took %.3fs", mt, time.perf_counter() - t0
        )
        return result


# ====================================================================
# RenderWorker -- background thread dispatcher
# ====================================================================


class RenderWorker(QObject):
    """Runs ``EbsdEngine.compute()`` in a background thread.

    Receives an immutable ``RenderParams`` snapshot and the shared
    ``EbsdEngine`` instance.  Because only one ``RenderWorker`` is
    active at any time (enforced by ``EbsdMainWindow``), there are no
    thread-safety issues with the engine's mutable cache.
    """

    finished: Signal = Signal(object)  # emits result dict
    error: Signal = Signal(str)

    def __init__(self, params: RenderParams, engine: EbsdEngine) -> None:
        super().__init__()
        self._params = params
        self._engine = engine

    def run(self) -> None:
        try:
            result = self._engine.compute(self._params)
            self.finished.emit(result)
        except Exception as exc:
            logger.error("RenderWorker failed: %s", exc, exc_info=True)
            self.error.emit(str(exc))


# ====================================================================
# MplView -- Matplotlib canvas widget
# ====================================================================


class MplView(QWidget):
    """Embeds a Matplotlib figure + navigation toolbar in a QWidget."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
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
# EbsdMainWindow -- pure GUI layer
# ====================================================================


class EbsdMainWindow(QMainWindow):
    """Main application window.

    Responsibilities (exclusively):
    * Build the widget hierarchy.
    * Collect user input and create ``RenderParams`` snapshots.
    * Dispatch ``RenderWorker`` via a ``QThread``.
    * Receive the result dict and render it onto the Matplotlib canvas.
    * Handle canvas click events.

    This class performs no data computation.
    """

    _SPACE_GROUP_TO_STRUCTURE: Dict[int, str] = {
        225: "FCC", 227: "FCC", 229: "BCC", 194: "HCP", 186: "HCP",
    }
    _LAUE_TO_SYSTEM: Dict[int, str] = {9: "Hexagonal", 11: "Cubic"}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EBSD Prototype - PySide6 + DefDAP")
        self.resize(1560, 960)

        self.current_path: Optional[Path] = None
        self.current_fig_ready: bool = False
        self._data_widgets: List[QWidget] = []
        self._is_rendering: bool = False
        self._pending_params: Optional[RenderParams] = None
        self._current_thread: Optional[QThread] = None
        self._engine = EbsdEngine()

        self._build_ui()
        self._connect_signals()
        self._update_control_states(False)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(DEBOUNCE_MS)
        self._refresh_timer.timeout.connect(self._do_refresh)

    # ------------------------------------------------------------------
    # Widget factory helpers
    # ------------------------------------------------------------------

    def _register(self, widget: QWidget) -> QWidget:
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

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------

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

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        vbox = QVBoxLayout(panel)

        # File group
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

        # ---- Tab 1: Map / GB ----
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

        self.spin_gb_angle = self._dspin(1.0, 60.0, 15.0, 0.5, 1, "\u00b0")
        form.addRow("HAGB threshold", self.spin_gb_angle)
        self.spin_min_grain = self._ispin(1, 1000, 10, 1, " px")
        form.addRow("Min grain", self.spin_min_grain)
        self.spin_kam_max = self._dspin(0.1, 20.0, 5.0, 0.5, 1, "\u00b0")
        form.addRow("KAM max", self.spin_kam_max)
        self.spin_misori_max = self._dspin(0.1, 20.0, 5.0, 0.5, 1, "\u00b0")
        form.addRow("Misori max", self.spin_misori_max)

        form.addRow(QLabel("-- Non-indexed / Denoise --"))

        self.combo_noindex_method = self._register(QComboBox())
        self.combo_noindex_method.addItems([
            "fill (neighbor)",
            "fill (median 3\u00d73)",
            "black",
            "white",
            "leave as-is",
        ])
        self.combo_noindex_method.setCurrentText("fill (neighbor)")
        self.combo_noindex_method.setToolTip(
            "fill (neighbor): iteratively fill with nearest valid pixel\n"
            "fill (median 3\u00d73): fill with local median\n"
            "black / white: paint solid\n"
            "leave as-is: no treatment"
        )
        form.addRow("Non-indexed", self.combo_noindex_method)

        self.spin_noindex_iter = self._ispin(1, 50, 5, 1)
        self.spin_noindex_iter.setToolTip(
            "Max iterations for neighbor fill (more = fill larger gaps)")
        form.addRow("Fill iterations", self.spin_noindex_iter)

        self.combo_denoise_method = self._register(QComboBox())
        self.combo_denoise_method.addItems(["None", "Median", "Gaussian"])
        self.combo_denoise_method.setCurrentText("None")  # off by default; user opts in
        self.combo_denoise_method.setToolTip(
            "None: no smoothing\n"
            "Median: rank filter (kernel size below must be >= 3)\n"
            "Gaussian: convolution with Gaussian (sigma below must be > 0)\n"
            "Kernel/sigma defaults are pre-set to sensible values so "
            "denoising takes effect immediately when enabled."
        )
        form.addRow("Denoise", self.combo_denoise_method)

        # Default kernel=3 (not 1): kernel=1 maps to a no-op 1x1 window,
        # and the k < 2 guard in denoise_2d/denoise_rgb would skip filtering.
        self.spin_median_kernel = self._ispin(1, 21, 3, 2)
        form.addRow("Median kernel", self.spin_median_kernel)

        # Default sigma=1.0 (not 0.0): sigma=0 triggers the sigma < 0.01
        # guard in denoise_2d/denoise_rgb, resulting in a no-op.
        self.spin_gauss_sigma = self._dspin(0.0, 10.0, 1.0, 0.1, 1)
        form.addRow("Gauss sigma", self.spin_gauss_sigma)

        form.addRow(QLabel("-- HAGB --"))
        self.combo_gb_color = self._register(QComboBox())
        self.combo_gb_color.addItems(["black", "white", "red", "yellow", "blue"])
        form.addRow("HAGB color", self.combo_gb_color)
        self.spin_gb_width = self._dspin(0.1, 5.0, 1.0, 0.1, 1, " pt")
        self.spin_gb_alpha = self._dspin(0.0, 1.0, 1.0, 0.1, 1)
        self.spin_gb_smooth = self._dspin(0.0, 5.0, 0.6, 0.1, 1)
        form.addRow("HAGB width", self.spin_gb_width)
        form.addRow("HAGB alpha", self.spin_gb_alpha)
        form.addRow("HAGB smooth", self.spin_gb_smooth)
        self.spin_hole_fill = self._ispin(0, 500, 10, 1, " px")
        self.spin_frag_merge = self._ispin(0, 200, 5, 1, " px")
        form.addRow("Fill holes \u2264", self.spin_hole_fill)
        form.addRow("Merge frags \u2264", self.spin_frag_merge)
        self.chk_plot_gbs = self._register(QCheckBox("Show HAGB"))
        self.chk_plot_gbs.setChecked(True)
        form.addRow(self.chk_plot_gbs)

        form.addRow(QLabel("-- LAGB --"))
        self.chk_plot_lagb = self._register(QCheckBox("Show LAGB"))
        self.chk_plot_lagb.setChecked(False)
        form.addRow(self.chk_plot_lagb)
        self.spin_lagb_min = self._dspin(0.5, 15.0, 2.0, 0.5, 1, "\u00b0")
        self.spin_lagb_max = self._dspin(1.0, 60.0, 15.0, 0.5, 1, "\u00b0")
        form.addRow("LAGB min", self.spin_lagb_min)
        form.addRow("LAGB max", self.spin_lagb_max)
        self.combo_lagb_color = self._register(QComboBox())
        self.combo_lagb_color.addItems([
            "red", "blue", "green", "magenta", "cyan",
            "yellow", "white", "black",
        ])
        self.combo_lagb_color.setCurrentText("red")
        form.addRow("LAGB color", self.combo_lagb_color)
        self.spin_lagb_width = self._dspin(0.1, 5.0, 0.5, 0.1, 1, " pt")
        self.spin_lagb_alpha = self._dspin(0.0, 1.0, 0.8, 0.1, 1)
        self.spin_lagb_smooth = self._dspin(0.0, 5.0, 0.4, 0.1, 1)
        form.addRow("LAGB width", self.spin_lagb_width)
        form.addRow("LAGB alpha", self.spin_lagb_alpha)
        form.addRow("LAGB smooth", self.spin_lagb_smooth)
        self.combo_lagb_style = self._register(QComboBox())
        self.combo_lagb_style.addItems(["solid", "dashed", "dotted", "dashdot"])
        self.combo_lagb_style.setCurrentText("dashed")
        form.addRow("LAGB style", self.combo_lagb_style)

        form.addRow(QLabel("-- Display --"))
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

        # ---- Tab 2: Twins ----
        tab2 = QWidget()
        tf = QFormLayout(tab2)
        self.lbl_crystal_sys = QLabel("(load file first)")
        tf.addRow("Crystal system", self.lbl_crystal_sys)
        self.chk_show_twins = self._register(QCheckBox("Show twin boundaries"))
        self.chk_show_twins.setChecked(False)
        tf.addRow(self.chk_show_twins)
        self.spin_twin_tol = self._dspin(1.0, 15.0, 5.0, 0.5, 1, "\u00b0")
        tf.addRow("Twin tolerance", self.spin_twin_tol)
        self.spin_twin_width = self._dspin(0.1, 5.0, 0.5, 0.1, 1, " pt")
        self.spin_twin_alpha = self._dspin(0.0, 1.0, 1.0, 0.1, 1)
        tf.addRow("Twin line width", self.spin_twin_width)
        tf.addRow("Twin line alpha", self.spin_twin_alpha)
        self.chk_strict_axis = self._register(QCheckBox("Strict axis check"))
        self.chk_strict_axis.setChecked(False)
        tf.addRow(self.chk_strict_axis)

        tf.addRow(QLabel("-- Twin systems --"))
        self.twin_checks: List[QCheckBox] = []
        tf.addRow(QLabel("Cubic:"))
        for td in TWIN_SYSTEMS["cubic"]:
            cb = self._register(
                QCheckBox(f'{td["name"]} ({td["angle_deg"]:.1f}\u00b0)')
            )
            cb.setChecked(td["name"] == "\u03a33 {111}<112>")
            cb.setProperty("twin_def", td)
            cb.setProperty("sym", "cubic")
            self.twin_checks.append(cb)
            tf.addRow(cb)
        tf.addRow(QLabel("Hexagonal:"))
        for td in TWIN_SYSTEMS["hexagonal"]:
            cb = self._register(
                QCheckBox(f'{td["name"]} ({td["angle_deg"]:.1f}\u00b0)')
            )
            cb.setChecked(td["name"] == "{10-12} tension")
            cb.setProperty("twin_def", td)
            cb.setProperty("sym", "hexagonal")
            self.twin_checks.append(cb)
            tf.addRow(cb)
        self.lbl_twin_stats = QLabel("-")
        self.lbl_twin_stats.setWordWrap(True)
        tf.addRow("Twin stats", self.lbl_twin_stats)
        self.tabs.addTab(tab2, "Twins")

        # ---- Tab 3: Pole Figure ----
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

        ff.addRow(QLabel("-- Phase details --"))
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

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self.btn_open.clicked.connect(self.open_file)
        self.btn_save_img.clicked.connect(self.save_image)
        self.btn_export_ctf.clicked.connect(self.export_ctf)
        self.btn_refresh.clicked.connect(self.refresh_plot)
        self.btn_plot_pf.clicked.connect(self.plot_pole_figure)
        self.btn_plot_ipf.clicked.connect(self.plot_inverse_pole_figure)
        self.viewer.canvas.mpl_connect("button_press_event", self.on_canvas_click)
        self.spin_gb_angle.valueChanged.connect(self._sync_lagb_max)

        for w in [
            self.combo_map, self.combo_denoise_method,
            self.combo_noindex_method,
            self.combo_gb_color, self.combo_scalar_cmap,
            self.combo_scalebar_loc, self.combo_lagb_color,
            self.combo_lagb_style, self.combo_bc_ipf_mode,
        ]:
            w.currentIndexChanged.connect(self._schedule_refresh)

        for w in [
            self.chk_show_colorbar, self.chk_scalebar,
            self.chk_plot_gbs, self.chk_plot_lagb, self.chk_show_twins,
        ]:
            w.toggled.connect(self._schedule_refresh)

        for w in [
            self.spin_gb_angle, self.spin_min_grain,
            self.spin_kam_max, self.spin_misori_max,
            self.spin_median_kernel, self.spin_gauss_sigma,
            self.spin_scalebar_frac,
            self.spin_gb_width, self.spin_gb_alpha,
            self.spin_gb_smooth, self.spin_hole_fill, self.spin_frag_merge,
            self.spin_lagb_min, self.spin_lagb_max,
            self.spin_lagb_width, self.spin_lagb_alpha, self.spin_lagb_smooth,
            self.spin_twin_tol, self.spin_twin_width, self.spin_twin_alpha,
            self.spin_bc_ipf_alpha, self.spin_bc_brightness,
            self.spin_bc_contrast, self.spin_noindex_iter,
        ]:
            w.valueChanged.connect(self._schedule_refresh)

        for cb in self.twin_checks:
            cb.toggled.connect(self._schedule_refresh)

    # ------------------------------------------------------------------
    # Debounce
    # ------------------------------------------------------------------

    def _schedule_refresh(self) -> None:
        self._refresh_timer.start()

    def _do_refresh(self) -> None:
        self.refresh_plot()

    # ------------------------------------------------------------------
    # Misc UI helpers
    # ------------------------------------------------------------------

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
        try:
            self.lbl_path.setText(str(self.current_path))
            ea = np.asarray(m.data.euler_angle)
            H, W = ea.shape[1], ea.shape[2]
            self.lbl_size.setText(f"{W} x {H}")
            self.lbl_step.setText(f"{m.step_size:.4f} \u00b5m")
            self.lbl_num_phases.setText(str(getattr(m, "num_phases", "-")))
        except Exception:
            logger.warning("Failed to read map dimensions", exc_info=True)
        try:
            phase = m.primary_phase
            sym = m.crystal_sym
            self.lbl_phase_name.setText(phase.name or "-")
            self.lbl_crystal_type.setText(sym.capitalize())
            sg = phase.spaceGroup
            structure = self._SPACE_GROUP_TO_STRUCTURE.get(sg)
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
                    f"a={lp[0]:.4f}  b={lp[1]:.4f}  c={lp[2]:.4f} \u00c5\n"
                    f"\u03b1={np.degrees(lp[3]):.1f}\u00b0  "
                    f"\u03b2={np.degrees(lp[4]):.1f}\u00b0  "
                    f"\u03b3={np.degrees(lp[5]):.1f}\u00b0"
                )
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
                self.lbl_space_group, self.lbl_lattice, self.lbl_c_over_a,
            ]:
                lbl.setText("-")

    def _update_control_states(self, enabled: bool) -> None:
        for w in self._data_widgets:
            w.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Parameter snapshot
    # ------------------------------------------------------------------

    def _snapshot_params(self) -> RenderParams:
        """Collect all UI widget values into an immutable RenderParams."""
        active_twin_list = []
        for cb in self.twin_checks:
            if cb.isChecked():
                td = cb.property("twin_def")
                sym = cb.property("sym")
                axis_t = tuple(td["axis"].tolist())
                active_twin_list.append((
                    td["name"],
                    td["angle_deg"],
                    axis_t,
                    sym,
                    self.spin_twin_tol.value(),
                    td["color"],
                ))
        return RenderParams(
            path=str(self.current_path),
            map_type=self.combo_map.currentText(),
            gb_angle=self.spin_gb_angle.value(),
            min_grain=self.spin_min_grain.value(),
            kam_max=self.spin_kam_max.value(),
            misori_max=self.spin_misori_max.value(),
            noindex_method=self.combo_noindex_method.currentText(),
            noindex_iter=self.spin_noindex_iter.value(),
            denoise_method=self.combo_denoise_method.currentText(),
            median_kernel=self.spin_median_kernel.value(),
            gauss_sigma=self.spin_gauss_sigma.value(),
            bc_brightness=self.spin_bc_brightness.value(),
            bc_contrast=self.spin_bc_contrast.value(),
            bc_ipf_alpha=self.spin_bc_ipf_alpha.value(),
            bc_ipf_mode=self.combo_bc_ipf_mode.currentText(),
            gb_color=self.combo_gb_color.currentText(),
            gb_width=self.spin_gb_width.value(),
            gb_alpha=self.spin_gb_alpha.value(),
            gb_smooth=self.spin_gb_smooth.value(),
            hole_fill=self.spin_hole_fill.value(),
            frag_merge=self.spin_frag_merge.value(),
            show_hagb=self.chk_plot_gbs.isChecked(),
            show_lagb=self.chk_plot_lagb.isChecked(),
            lagb_min=self.spin_lagb_min.value(),
            lagb_max=self.spin_lagb_max.value(),
            lagb_color=self.combo_lagb_color.currentText(),
            lagb_width=self.spin_lagb_width.value(),
            lagb_alpha=self.spin_lagb_alpha.value(),
            lagb_smooth=self.spin_lagb_smooth.value(),
            lagb_style=self.combo_lagb_style.currentText(),
            show_twins=self.chk_show_twins.isChecked(),
            twin_tol=self.spin_twin_tol.value(),
            twin_width=self.spin_twin_width.value(),
            twin_alpha=self.spin_twin_alpha.value(),
            strict_axis=self.chk_strict_axis.isChecked(),
            active_twin_defs=tuple(active_twin_list),
            scalar_cmap=self.combo_scalar_cmap.currentText(),
            show_colorbar=self.chk_show_colorbar.isChecked(),
            show_scalebar=self.chk_scalebar.isChecked(),
            scalebar_frac=self.spin_scalebar_frac.value(),
            scalebar_loc=self.combo_scalebar_loc.currentText(),
        )

    # ------------------------------------------------------------------
    # Worker dispatch
    # ------------------------------------------------------------------

    def _dispatch_worker(self, params: RenderParams) -> None:
        """Create a RenderWorker and move it to a background QThread."""
        self._is_rendering = True
        self.statusBar().showMessage("Rendering...")

        worker = RenderWorker(params, self._engine)
        thread = QThread(self)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_render_finished)
        worker.error.connect(self._on_render_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._current_thread = thread
        thread.start()

    def _on_render_finished(self, result: dict) -> None:
        """Receive result dict from worker and render to matplotlib canvas."""
        self._is_rendering = False
        try:
            self._draw_result(result)
            self.current_fig_ready = True
            self.statusBar().showMessage("Ready")
        except Exception as exc:
            logger.error("Draw failed: %s", exc, exc_info=True)
            self.statusBar().showMessage(f"Draw error: {exc}")
        finally:
            # Process pending request if any
            if self._pending_params is not None:
                pending = self._pending_params
                self._pending_params = None
                self._dispatch_worker(pending)

    def _on_render_error(self, msg: str) -> None:
        self._is_rendering = False
        self.statusBar().showMessage(f"Render error: {msg}")
        QMessageBox.critical(self, "Render error", msg)
        if self._pending_params is not None:
            pending = self._pending_params
            self._pending_params = None
            self._dispatch_worker(pending)

    # ------------------------------------------------------------------
    # Drawing (main thread only -- called by _on_render_finished)
    # ------------------------------------------------------------------

    def _draw_result(self, result: dict) -> None:
        """Render the result dict onto the matplotlib canvas."""
        fig = self.viewer.figure
        fig.clear()
        ax = fig.add_subplot(111)

        image = result["image"]
        image_type = result["image_type"]

        if image_type == "rgb":
            h, w = image.shape[:2]
            ax.imshow(
                image, origin="upper", interpolation="none",
                extent=[-0.5, w - 0.5, h - 0.5, -0.5], aspect="equal",
            )
            self._setup_axes(ax, h, w)

        elif image_type == "scalar":
            if image.ndim == 1:
                image = image[:, np.newaxis]
            h, w = image.shape[:2]
            im = ax.imshow(
                image, origin="upper", interpolation="none",
                extent=[-0.5, w - 0.5, h - 0.5, -0.5], aspect="equal",
                cmap=result["scalar_cmap"],
                vmin=result["vmin"], vmax=result["vmax"],
            )
            self._setup_axes(ax, h, w)
            if result["show_colorbar"] and result["scalar_label"]:
                fig.colorbar(
                    im, ax=ax, fraction=0.046, pad=0.04,
                    label=result["scalar_label"],
                )
        else:
            raise ValueError(f"Unknown image_type: {image_type!r}")

        # HAGB contours
        if result["hagb_contours"]:
            ax.add_collection(LineCollection(
                result["hagb_contours"],
                colors=result["gb_color"],
                linewidths=result["gb_width"],
                alpha=result["gb_alpha"],
                antialiaseds=True,
                capstyle="round", joinstyle="round",
                zorder=10,
            ))

        # LAGB contours
        if result["lagb_contours"]:
            _style_map = {
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
                linestyles=_style_map.get(result["lagb_style"], "dashed"),
                antialiaseds=True,
                capstyle="round", joinstyle="round",
                zorder=11,
            ))

        # Twin boundaries
        twin_data = result.get("twin_data")
        if twin_data:
            for name, data in twin_data.items():
                ax.add_collection(LineCollection(
                    data["contours"],
                    colors=data["color"],
                    linewidths=result["twin_width"],
                    alpha=result["twin_alpha"],
                    antialiaseds=True,
                    capstyle="round", joinstyle="round",
                    zorder=12,
                ))
            stats = [
                f"{name}: {len(data['contours'])}"
                for name, data in twin_data.items()
            ]
            self.lbl_twin_stats.setText(
                "\n".join(stats) if stats else "No twins detected"
            )
        else:
            self.lbl_twin_stats.setText("-")

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

    @staticmethod
    def _setup_axes(ax: Any, sh: int, sw: int) -> None:
        ax.set_xlim(-0.5, sw - 0.5)
        ax.set_ylim(sh - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])

    # ------------------------------------------------------------------
    # Pole figure / IPF windows
    # ------------------------------------------------------------------

    def _get_pf_direction(self) -> np.ndarray:
        txt = self.combo_pf_direction.currentText()
        if "X" in txt:
            return np.array([1, 0, 0])
        if "Y" in txt:
            return np.array([0, 1, 0])
        return np.array([0, 0, 1])

    def plot_pole_figure(self) -> None:
        m = self._engine.main_map
        if m is None:
            QMessageBox.information(self, "No data", "Load a file first.")
            return
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
                f"{pl} PF -- {self.combo_pf_direction.currentText()}")
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
                            np.array([q]), direction, sym)[:, 0]
                    )
                else:
                    cols.append([0, 0, 1])

            c = (
                np.clip(np.array(cols), 0, 1)
                if self.chk_pf_color_ipf.isChecked()
                else "blue"
            )
            ax_pf.scatter(xp, yp, s=ms, c=c, marker=marker,
                          alpha=alpha, edgecolors="none")

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
        m = self._engine.main_map
        if m is None:
            QMessageBox.information(self, "No data", "Load a file first.")
            return
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
            logger.warning("plot_ipf failed; using manual fallback", exc_info=True)
            alpha_f, beta_f = Quat.calc_fund_dirs(
                np.array(quats_s), direction, sym
            )
            ac = np.tan(alpha_f / 2)
            xp = ac * np.cos(beta_f - np.pi / 2)
            yp = ac * np.sin(beta_f - np.pi / 2)
            rgb = Quat.calc_ipf_colours(np.array(quats_s), direction, sym)
            ax_ipf.scatter(
                xp, yp, s=ms,
                c=np.clip(rgb.T, 0, 1),
                marker=marker, alpha=alpha, edgecolors="none",
            )
            ax_ipf.set_aspect("equal")
            ax_ipf.axis("off")
            ax_ipf.set_title(
                f"IPF -- {self.combo_pf_direction.currentText()}")

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

    # ------------------------------------------------------------------
    # Canvas click
    # ------------------------------------------------------------------

    def on_canvas_click(self, event: Any) -> None:
        if self._is_rendering:
            return
        m = self._engine.main_map
        if m is None or event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        try:
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

            pid = -1
            try:
                pm = np.asarray(m.data["phase"])
                pv = pm[grain_mask]
                pv = pv[pv > 0]
                if pv.size:
                    pid = int(np.bincount(pv).argmax())
            except Exception:
                logger.debug("Could not determine phase for grain %d", gid)

            try:
                grain.calc_average_ori()
                ro = getattr(grain, "ref_ori", None)
                rot = str(ro) if ro else "-"
            except Exception:
                logger.debug("Could not compute ref_ori for grain %d", gid)
                rot = "-"

            try:
                grain.build_mis_ori_list()
                am = getattr(grain, "average_mis_ori", None)
                amt = "-" if am is None else f"{float(am):.3f}\u00b0"
            except Exception:
                logger.debug("Could not compute avg misori for grain %d", gid)
                amt = "-"

            self.lbl_grain_id.setText(str(gid))
            self.lbl_grain_pixels.setText(str(px))
            self.lbl_grain_area.setText(f"{area:.2f} \u00b5m\u00b2")
            self.lbl_grain_eqd.setText(f"{eqd:.2f} \u00b5m")
            self.lbl_grain_phase.setText(str(pid))
            self.lbl_grain_avg_mis.setText(amt)
            self.lbl_grain_ref_ori.setText(rot)
            self.lbl_grain_twin.setText("-")
            self.statusBar().showMessage(f"Selected grain {gid}")
        except Exception as exc:
            logger.error("Click handler failed", exc_info=True)
            self.statusBar().showMessage(f"Click failed: {exc}")

    # ------------------------------------------------------------------
    # File actions
    # ------------------------------------------------------------------

    def open_file(self) -> None:
        ps, _ = QFileDialog.getOpenFileName(
            self, "Open CPR file", "", "Oxford CPR files (*.cpr)")
        if not ps:
            return
        p = Path(ps)
        if not p.with_suffix(".crc").exists():
            QMessageBox.warning(
                self, "Missing CRC",
                f"No CRC found:\n{p.with_suffix('.crc')}"
            )
            return
        self.current_path = p
        self.statusBar().showMessage(f"Loading {p.name}...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._engine.load(p)
            m = self._engine.main_map
            self._update_info(m)
            self._update_control_states(True)
            self.statusBar().showMessage(f"Loaded: {p}")
        except Exception as exc:
            logger.error("Failed to load %s", p, exc_info=True)
            QMessageBox.critical(self, "Load error", str(exc))
            self.statusBar().showMessage("Load failed")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._clear_grain_info()
        self.refresh_plot()

    def refresh_plot(self) -> None:
        if self.current_path is None or self._engine.main_map is None:
            return
        params = self._snapshot_params()
        if self._is_rendering:
            self._pending_params = params
            return
        self._dispatch_worker(params)

    def save_image(self) -> None:
        if not self.current_fig_ready:
            QMessageBox.information(self, "No figure", "Render a map first.")
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
            QMessageBox.critical(self, "Save error", str(exc))

    def export_ctf(self) -> None:
        if self.current_path is None:
            QMessageBox.information(self, "No file", "Load first.")
            return
        ps, _ = QFileDialog.getSaveFileName(
            self, "Export CTF",
            str(self.current_path.with_suffix(".ctf")),
            "Oxford Text (*.ctf)",
        )
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
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.statusBar().showMessage("Exporting CTF...")
            m = self._engine.main_map
            m.save(
                file_name=out.stem,
                data_type="OxfordText",
                file_dir=str(out.parent),
            )
            self.statusBar().showMessage(f"Exported: {out}")
        except Exception as exc:
            logger.error("CTF export failed", exc_info=True)
            QMessageBox.critical(self, "Export error", str(exc))
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
    sys.exit(main())
