"""
Fast pointing refinement by robust least squares on matched star pairs.

This is an ICP-style alternative to fitting many random asterisms and taking the median of each parameter:

1. project the catalog with the current pointing
2. match catalog stars to observed stars (mutual nearest neighbours within a radius)
3. fit crval1, crval2 and crota (optionally the platescale and the AZP mu) with those matches held fixed,
   returning one residual per star per axis and a robust (soft-L1) loss
4. shrink the match radius and repeat

Inside the fit, catalog stars are projected with a vectorised NumPy implementation of the FITS AZP projection
(Calabretta & Greisen 2002), so each residual evaluation costs microseconds rather than an astropy
``world_to_pixel_values`` call. ``check_projection_against_astropy`` compares it to astropy for a given WCS.

Conventions follow ``solve_pointing``:

* observed star positions are 0-based pixels that have already had the distortion (cpdis) offsets applied;
  the fit uses only the linear WCS and the AZP projection
* the PC matrix is a pure rotation, ``[[cos, -sin], [sin, cos]]``, as in ``refine_pointing_single_step``
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from astropy.wcs import WCS
from scipy.optimize import least_squares
from scipy.spatial import KDTree

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


def crota_to_pc(crota_deg: float) -> np.ndarray:
    """Build the PC matrix used throughout the alignment code from a rotation angle in degrees."""
    c, s = np.cos(crota_deg * D2R), np.sin(crota_deg * D2R)
    return np.array([[c, -s], [s, c]])


def pc_to_crota(pc: np.ndarray) -> float:
    """Recover the rotation angle in degrees from a PC matrix built by ``crota_to_pc``."""
    return float(np.arctan2(pc[1, 0], pc[0, 0]) * R2D)


@dataclass
class AZPModel:
    """Minimal state needed to map (RA, Dec) to 0-based pixel coordinates for an AZP WCS."""

    crpix: np.ndarray  # 1-based, as stored in the WCS
    cdelt: np.ndarray  # deg / pixel
    crval: np.ndarray  # deg
    crota: float  # deg
    mu: float = 0.0  # PV2_1
    gamma: float = 0.0  # PV2_2, deg
    lonpole: float = 180.0  # deg

    def pc(self) -> np.ndarray:
        """Return the PC matrix for the current rotation."""
        return crota_to_pc(self.crota)

    def world_to_pixel(self,
                       ra_deg: np.ndarray,
                       dec_deg: np.ndarray,
                       crval: np.ndarray | None = None,
                       crota: float | None = None,
                       scale: float = 1.0,
                       mu: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Project sky coordinates to 0-based pixel coordinates.

        Parameters
        ----------
        ra_deg, dec_deg : np.ndarray
            sky coordinates in degrees
        crval, crota, scale, mu
            optional overrides of the model state, used while fitting

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            x and y pixel coordinates; NaN where the AZP projection is undefined

        """
        crval = self.crval if crval is None else crval
        crota = self.crota if crota is None else crota
        mu = self.mu if mu is None else mu
        cdelt = self.cdelt * scale

        a = np.asarray(ra_deg) * D2R
        d = np.asarray(dec_deg) * D2R
        ap, dp, php = crval[0] * D2R, crval[1] * D2R, self.lonpole * D2R

        sd, cd = np.sin(d), np.cos(d)
        sdp, cdp = np.sin(dp), np.cos(dp)
        da = a - ap
        cda = np.cos(da)

        # celestial -> native spherical; for zenithal projections the native pole is the reference point
        sin_t = sd * sdp + cd * cdp * cda
        cos_t = np.sqrt(np.clip(1.0 - sin_t * sin_t, 0.0, None))
        phi = php + np.arctan2(-cd * np.sin(da), sd * cdp - cd * sdp * cda)

        # native spherical -> intermediate world coordinates (AZP)
        g = self.gamma * D2R
        sphi, cphi = np.sin(phi), np.cos(phi)
        denom = mu + sin_t + cos_t * cphi * np.tan(g)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = R2D * (mu + 1.0) * cos_t / denom
        r = np.where(denom > 0, r, np.nan)
        x = r * sphi
        y = -r * cphi / np.cos(g)

        # intermediate world (deg) -> pixel:  [x, y] = diag(cdelt) @ PC @ (p - crpix)
        minv = np.linalg.inv(np.diag(cdelt) @ crota_to_pc(crota))
        px = minv[0, 0] * x + minv[0, 1] * y + self.crpix[0] - 1.0
        py = minv[1, 0] * x + minv[1, 1] * y + self.crpix[1] - 1.0
        return px, py


def azp_model_from_wcs(wcs: WCS) -> AZPModel:
    """Read a celestial AZP WCS in PC form into an ``AZPModel``."""
    w = wcs.wcs
    ctype = [str(c) for c in w.ctype]
    if not (ctype[0].endswith("-AZP") and ctype[1].endswith("-AZP")):
        msg = f"Fast pointing refinement only implements the AZP projection, got {ctype}."
        raise ValueError(msg)
    if w.has_cd():
        msg = "WCS uses a CD matrix; convert it with convert_cd_matrix_to_pc_matrix first."
        raise ValueError(msg)

    pc = np.array(w.pc, dtype=float)
    crota = pc_to_crota(pc)
    if not np.allclose(pc, crota_to_pc(crota), atol=1e-9):
        warnings.warn("PC matrix is not a pure rotation; the non-rotation part will be discarded.", stacklevel=2)

    mu, gamma = 0.0, 0.0
    for i, m, value in w.get_pv():
        if i == 2 and m == 1:
            mu = float(value)
        elif i == 2 and m == 2:
            gamma = float(value)

    lonpole = float(w.lonpole)
    if not np.isfinite(lonpole):
        lonpole = 180.0

    return AZPModel(crpix=np.array(w.crpix, dtype=float),
                    cdelt=np.array(w.cdelt, dtype=float),
                    crval=np.array(w.crval, dtype=float),
                    crota=crota,
                    mu=mu,
                    gamma=gamma,
                    lonpole=lonpole)


def write_model_to_wcs(model: AZPModel, wcs: WCS) -> WCS:
    """Return a deep copy of ``wcs`` carrying the pointing in ``model``."""
    out = wcs.deepcopy()
    out.wcs.crval = model.crval
    out.wcs.cdelt = model.cdelt
    out.wcs.pc = model.pc()
    pv = [(i, m, v) for (i, m, v) in out.wcs.get_pv() if not (i == 2 and m == 1)]
    pv.append((2, 1, model.mu))
    out.wcs.set_pv(pv)
    out.wcs.set()
    return out


def check_projection_against_astropy(wcs: WCS, ra_deg: np.ndarray, dec_deg: np.ndarray) -> float:
    """
    Return the largest pixel difference between the fast projection and astropy.

    Use a WCS without distortion lookup tables attached; the fast projection does not apply them.
    """
    model = azp_model_from_wcs(wcs)
    fx, fy = model.world_to_pixel(np.asarray(ra_deg), np.asarray(dec_deg))
    ax, ay = wcs.world_to_pixel_values(ra_deg, dec_deg)
    ok = np.isfinite(fx) & np.isfinite(ax)
    return float(np.max(np.hypot(fx[ok] - ax[ok], fy[ok] - ay[ok])))


def match_mutual(catalog_xy: np.ndarray, observed_tree: KDTree, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Match projected catalog stars to observed stars, keeping only mutual nearest neighbours within ``radius``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        indices into the catalog and into the observed stars for each matched pair

    """
    good = np.flatnonzero(np.all(np.isfinite(catalog_xy), axis=1))
    if good.size == 0:
        return good, good
    distances, obs_idx = observed_tree.query(catalog_xy[good], distance_upper_bound=radius)
    hit = np.isfinite(distances)
    cat_idx, obs_idx = good[hit], obs_idx[hit]
    if cat_idx.size == 0:
        return cat_idx, obs_idx
    # keep a pair only if the catalog star is also the observed star's nearest catalog star
    _, back = KDTree(catalog_xy[good]).query(observed_tree.data[obs_idx])
    mutual = good[back] == cat_idx
    return cat_idx[mutual], obs_idx[mutual]


@dataclass
class RefinementInfo:
    """Diagnostics from ``refine_pointing_matched``."""

    n_matched: int = 0
    rms_px: float = np.nan  # RMS of the final residuals, both axes pooled
    median_abs_px: float = np.nan
    sigma: dict = field(default_factory=dict)  # 1-sigma parameter uncertainties (degrees, or unitless)
    history: list = field(default_factory=list)  # (radius, n_matched, rms) for each fit


def refine_pointing_matched(
        guess_wcs: WCS,
        catalog_ra: np.ndarray,
        catalog_dec: np.ndarray,
        observed_xy: np.ndarray,
        radii: tuple[float, ...] = (30.0, 10.0, 4.0, 2.0),
        fit_scale: bool = False,
        fit_mu: bool = False,
        f_scale_px: float | None = None,
        min_matches: int = 6,
        max_rematch: int = 3) -> tuple[WCS, RefinementInfo]:
    """
    Refine the pointing of a WCS using robust least squares on matched star pairs.

    Parameters
    ----------
    guess_wcs : WCS
        starting celestial WCS: AZP projection, PC form, no distortion tables
    catalog_ra, catalog_dec : np.ndarray
        catalog star coordinates in degrees, already restricted to the usable part of the image
    observed_xy : np.ndarray
        (N, 2) array of 0-based observed star positions with distortion offsets already applied
    radii : tuple[float, ...]
        match radii in pixels for successive rounds of matching and fitting
    fit_scale : bool
        if True, also fit a common platescale factor
    fit_mu : bool
        if True, also fit the AZP mu parameter (PV2_1)
    f_scale_px : float | None
        soft-L1 transition scale in pixels; defaults to a quarter of the match radius, at least 0.3 px
    min_matches : int
        the fewest matched stars a fit may use
    max_rematch : int
        the most times to rematch and refit at a single radius before moving to the next

    Returns
    -------
    tuple[WCS, RefinementInfo]
        a deep copy of ``guess_wcs`` with the refined pointing, and fit diagnostics

    """
    catalog_ra = np.asarray(catalog_ra, dtype=float)
    catalog_dec = np.asarray(catalog_dec, dtype=float)
    observed_xy = np.asarray(observed_xy, dtype=float)
    observed_tree = KDTree(observed_xy)
    model = azp_model_from_wcs(guess_wcs)

    names = ["crval1", "crval2", "crota"] + (["scale"] if fit_scale else []) + (["mu"] if fit_mu else [])
    p = np.array([model.crval[0], model.crval[1], model.crota]
                 + ([1.0] if fit_scale else []) + ([model.mu] if fit_mu else []))

    def unpack(q: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        k = 3
        scale = q[k] if fit_scale else 1.0
        k += int(fit_scale)
        mu = q[k] if fit_mu else model.mu
        return np.array([q[0], q[1]]), q[2], scale, mu

    def project(q: np.ndarray, ra: np.ndarray, dec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        crval, crota, scale, mu = unpack(q)
        return model.world_to_pixel(ra, dec, crval=crval, crota=crota, scale=scale, mu=mu)

    def residuals(q: np.ndarray, ra: np.ndarray, dec: np.ndarray, ox: np.ndarray, oy: np.ndarray) -> np.ndarray:
        px, py = project(q, ra, dec)
        # a star pushed past the AZP horizon gets a large but finite residual
        return np.nan_to_num(np.concatenate([px - ox, py - oy]), nan=1e3)

    # parameter scales: roughly one pixel's worth of change in each parameter
    pix_deg = abs(model.cdelt[0])
    cos_dec = max(np.cos(model.crval[1] * D2R), 1e-3)
    field_radius = 0.5 * np.hypot(*model.crpix)
    x_scale = ([pix_deg / cos_dec, pix_deg, R2D / field_radius]
               + ([1.0 / field_radius] if fit_scale else []) + ([0.01] if fit_mu else []))
    needed = max(min_matches, len(p) + 1)

    info = RefinementInfo()
    solution = None
    for radius in radii:
        n_matched, previous = 0, None
        for _ in range(max_rematch):
            px, py = project(p, catalog_ra, catalog_dec)
            cat_idx, obs_idx = match_mutual(np.column_stack([px, py]), observed_tree, radius)
            n_matched = cat_idx.size
            if n_matched < needed:
                break
            key = (cat_idx.tobytes(), obs_idx.tobytes())
            if key == previous:
                break
            previous = key
            f_scale = f_scale_px if f_scale_px is not None else max(radius / 4.0, 0.3)
            solution = least_squares(residuals, p, x_scale=x_scale, loss="soft_l1", f_scale=f_scale,
                                     args=(catalog_ra[cat_idx], catalog_dec[cat_idx],
                                           observed_xy[obs_idx, 0], observed_xy[obs_idx, 1]))
            p = solution.x
            info.history.append((radius, int(n_matched), float(np.sqrt(np.mean(solution.fun ** 2)))))
        if n_matched < needed:
            warnings.warn(f"Only {n_matched} star matches at radius {radius} px; stopping refinement.",
                          stacklevel=2)
            break

    if solution is None:
        msg = "Pointing refinement found too few star matches."
        raise RuntimeError(msg)

    # 1-sigma uncertainties from the final Jacobian (Gauss-Newton approximation)
    r = solution.fun
    dof = max(r.size - p.size, 1)
    covariance = np.linalg.pinv(solution.jac.T @ solution.jac) * np.sum(r * r) / dof
    info.n_matched = r.size // 2
    info.rms_px = float(np.sqrt(np.mean(r * r)))
    info.median_abs_px = float(np.median(np.abs(r)))
    info.sigma = dict(zip(names, np.sqrt(np.clip(np.diag(covariance), 0, None)), strict=True))

    crval, crota, scale, mu = unpack(p)
    crval[0] %= 360.0
    model.crval = crval
    model.crota = float((crota + 180.0) % 360.0 - 180.0)
    model.cdelt = model.cdelt * scale
    model.mu = float(mu)
    return write_model_to_wcs(model, guess_wcs), info
