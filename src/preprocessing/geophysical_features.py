import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None

try:
    import harmonica as hm
except ImportError:  # pragma: no cover
    hm = None

try:
    from skimage.filters.rank import entropy as _rank_entropy
    from skimage.morphology import square as _morphology_square
    _SKIMAGE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKIMAGE_AVAILABLE = False


def _as_array(grid):
    """Return a numpy array from ndarray/xarray.DataArray inputs."""
    if xr is not None and isinstance(grid, xr.DataArray):
        return grid.values
    return np.asarray(grid)


def _get_field_derivatives(field, dx=1.0, dy=1.0, use_harmonica=False):
    """
    Return (gx, gy, gz) for a potential field.

    gz is the upward derivative; for finite-difference mode it is approximated
    via the Laplacian identity for harmonic fields: gz ≈ -(d²F/dx² + d²F/dy²).
    """
    if use_harmonica and hm is not None and xr is not None and isinstance(field, xr.DataArray):
        gx = hm.derivative_easting(field).values
        gy = hm.derivative_northing(field).values
        gz = hm.derivative_upward(field).values
    else:
        field_arr = _as_array(field).astype(float)
        gy, gx = np.gradient(field_arr, dy, dx)
        dyy, _  = np.gradient(gy, dy, dx)
        _, dxx  = np.gradient(gx, dy, dx)
        gz = -(dxx + dyy)
    return gx, gy, gz


# ---------------------------------------------------------------------------
# DEM-derived attributes
# ---------------------------------------------------------------------------

def compute_dem_derivatives(dem, dx=1.0, dy=1.0):
    """
    Compute DEM-derived attributes used in structural mapping.

    Returns
    -------
    dict
        grad_x, grad_y, slope_rad, slope_deg, curvature
    """
    dem_arr = _as_array(dem).astype(float)
    grad_y, grad_x = np.gradient(dem_arr, dy, dx)

    slope_rad = np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2))
    slope_deg = np.degrees(slope_rad)

    dyy, _ = np.gradient(grad_y, dy, dx)
    _, dxx = np.gradient(grad_x, dy, dx)
    curvature = dxx + dyy

    return {
        "dem_grad_x": grad_x,
        "dem_grad_y": grad_y,
        "dem_slope_rad": slope_rad,
        "dem_slope_deg": slope_deg,
        "dem_curvature": curvature,
    }


def compute_dem_curvatures(dem, dx=1.0, dy=1.0):
    """
    Compute higher-order curvature attributes of a DEM.

    Returns Gaussian, mean, profile (along-slope), and plan (across-slope)
    curvatures. These complement the simple Laplacian curvature for detecting
    structural boundaries and lineaments.

    Returns
    -------
    dict
        dem_gaussian_curvature, dem_mean_curvature,
        dem_profile_curvature, dem_plan_curvature
    """
    dem_arr = _as_array(dem).astype(float)

    fy, fx    = np.gradient(dem_arr, dy, dx)
    dyx, dxx  = np.gradient(fx, dy, dx)   # dxx = d²z/dx²,  dyx = d²z/dydx
    dyy, _    = np.gradient(fy, dy, dx)   # dyy = d²z/dy²

    p, q  = fx, fy
    r, s, t = dxx, dyx, dyy
    denom   = 1.0 + p ** 2 + q ** 2
    eps     = 1e-12

    gaussian_curvature = (r * t - s ** 2) / (denom ** 2 + eps)

    # Evans (1998) sign convention: positive = concave (bowl-shaped)
    mean_curvature = -((1 + q ** 2) * r - 2 * p * q * s + (1 + p ** 2) * t) / (
        2 * (denom + eps) ** 1.5
    )

    slope_sq = p ** 2 + q ** 2
    profile_curvature = -(
        p ** 2 * r + 2 * p * q * s + q ** 2 * t
    ) / ((slope_sq + eps) * (denom + eps) ** 1.5)

    plan_curvature = -(
        q ** 2 * r - 2 * p * q * s + p ** 2 * t
    ) / ((slope_sq + eps) * np.sqrt(denom + eps))

    return {
        "dem_gaussian_curvature": gaussian_curvature,
        "dem_mean_curvature":     mean_curvature,
        "dem_profile_curvature":  profile_curvature,
        "dem_plan_curvature":     plan_curvature,
    }


# ---------------------------------------------------------------------------
# Potential-field attributes
# ---------------------------------------------------------------------------

def separate_regional_residual(field, sigma=12.0):
    """
    Split potential-field data into regional and residual components.

    Parameters
    ----------
    field : ndarray
        2D gravity/magnetic grid.
    sigma : float
        Gaussian smoothing sigma in grid cells.
    """
    field_arr = _as_array(field).astype(float)
    regional = gaussian_filter(field_arr, sigma=sigma)
    residual = field_arr - regional
    return regional, residual


def compute_tga_tilt(field, dx=1.0, dy=1.0, use_harmonica=False):
    """
    Compute Total Gradient Amplitude (TGA) and tilt angle.

    If ``use_harmonica`` is True and ``field`` is an xarray.DataArray,
    horizontal and upward derivatives are computed with Harmonica.
    Otherwise, finite-difference proxies are used.
    """
    gx, gy, gz = _get_field_derivatives(field, dx=dx, dy=dy, use_harmonica=use_harmonica)
    tga  = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    tilt = np.arctan2(gz, np.sqrt(gx ** 2 + gy ** 2) + 1e-12)
    return tga, tilt


def compute_analytic_signal_features(field, dx=1.0, dy=1.0, use_harmonica=False):
    """
    Compute analytic signal amplitude, HGM, theta map, and second vertical derivative.

    These complement TGA and tilt angle for edge/contact detection in potential fields:

    - **Analytic signal** (AS): sqrt(gx² + gy² + gz²) — amplitude maximum directly
      over contacts, independent of magnetization direction for magnetic data.
    - **HGM** (Horizontal Gradient Magnitude): sqrt(gx² + gy²) — highlights lateral
      density or susceptibility contrasts (edges).
    - **Theta map**: tilt angle normalized to [-1, 1] by dividing by π/2 — suppresses
      amplitude variation across different source depths.
    - **SVD** (Second Vertical Derivative): -(d²F/dx² + d²F/dy²) via Laplace identity
      — enhances shallow, local anomalies relative to deep regional signal.

    Returns
    -------
    dict
        analytic_signal, hgm, theta_map, svd
    """
    gx, gy, gz = _get_field_derivatives(field, dx=dx, dy=dy, use_harmonica=use_harmonica)

    hgm             = np.sqrt(gx ** 2 + gy ** 2)
    analytic_signal = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    theta_map       = np.arctan2(gz, hgm + 1e-12) / (np.pi / 2)
    svd             = gz  # Laplace identity: d²F/dz² = -(d²F/dx² + d²F/dy²) = gz proxy

    return {
        "analytic_signal": analytic_signal,
        "hgm":             hgm,
        "theta_map":       theta_map,
        "svd":             svd,
    }


def reduction_to_pole(magnetic_grid, inclination, declination,
                      magnetization_inclination=None, magnetization_declination=None):
    """
    Apply magnetic Reduction-To-The-Pole (RTP).

    Requires harmonica and xarray.DataArray input.
    """
    if hm is None or xr is None:
        raise ImportError("harmonica and xarray are required for reduction_to_pole")
    if not isinstance(magnetic_grid, xr.DataArray):
        raise TypeError("magnetic_grid must be an xarray.DataArray for RTP")

    return hm.reduction_to_pole(
        magnetic_grid,
        inclination=inclination,
        declination=declination,
        magnetization_inclination=magnetization_inclination,
        magnetization_declination=magnetization_declination,
    )


# ---------------------------------------------------------------------------
# Textural features (SAR / optical / geophysical)
# ---------------------------------------------------------------------------

def compute_glcm_spatial_features(field, window_size=15, levels=64):
    """
    Compute spatial textural features via a sliding-window approach.

    - **texture_contrast** (local variance): measures tonal variation within the
      neighbourhood — high along lineament edges.
    - **texture_entropy**: information content of local intensity distribution;
      requires scikit-image for the efficient rank-filter implementation. Falls back
      to a Gaussian-entropy approximation when scikit-image is unavailable.

    Parameters
    ----------
    field : ndarray (H, W)
        Input grid (any potential field, DEM, SAR backscatter, etc.).
    window_size : int
        Side length of the square analysis window (should be odd).
    levels : int
        Number of grey levels for histogram quantisation (entropy only).

    Returns
    -------
    dict
        texture_contrast, texture_entropy
    """
    field_arr   = _as_array(field).astype(float)
    ws          = window_size

    local_mean   = uniform_filter(field_arr,        size=ws)
    local_sq     = uniform_filter(field_arr ** 2,   size=ws)
    local_var    = np.maximum(local_sq - local_mean ** 2, 0.0)

    if _SKIMAGE_AVAILABLE:
        fmin, fmax = field_arr.min(), field_arr.max()
        if fmax > fmin:
            quantized = (
                (field_arr - fmin) / (fmax - fmin) * (levels - 1)
            ).astype(np.uint8)
        else:
            quantized = np.zeros_like(field_arr, dtype=np.uint8)
        local_entropy = _rank_entropy(quantized, _morphology_square(ws)).astype(float)
    else:
        # Gaussian-entropy approximation: H = 0.5 * log2(2πe·σ²)
        eps           = 1e-12
        local_entropy = 0.5 * np.log2(2 * np.pi * np.e * (local_var + eps))

    return {
        "texture_contrast": local_var,
        "texture_entropy":  local_entropy,
    }


# ---------------------------------------------------------------------------
# Directional and orientation features
# ---------------------------------------------------------------------------

def compute_directional_derivatives(field, dx=1.0, dy=1.0, azimuths=(45, 135)):
    """
    Directional first derivatives at specified geographic azimuths.

    For azimuth θ measured clockwise from North:
        d_θ = gx · sin(θ) + gy · cos(θ)

    The diagonal azimuths (45° NE-SW and 135° NW-SE) are the most useful
    additions since gx (90°) and gy (0°) are already available from
    ``compute_dem_derivatives`` / ``compute_analytic_signal_features``.
    NE-SW and NW-SE correspond to the dominant fault orientations in many
    orogenic belts.

    Parameters
    ----------
    field : ndarray (H, W)
        Any potential field or DEM grid.
    azimuths : tuple[float]
        Geographic azimuths in degrees, clockwise from North.
        0° = N→S, 90° = E→W, 45° = NE→SW, 135° = NW→SE.

    Returns
    -------
    dict
        Keys formatted as ``dir_000``, ``dir_045``, etc.
    """
    gx, gy, _ = _get_field_derivatives(field, dx=dx, dy=dy, use_harmonica=False)
    result = {}
    for az in azimuths:
        theta = np.radians(az)
        result[f"dir_{int(az):03d}"] = gx * np.sin(theta) + gy * np.cos(theta)
    return result


def compute_gabor_features(
    field,
    frequencies=(0.05, 0.1, 0.2),
    thetas=(0, 45, 90, 135),
):
    """
    Gabor filter bank for orientation-selective lineament detection.

    A Gabor filter at orientation θ responds strongly to linear structures
    running **perpendicular** to θ. Running the bank at multiple orientations
    and frequencies produces a rich descriptor of lineament strength and
    dominant structural direction.

    Requires scikit-image (``skimage.filters.gabor``).

    Parameters
    ----------
    field : ndarray (H, W)
        Input grid (any potential field, DEM, SAR, etc.).
    frequencies : tuple[float]
        Spatial frequencies in cycles/pixel. Lower values target broader,
        regional structures; higher values target narrow lineaments.
        Typical range for geophysical grids: 0.03–0.25.
    thetas : tuple[float]
        Filter orientations in degrees.  0° = horizontal structures (E-W
        lineaments), 90° = vertical structures (N-S lineaments).

    Returns
    -------
    dict
        ``gabor_{theta:03d}_{freq:.3f}`` — energy map per (theta, freq) pair.
        ``gabor_max_energy`` — pixel-wise max energy over all orientations
        and frequencies (overall lineament strength).
        ``gabor_dominant_angle`` — dominant orientation index normalised to
        [0, 1]; useful as a single-channel orientation summary.
    """
    if not _SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image is required for compute_gabor_features")
    from skimage.filters import gabor as skgabor  # noqa: PLC0415

    field_arr = _as_array(field).astype(float)
    std = field_arr.std()
    field_norm = (field_arr - field_arr.mean()) / std if std > 0 else field_arr

    energy_maps = []
    result = {}

    for theta_deg in thetas:
        theta_rad = np.radians(theta_deg)
        for freq in frequencies:
            real, imag = skgabor(field_norm, frequency=freq, theta=theta_rad)
            energy = real ** 2 + imag ** 2
            result[f"gabor_{int(theta_deg):03d}_{freq:.3f}"] = energy
            energy_maps.append(energy)

    if energy_maps:
        stack_e = np.stack(energy_maps, axis=0)
        result["gabor_max_energy"] = stack_e.max(axis=0)

        # Per-theta mean energy (averaged over frequencies)
        n_freq = len(frequencies)
        theta_mean = np.stack([
            np.mean(energy_maps[ti * n_freq:(ti + 1) * n_freq], axis=0)
            for ti in range(len(thetas))
        ], axis=0)
        dominant_idx = theta_mean.argmax(axis=0).astype(float)
        result["gabor_dominant_angle"] = dominant_idx / max(len(thetas) - 1, 1)

    return result


# ---------------------------------------------------------------------------
# Multi-source feature stack builder
# ---------------------------------------------------------------------------

def build_multisource_feature_stack(
    dem,
    gravity,
    magnetic,
    dx=1.0,
    dy=1.0,
    regional_sigma=12.0,
    use_harmonica_derivatives=False,
    include_analytic_features=True,
    include_dem_curvatures=True,
    include_directional=True,
    include_gabor=False,
    include_texture=False,
    directional_azimuths=(45, 135),
    gabor_frequencies=(0.05, 0.1, 0.2),
    gabor_thetas=(0, 45, 90, 135),
    texture_window_size=15,
    texture_levels=64,
):
    """
    Build an aligned multichannel stack from DEM, gravity and magnetic grids.

    Parameters
    ----------
    include_analytic_features : bool
        Adds analytic signal, HGM, theta map and SVD for gravity and magnetic.
    include_dem_curvatures : bool
        Adds Gaussian, mean, profile and plan curvatures for the DEM.
    include_directional : bool
        Adds directional first derivatives at ``directional_azimuths`` for
        gravity and magnetic.  The diagonal directions (45° NE-SW, 135° NW-SE)
        capture structural trends not covered by the already-present gx/gy.
    include_gabor : bool
        Adds a Gabor filter bank for orientation-selective lineament detection.
        Expensive on large grids; disabled by default.  Requires scikit-image.
    include_texture : bool
        Adds sliding-window texture features (local variance and entropy).
        Requires scikit-image for entropy; falls back to Gaussian approximation.
    directional_azimuths : tuple[float]
        Geographic azimuths (°, clockwise from N) for directional derivatives.
    gabor_frequencies : tuple[float]
        Spatial frequencies (cycles/pixel) for the Gabor bank.
    gabor_thetas : tuple[float]
        Orientations (°) for the Gabor bank.
    texture_window_size : int
        Sliding-window size for texture features (odd recommended).
    texture_levels : int
        Quantisation levels for the entropy texture feature.

    Returns
    -------
    stack : ndarray  (H, W, C)
    channel_names : list[str]
    feature_dict : dict[str, ndarray]

    Notes
    -----
    To apply multiscale pyramids to a subset of features after this call::

        from src.preprocessing.multiscale import append_multiscale_channels
        feature_dict = append_multiscale_channels(
            feature_dict,
            target_keys=["dem_slope_deg", "dem_curvature",
                         "gravity_tga", "magnetic_analytic_signal",
                         "gravity_hgm", "dem_profile_curvature"],
            scales=(1, 2, 4),
        )
    """
    dem_arr  = _as_array(dem).astype(float)
    grav_arr = _as_array(gravity).astype(float)
    mag_arr  = _as_array(magnetic).astype(float)

    if dem_arr.shape != grav_arr.shape or dem_arr.shape != mag_arr.shape:
        raise ValueError("dem, gravity and magnetic grids must share the same shape")

    # --- DEM base features ---
    dem_features = compute_dem_derivatives(dem_arr, dx=dx, dy=dy)

    # --- Regional / residual separation ---
    grav_regional, grav_residual = separate_regional_residual(grav_arr, sigma=regional_sigma)
    mag_regional,  mag_residual  = separate_regional_residual(mag_arr,  sigma=regional_sigma)

    # --- TGA + tilt ---
    grav_tga, grav_tilt = compute_tga_tilt(
        gravity, dx=dx, dy=dy, use_harmonica=use_harmonica_derivatives,
    )
    mag_tga, mag_tilt = compute_tga_tilt(
        magnetic, dx=dx, dy=dy, use_harmonica=use_harmonica_derivatives,
    )

    feature_dict = {
        "dem": dem_arr,
        **dem_features,
        "gravity":          grav_arr,
        "gravity_regional": grav_regional,
        "gravity_residual": grav_residual,
        "gravity_tga":      grav_tga,
        "gravity_tilt":     grav_tilt,
        "magnetic":         mag_arr,
        "magnetic_regional": mag_regional,
        "magnetic_residual": mag_residual,
        "magnetic_tga":     mag_tga,
        "magnetic_tilt":    mag_tilt,
    }

    # --- Optional: analytic signal features ---
    if include_analytic_features:
        grav_as = compute_analytic_signal_features(
            gravity, dx=dx, dy=dy, use_harmonica=use_harmonica_derivatives,
        )
        mag_as = compute_analytic_signal_features(
            magnetic, dx=dx, dy=dy, use_harmonica=use_harmonica_derivatives,
        )
        for key, val in grav_as.items():
            feature_dict[f"gravity_{key}"] = val
        for key, val in mag_as.items():
            feature_dict[f"magnetic_{key}"] = val

    # --- Optional: higher-order DEM curvatures ---
    if include_dem_curvatures:
        feature_dict.update(compute_dem_curvatures(dem_arr, dx=dx, dy=dy))

    # --- Optional: directional derivatives ---
    if include_directional:
        for src_name, src_field in (("gravity", gravity), ("magnetic", magnetic)):
            dirs = compute_directional_derivatives(
                src_field, dx=dx, dy=dy, azimuths=directional_azimuths,
            )
            for key, val in dirs.items():
                feature_dict[f"{src_name}_{key}"] = val

    # --- Optional: Gabor filter bank ---
    if include_gabor:
        for src_name, src_field in (("gravity", gravity), ("magnetic", magnetic), ("dem", dem)):
            gab = compute_gabor_features(
                src_field,
                frequencies=gabor_frequencies,
                thetas=gabor_thetas,
            )
            for key, val in gab.items():
                feature_dict[f"{src_name}_{key}"] = val

    # --- Optional: spatial texture ---
    if include_texture:
        for src_name, src_field in (("gravity", grav_arr), ("magnetic", mag_arr)):
            tex = compute_glcm_spatial_features(
                src_field, window_size=texture_window_size, levels=texture_levels,
            )
            for key, val in tex.items():
                feature_dict[f"{src_name}_{key}"] = val

    channel_names = list(feature_dict.keys())
    stack = np.stack([feature_dict[name] for name in channel_names], axis=-1)
    return stack, channel_names, feature_dict
