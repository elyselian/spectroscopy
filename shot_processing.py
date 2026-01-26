# Python equivalent of the provided MATLAB script
# Requires: numpy, scipy, spe2py (or your spe loader), scikit-image, matplotlib
import numpy as np
from pathlib import Path
import scipy.io
from scipy.interpolate import griddata
import spe2py as spe         # or your SPE loader (sl)
from skimage import exposure, filters, morphology
from skimage.filters import threshold_local

# ...existing code...
def mat2gray(img):
    """Normalize array to [0,1] like MATLAB mat2gray."""
    mn = img.min()
    mx = img.max()
    if mx == mn:
        return (img - mn) * 0.0
    return (img - mn) / (mx - mn)

def make_active_mask(local_shot, block_size=35, sensitivity=0.3, min_size=70):
    """
    Return a boolean mask of active pixels for a cropped region.
    """
    h, w = local_shot.shape[:2]
    if h == 0 or w == 0:
        return np.zeros_like(local_shot, dtype=bool)

    # choose an odd block size <= min(h, w), fallback to global if needed
    max_blk = max(3, min(h, w))
    blk = min(block_size, max_blk)
    if blk % 2 == 0:
        blk -= 1
    blk = max(3, blk)

    norm = mat2gray(local_shot)
    floor = 0.1 * norm.max() if norm.size else 0.0
    norm = norm * (norm > floor)

    try:
        th = threshold_local(norm, blk, method="gaussian")
        bw = norm > th * sensitivity
    except Exception:
        thr = norm.mean() if norm.size else 0.0
        bw = norm > thr * sensitivity

    bw = morphology.remove_small_objects(
        bw.astype(bool),
        min_size=min_size if (h * w) >= min_size else 1
    )
    return bw

def _to_float4_bbox(vals, W, H):
    """Robustly extract x1,y1,w,h as Python floats from MATLAB-style bbox arrays."""
    if vals is None:
        return 1.0, 1.0, float(W), float(H)
    flat = np.asarray(vals, dtype=object).ravel()
    if flat.size < 4:
        return 1.0, 1.0, float(W), float(H)

    def to_float(v):
        # Peel nested 1-element arrays/lists
        while isinstance(v, (np.ndarray, list, tuple)) and np.size(v) == 1:
            v = np.asarray(v, dtype=object).ravel()[0]
        try:
            return float(v)
        except Exception:
            return float(np.asarray(v, dtype=float).ravel()[0])

    return tuple(to_float(x) for x in flat[:4])
# ...existing code...

def compute_average_irradiance_per_fiber(image, fibers, exposure_ns,
                                         gate_reference_s =20.999,
                                         block_size=35, sensitivity=0.3, min_blob_size=70):
    """
    Compute average irradiance and photon flux per fiber for a single 1024x1024 image.
    """
    H, W = image.shape
    N = len(fibers)
    avgI = np.zeros(N, dtype=float)

    # exposure to seconds
    gate_ratio = (gate_reference_s / (exposure_ns * 1e-9))

    for i, fiber in enumerate(fibers):
        # Bounding box (handle MATLAB structs, dicts, and nested arrays)
        bbox = None
        if hasattr(fiber, "BoundingBox"):
            bbox = getattr(fiber, "BoundingBox")
        elif isinstance(fiber, dict) and ("BoundingBox" in fiber):
            bbox = fiber["BoundingBox"]

        x1f, y1f, wboxf, hboxf = _to_float4_bbox(bbox, W, H)

        # convert to 0-based and clamp
        x1 = max(0, int(round(x1f)) - 1)
        y1 = max(0, int(round(y1f)) - 1)
        wbox = int(round(wboxf))
        hbox = int(round(hboxf))
        x2 = min(W, x1 + wbox)
        y2 = min(H, y1 + hbox)
        if (x1 >= x2) or (y1 >= y2):
            avgI[i] = 0.0
            continue

        # crop once
        local_shot = image[y1:y2, x1:x2]
        hbox = local_shot.shape[0]
        wbox = local_shot.shape[1]

        # irradiancepcount: accept attr or dict; default 1.0
        irr_pc = getattr(fiber, "irradiancepcount", None)
        if irr_pc is None and isinstance(fiber, dict):
            irr_pc = fiber.get("irradiancepcount", 1.0)
        if irr_pc is None:
            irr_pc = 1.0

        arr_pc = np.asarray(irr_pc).reshape(-1)
        if arr_pc.size == 1:
            arr_pc = np.full(wbox, float(np.asarray(arr_pc).ravel()[0]))
        elif arr_pc.size != wbox:
            xp = np.linspace(0, 1, max(1, arr_pc.size))
            xq = np.linspace(0, 1, wbox)
            arr_pc = np.interp(xq, xp, np.asarray(arr_pc, dtype=float))

        # build irradiance image ON THE LOCAL CROP
        local_scale = np.tile(arr_pc.reshape(1, -1), (hbox, 1))
        local_irr = gate_ratio * local_shot * local_scale

        # active mask and mean
        active_mask = make_active_mask(local_shot, block_size=35,
                                       sensitivity=sensitivity, min_size=min_blob_size)
        if np.any(active_mask):
            avgI[i] = float(local_irr[active_mask].mean())
        else:
            avgI[i] = 0.0

    # compute photon flux: 4*pi*I / (h*c / lambda)
    h = 6.62607015e-34
    c = 3e8
    lam_m = 229.7e-9  # default; can be adjusted externally if needed
    photonFlux = 4 * np.pi * avgI / (h * c / lam_m)

    return avgI, photonFlux

def compute_erosion_rate(sxb_coeffs, photonFlux, fiber_centroid_indices):
    """
    sxb_coeffs : 1D array indexed by radial pixel (must align with fiber_centroid_indices)
    photonFlux  : output from compute_average_irradiance_per_fiber (1D)
    fiber_centroid_indices : integer array of length N giving radial pixel index for each fiber

    Returns erosion_rate per fiber (sxb[idx] * photonFlux)
    """
    idx = np.asarray(fiber_centroid_indices).astype(int)
    return sxb_coeffs[idx] * photonFlux

def overlay_active_region_for_display(local_shot, active_mask, alpha=0.4):
    """
    Return an RGB image where active_mask is overlaid in red with given alpha.
    local_shot used only for background luminance (grayscale).
    """
    bg = mat2gray(local_shot)
    rgb = np.stack([bg, bg, bg], axis=-1)
    red = np.zeros_like(rgb)
    red[..., 0] = 1.0
    overlay = rgb.copy()
    overlay[active_mask] = (1 - alpha) * rgb[active_mask] + alpha * red[active_mask]
    return overlay
# ...existing code...

def sxb_coefficients(Tempq, Densityq, adas_csv_path):
    """
    Python translation of the MATLAB sxb_coefficients function.

    Args:
      Tempq        : array-like of temperatures (same units as CSV temperature column)
      Densityq     : array-like of densities (same units as CSV density column)
      adas_csv_path: path to 'sxb coefficients - Sheet1.csv' (string or Path)

    Returns:
      sxb_value    : array with same shape as Tempq/Densityq with interpolated sxb values

    Notes:
      - Requires numpy as np and scipy.interpolate.griddata to be available in the caller.
      - The CSV is read, columns 2:end are transposed (matching the MATLAB code),
        then temperature = col0, density = col1 and columns 3:7 are placed into
        sxb[:,19:24]. The function then interpolates sxb over (log10(density), temperature)
        to the query points (log10(Densityq), Tempq).
    """
    # read CSV and transpose columns 2:end like MATLAB's transpose(adas_data(:,2:end))
    raw = np.loadtxt(str(adas_csv_path), delimiter=',', skiprows=0)
    adas_data = raw[:, 1:].T

    # build vectors like in MATLAB
    temperature = adas_data[:, 0]    # (N,)
    density = adas_data[:, 1]        # (N,)

    # construct sxb grid (24 x 24) and fill columns 20..24 (0-based 19:24) with adas_data cols 3..7
    sxb = np.zeros((24, 24), dtype=float)
    sxb[:, 19:24] = adas_data[:, 2:7]

    # prepare points for interpolation: (log_density, temperature)
    log_density = np.log10(density)
    pts = np.column_stack((log_density, temperature))   # shape (N,2)

    # flatten query arrays to a list of points
    dq = np.log10(np.asarray(Densityq)).ravel()
    tq = np.asarray(Tempq).ravel()
    query_pts = np.column_stack((dq, tq))               # shape (M,2)

    # interpolate. griddata supports values shaped (N, K) so we pass sxb (N,24)
    interp = griddata(pts, sxb, query_pts, method='linear')

    # interp shape will be (M, 24). The MATLAB result is a single sxb_value per query point;
    # the original code likely expects to return a single column of sxb values — keep same behavior:
    # if interp is 2D, choose appropriate column or reduce; here we take the first column where non-nan,
    # but to match MATLAB's griddata call (which returned as many columns as in sxb),
    # return interp reshaped to (Tempq.shape + (24,)) so caller can select components if needed.
    if interp is None:
        # griddata returns None if it fails
        sxb_value = np.full(tq.shape, np.nan).reshape(np.asarray(Tempq).shape + (24,))
    else:
        interp = np.asarray(interp)  # shape (M, 24)
        # reshape to original Tempq shape plus 24
        target_shape = np.asarray(Tempq).shape + (interp.shape[1],)
        sxb_value = interp.reshape(target_shape)

    return sxb_value

def path_length_annulus(b, r_in, r_out, elec_rad):
    b = abs(b)
    root_out = np.sqrt(max(r_out*r_out - b*b, 0.0))
    if b >= r_in or r_in < elec_rad:
        return root_out
    root_in = np.sqrt(max(r_in*r_in - b*b, 0.0))
    return root_out - root_in

def build_L_onion(chords_abs, edges, elec_rad=0.0):
    M = len(chords_abs)
    N = len(edges) - 1
    L = np.zeros((M, N), dtype=float)

    for i, b in enumerate(chords_abs):
        one_sided = (b < elec_rad)
        for j in range(N):
            r_in  = edges[j+1]
            r_out = edges[j]
            seg = path_length_annulus(b, r_in, r_out, elec_rad)
            if one_sided: 
                L[i, j] = seg
                if r_in < elec_rad:
                    L[i, j] = L[i, j] - np.sqrt(elec_rad*elec_rad - b*b)
            else: 
                L[i, j] = 2.0*seg
    return L

def onion_peel(L, I, eps_piv=1e-15):
    N = L.shape[0]
    eps = np.zeros(N, float)
    for k in range(N):
        contrib = np.dot(L[k, :k], eps[:k])
        rhs = I[k] - contrib
        piv = L[k, k]
        # print(f"k={k}: I={I[k]:.6g}, contrib={contrib:.6g}, rhs={rhs:.6g}, piv={piv:.6g}")
        if piv <= 0:
            raise RuntimeError(f"Zero/negative pivot at k={k}")
        eps[k] = rhs / (piv + eps_piv)
        # print(f"   -> eps[{k}]={eps[k]:.6g}")
    return eps

# def tikhonov_inversion(L, I, lam, w_inner):
#     N = L.shape[0]
#     W = np.eye(N)
#     W[-1, -1] = w_inner   # heavier regularization on inner shell
#     A = L.T @ L + (lam**2) * (W.T @ W)
#     b = L.T @ I
#     eps = np.linalg.solve(A, b)
#     return eps

def tikhonov_inversion(L, I, lam, w_inner):
    N = L.shape[0]

    # Build second-derivative matrix (size (N-2) x N)
    D2 = np.zeros((N-2, N))
    for i in range(N-2):
        D2[i, i]   = 1.0
        D2[i, i+1] = -2.0
        D2[i, i+2] = 1.0

    # Optional weighting matrix for inner bin
    W = np.eye(N)
    W[-1, -1] = w_inner  # stronger damping on inner shell if desired

    # Normal equations
    A = L.T @ L + (lam**2) * (D2.T @ D2 + W.T @ W)
    b = L.T @ I

    eps, *_ = scipy.optimize.nnls(A, b)
    return eps

def ascending_suffix(b_abs, I_avg, elec_rad, min_len=2, inner_chord_num=0):
    b_abs = np.asarray(b_abs, float)
    I_avg = np.asarray(I_avg, float)

    idx = np.where(b_abs < elec_rad)[0]
    if idx.size > 0:
        # first_inner_idx = int(idx[0])
        # b_keep = b_abs[:first_inner_idx+1]
        # I_keep = I_avg[:first_inner_idx+1]
        first_inner_idx = int(idx[0])
        b_abs = b_abs.tolist()
        b_keep = np.array(np.append(b_abs[:first_inner_idx], b_abs[first_inner_idx+inner_chord_num]))
        I_avg = I_avg.tolist()
        I_keep = np.array(np.append(I_avg[:first_inner_idx], I_avg[first_inner_idx+inner_chord_num]))
    else:
        b_keep = b_abs
        I_keep = I_avg

    if len(b_keep) < min_len:
        raise ValueError(
            f"After ascending-suffix + electrode crop, only {len(b_keep)} chord(s) remain; "
            f"need at least {min_len}."
        )

    return b_keep, I_keep

def build_synthetic_profiles(eps, b_raw, I_raw, edges):
    L = build_L_onion(b_raw, edges)
    I_synth = L @ eps
    return I_synth

def prepare_and_peel(b_raw, I_raw, lam=1e-3, w_inner=50.0,
                     R=12.42, elec_rad=0.0, N_shells=10, inner_chord_num=0, 
                     use_tikhonov = True, use_synthetic=False, eps_input=None):
    # 2) enforce your monotone-from-center rule (truncate outer side if needed)
    b_used, I_used = ascending_suffix(b_raw, I_raw, elec_rad=elec_rad, inner_chord_num=inner_chord_num)

    # b_used = b_raw
    # I_used = I_raw

    N_shells = len(b_used)
    if N_shells < 2:
        raise ValueError("Ascending range from center has < 2 chords; aborting.")

    edges = np.linspace(R, elec_rad-0.4, N_shells+1)

    if use_synthetic:
        if eps_input is None:
            raise ValueError("eps_input must be provided when use_synthetic is True.")
        I_used = build_synthetic_profiles(eps_input, b_used, I_used, edges=edges)
        I_used = I_used[:len(b_used)]
        I_syn = build_synthetic_profiles(eps_input, b_raw, I_raw, edges=edges)

    L_tot = build_L_onion(b_used, edges, elec_rad=elec_rad)
    I_used_tot = I_used

    # for i in range(3):
    #     b_used, I_used = ascending_suffix(b_raw, I_raw, elec_rad=elec_rad, inner_chord_num=i+1)
    #     I_used_tot[-1] += I_used[-1]
    #     L = build_L_onion(b_used, edges, elec_rad=elec_rad)
    #     L_tot[-1,:] += L[-1,:]

    if use_tikhonov:
        eps = tikhonov_inversion(L_tot, I_used_tot, lam=lam, w_inner=w_inner)
    else:
        # eps = scipy.linalg.lstsq(L_tot.astype(np.float64), I_used_tot.astype(np.float64))[0]
        eps = scipy.optimize.nnls(L_tot.astype(np.float64), I_used_tot.astype(np.float64))[0]

    # eps[-1] = eps[-1] / 4
    # I_used_tot[-1] = I_used_tot[-1] / 4
    if use_synthetic:
        return eps, (L_tot, edges, b_used, I_syn, edges)
    return eps, (L_tot, edges, b_used, I_used_tot, edges)

###############################################################
# Simplified irradiance & photon flux computation (no masking)
###############################################################
def compute_average_irradiance_per_fiber(image, fibers, exposure_ns,
                                         gate_reference_s=20.999,
                                         wavelength_nm=229.7):
    """Compute average irradiance and photon flux per fiber region.

    This simplified version removes any active pixel thresholding/overlay.
    Each fiber's bounding box is cropped; irradiance scaling using
    irradiancepcount (per-column) is applied; the mean over the entire
    cropped region is taken. If irradiancepcount is a scalar it is
    broadcast; if its length differs from the crop width it is linearly
    interpolated.

    Parameters
    ----------
    image : 2D ndarray
        Shot image (counts).
    fibers : sequence
        Fiber structs (MATLAB structs or dicts) each with BoundingBox and
        optionally irradiancepcount.
    exposure_ns : float
        Gate width in nanoseconds.
    gate_reference_s : float, optional
        Reference scaling constant (seconds); retained from earlier code.
    wavelength_nm : float, optional
        Wavelength for photon energy conversion.

    Returns
    -------
    avgI : ndarray (N,)
        Average irradiance per fiber (arbitrary units scaled by gate reference).
    photonFlux : ndarray (N,)
        Photon flux per fiber (photons / m^2 / s).
    """
    H, W = image.shape
    N = len(fibers)
    avgI = np.zeros(N, dtype=float)
    gate_ratio = gate_reference_s / (exposure_ns * 1e-9)

    def _bbox(vals):
        if vals is None:
            return 1.0, 1.0, float(W), float(H)
        flat = np.asarray(vals, dtype=object).ravel()
        if flat.size < 4:
            return 1.0, 1.0, float(W), float(H)
        def _f(v):
            while isinstance(v, (np.ndarray, list, tuple)) and np.size(v) == 1:
                v = np.asarray(v, dtype=object).ravel()[0]
            try:
                return float(v)
            except Exception:
                return float(np.asarray(v, dtype=float).ravel()[0])
        return tuple(_f(x) for x in flat[:4])

    for i, fiber in enumerate(fibers):
        bbox = None
        if hasattr(fiber, 'BoundingBox'):
            bbox = getattr(fiber, 'BoundingBox')
        elif isinstance(fiber, dict) and 'BoundingBox' in fiber:
            bbox = fiber['BoundingBox']
        x1f, y1f, wboxf, hboxf = _bbox(bbox)
        x1 = max(0, int(round(x1f)) - 1)
        y1 = max(0, int(round(y1f)) - 1)
        wbox = int(round(wboxf))
        hbox = int(round(hboxf))
        x2 = min(W, x1 + wbox)
        y2 = min(H, y1 + hbox)
        if (x1 >= x2) or (y1 >= y2):
            avgI[i] = 0.0
            continue
        crop = image[y1:y2, x1:x2]
        hbox = crop.shape[0]
        wbox = crop.shape[1]

        irr_pc = getattr(fiber, 'irradiancepcount', None)
        if irr_pc is None and isinstance(fiber, dict):
            irr_pc = fiber.get('irradiancepcount', 1.0)
        if irr_pc is None:
            irr_pc = 1.0
        arr_pc = np.asarray(irr_pc).reshape(-1)
        if arr_pc.size == 1:
            arr_pc = np.full(wbox, float(arr_pc.ravel()[0]))
        elif arr_pc.size != wbox:
            xp = np.linspace(0, 1, arr_pc.size)
            xq = np.linspace(0, 1, wbox)
            arr_pc = np.interp(xq, xp, arr_pc.astype(float))
        scale = np.tile(arr_pc.reshape(1, -1), (hbox, 1))
        irr = gate_ratio * crop * scale
        avgI[i] = irr.mean() if irr.size else 0.0

    h = 6.62607015e-34
    c = 3e8
    lam_m = wavelength_nm * 1e-9
    photonFlux = 4 * np.pi * avgI / (h * c / lam_m)
    return avgI, photonFlux

###############################################################
# Strict 20-peak extraction from a middle column
###############################################################
def extract_20_peaks_middle_column(image, smooth_window=12, min_distance=35,
                                   start_mult=0.6, max_iter=40, allow_trim=True):
    """Return exactly 20 peak intensities along the middle column.

    Adaptive threshold lowers until >=20 peaks found. If more than 20 peaks
    are present and *allow_trim* is True, the tallest 20 are selected and
    then reordered by spatial position.

    Returns
    -------
    intensities : ndarray shape (20,) or None if cannot find 20.
    peaks_idx   : ndarray of peak indices (after spatial ordering) or None.
    """
    from scipy.signal import find_peaks
    H, W = image.shape
    col_idx = W // 2
    col = image[:, col_idx].astype(float)
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        col_smooth = np.convolve(col, kernel, mode='same')
    else:
        col_smooth = col

    multiplication = start_mult
    peaks, props = [], {}
    for it in range(max_iter):
        height_thresh = col_smooth.mean() - multiplication * col_smooth.std()
        peaks, props = find_peaks(col_smooth, distance=min_distance, height=height_thresh)
        if len(peaks) == 20:
            break
        if len(peaks) > 20 and allow_trim:
            break  # will trim after loop
        multiplication += 0.1  # lower threshold progressively

    if len(peaks) < 20:
        return None, None

    peak_heights = props.get('peak_heights', col_smooth[peaks])
    if len(peaks) > 20:
        top_idx_global = np.argsort(peak_heights)[-20:]
        peaks_20 = peaks[top_idx_global]
        heights_20 = peak_heights[top_idx_global]
    else:
        peaks_20 = peaks
        heights_20 = peak_heights

    # reorder by spatial position
    order = np.argsort(peaks_20)
    peaks_sorted = peaks_20[order]
    intensities = heights_20[order]
    return intensities.astype(float), peaks_sorted.astype(int)
