"""
Fiber ROI extraction + irradiance-per-count multiplier pipeline (Python-native)

Root path is Windows:
    G:\Shared drives\Shumlak Lab

What this does:
1) Loops over calibration SPEs (e.g., 250422  007.spe ... 026.spe)
2) Segments the bright fiber region (robust local threshold)
3) Stores:
   - bbox (x, y, w, h)
   - centroid (row, col) and centroid_index
   - crop (ROI image)
   - row_profile_full: 1D counts profile across the *full 1024-pixel width*,
     computed by averaging a few rows around the region centroid (recommended)
   - irradiance multiplier vs pixel: µW/cm^2/count (matches your MATLAB core behavior)

Saves:
    fiber_regions.pkl  (recommended, lossless, fast)

Dependencies:
    pip install numpy scipy scikit-image matplotlib spe2py pandas
"""

import os
import pickle
from threading import local
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from scipy.integrate import trapz
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import CubicSpline
from skimage.filters import threshold_sauvola
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
from matplotlib.patches import Rectangle

import spe_loader as sl  # this is what your installed spe2py uses


ROOT = r"G:\Shared drives\Shumlak Lab"
S_XB_DIR = os.path.join(ROOT, "Diagnostics", "Spectroscopy", "S_XB")

CAL_FOLDER = os.path.join(S_XB_DIR, "Data", "Calibration", "250422")
CAL_PREFIX = "250422"
CAL_RANGE = range(7, 27)  #

LAMP_CSV_PATH = r"G:\Shared drives\Shumlak Lab\Diagnostics\Spectroscopy\S_XB\Ocean Optics DH-3-Plus\DH3_Plus_Calibration_UV - Sheet1.csv"

OUT_FILE = "fiber_regions.pkl"

# def load_spe_image(path: str) -> np.ndarray:
#     # load_from_files returns a list of SPE objects
#     f = sl.load_from_files([str(path)])
#     img = np.asarray(f.data[0][0], dtype=np.float32)
#     return np.asarray(img)

import numpy as np
import struct
import xml.etree.ElementTree as ET

def load_spe_image(path: str) -> np.ndarray:
    """
    Read the *largest 2D ROI* image from a Princeton Instruments SPE v3.x file.

    This bypasses spe_loader's default selection (which in your 250422 files is
    returning 3x(1x1024) track ROIs) and instead selects the largest image ROI
    found in the XML footer, then reads that block from the binary pixel data.

    Returns: 2D np.ndarray (H, W)
    """
    with open(path, "rb") as f:
        # ---- SPE version (float32) at byte 1992
        f.seek(1992)
        version = struct.unpack("<f", f.read(4))[0]
        if version < 3.0:
            raise ValueError(f"{path} is SPE v{version:.2f} (<3). This loader is for v3.x.")

        # ---- XML footer offset (uint64) at byte 678
        f.seek(678)
        footer_offset = struct.unpack("<Q", f.read(8))[0]

        # ---- Read XML footer
        f.seek(footer_offset)
        xml_bytes = f.read()
        xml_text = xml_bytes.decode("utf-8", errors="ignore")
        root = ET.fromstring(xml_text)

        def strip_ns(tag):
            return tag.split("}", 1)[-1] if "}" in tag else tag

        # ---- Find all "top-level" DataBlock parents that declare pixelFormat + count
        # and contain child DataBlock entries with width/height.
        datasets = []
        for el in root.iter():
            if strip_ns(el.tag) != "DataBlock":
                continue
            if "pixelFormat" in el.attrib and "count" in el.attrib:
                children = [c for c in list(el) if strip_ns(c.tag) == "DataBlock" and
                            "width" in c.attrib and "height" in c.attrib]
                if children:
                    pixel_format = el.attrib["pixelFormat"]
                    nframes = int(el.attrib["count"])
                    rois = [(int(c.attrib["height"]), int(c.attrib["width"])) for c in children]  # (H,W)
                    datasets.append({
                        "pixelFormat": pixel_format,
                        "nFrames": nframes,
                        "rois": rois,
                        "xml_el": el,
                    })

        if not datasets:
            raise RuntimeError("No readable DataBlock datasets found in XML footer.")

        # ---- Map pixelFormat -> dtype/bytes
        def pf_to_dtype(pf):
            if pf == "MonochromeUnsigned16":
                return np.uint16
            if pf == "MonochromeUnsigned32":
                return np.uint32
            if pf == "MonochromeFloating32":
                return np.float32
            raise ValueError(f"Unsupported pixelFormat: {pf}")

        # ---- Choose dataset with the largest single ROI area (prefer 2D images)
        def dataset_score(ds):
            return max(h*w for (h,w) in ds["rois"])

        chosen_idx = int(np.argmax([dataset_score(ds) for ds in datasets]))
        chosen = datasets[chosen_idx]
        dtype = pf_to_dtype(chosen["pixelFormat"])
        bpp = np.dtype(dtype).itemsize

        # ---- Figure out where the chosen dataset begins in the binary pixel stream.
        # SPE v3 pixel data begins at byte 4100 (same assumption as your MATLAB loader).
        # Datasets are stored sequentially in the order they appear in the XML; we
        # approximate file order by the order we discovered them in iteration.
        #
        # We compute how many bytes to skip for datasets before the chosen one.
        def dataset_n_values(ds):
            return ds["nFrames"] * sum(h*w for (h,w) in ds["rois"])

        skip_values = sum(dataset_n_values(ds) for ds in datasets[:chosen_idx])
        data_start = 4100 + skip_values * bpp

        # ---- Now read the chosen dataset (frame 0, ROI = largest 2D ROI)
        # Read all ROIs for frame 0 so we can pick the largest ROI robustly.
        f.seek(data_start)

        nframes = chosen["nFrames"]
        rois = chosen["rois"]

        # File ordering (common in PI SPE): ROIs stored sequentially per frame
        # We'll read only frame 0 (first frame).
        frame0_rois = []
        for (h, w) in rois:
            nvals = h * w
            arr = np.fromfile(f, dtype=dtype, count=nvals)
            if arr.size != nvals:
                raise EOFError(f"Unexpected EOF reading ROI (H={h}, W={w}). Read {arr.size}/{nvals}")
            # MATLAB fread([width,height]) then transpose to (H,W)
            frame0_rois.append(arr.reshape((w, h)))

        # Pick largest 2D ROI from frame 0
        areas = [r.shape[0]*r.shape[1] for r in frame0_rois]
        img = frame0_rois[int(np.argmax(areas))]

        return img

def get_irradiance_multiplier(
    row_intensity_full: np.ndarray,
    *,
    lamp_csv_path: str,
    wavelength_min_nm: float = 226.654,
    wavelength_max_nm: float = 232.726,
    n_pixels: int = 1024,
    eps: float = 1e-12,
) -> dict:
    """
    Core behavior from your MATLAB getIrradianceMultiplier_V2:

      lamp CSV: wavelength (nm), irradiance_per_nm (µW/cm^2/nm)
      irradiance_q = spline_interp(irradiance_per_nm onto wavelength_query) * wavelength_query
      irr_multiplier = irradiance_q / count_fitted
      (optionally scaled by gate_ratio)

    Returns a dict with multiplier and intermediate curves.
    """
    ri = np.asarray(row_intensity_full, dtype=np.float64).ravel()

    # Force length == n_pixels
    if ri.size < n_pixels:
        ri = np.pad(ri, (0, n_pixels - ri.size), mode="edge")
    elif ri.size > n_pixels:
        ri = ri[:n_pixels]

    # Counts should not be negative for this ratio
    ri = np.maximum(ri, 0.0)

    df = pd.read_csv(lamp_csv_path, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"Lamp CSV must have >=2 columns; got {df.shape}")

    wavelength = df.iloc[:, 0].to_numpy(dtype=np.float64)
    irr_per_nm = df.iloc[:, 1].to_numpy(dtype=np.float64)

    # Sort + de-duplicate wavelengths for CubicSpline
    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    irr_per_nm = irr_per_nm[order]
    wavelength, uniq_idx = np.unique(wavelength, return_index=True)
    irr_per_nm = irr_per_nm[uniq_idx]

    wavelength_q = np.linspace(wavelength_min_nm, wavelength_max_nm, n_pixels)

    spline = CubicSpline(wavelength, irr_per_nm, extrapolate=False)
    irr_per_nm_q = spline(wavelength_q)

    # Fill any out-of-range NaNs with nearest valid edge
    if np.any(~np.isfinite(irr_per_nm_q)):
        valid = np.isfinite(irr_per_nm_q)
        if not np.any(valid):
            raise ValueError("Lamp interpolation produced all-NaN; check wavelength range vs CSV.")
        first = irr_per_nm_q[valid][0]
        last = irr_per_nm_q[valid][-1]
        irr_per_nm_q = np.where(
            np.isfinite(irr_per_nm_q),
            irr_per_nm_q,
            np.where(wavelength_q < wavelength[0], first, last),
        )

    # Match your MATLAB behavior: * wavelength_query
    irradiance_q = irr_per_nm_q * wavelength_q  # "µW/cm^2" per your pipeline

    # To get irradiance of a shot image, do gate_ratio * multiplier * counts
    multiplier = irradiance_q / np.maximum(ri, eps)

    return {
        "pixel": np.arange(n_pixels, dtype=np.int32),
        "wavelength_nm": wavelength_q,
        "irradiance_q_uW_cm2": irradiance_q,
        "row_intensity_full": ri,
        "multiplier_uW_cm2_per_count": multiplier,
    }


def normalize01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-12)

def contrast_clip_rescale(img, low_pct=1.0, high_pct=99.7):
    """
    Clip intensity tails and rescale to [0,1] for segmentation.
    Good default: low_pct ~ 0.5–2, high_pct ~ 99.5–99.9.
    """
    x = img.astype(np.float32)
    lo = np.percentile(x, low_pct)
    hi = np.percentile(x, high_pct)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    return x

import numpy as np
from scipy.ndimage import uniform_filter1d
from skimage.filters import threshold_sauvola
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects


def extract_fiber_and_calibrate(
    img: np.ndarray,
    *,
    lamp_csv_path: str,
    min_area: int = 70,
    sauvola_window: int = 51,
    sauvola_k: float = 0.2,
    profile_half_height: int = 3,      # extraction half-height around traced centerline
    trace_half_height: int = 8,        # search half-height for finding yc(x)
    smooth_window: int = 31,           # smoothing for traced centerline
    spectral_smooth_window: int = 30,  # smoothing for final extracted spectrum
    n_pixels: int = 1024,
    n_fibers: int = 20,
):
    """
    1) Segment fiber regions.
    2) Pick the top *n_fibers* regions by area.
    3) For each fiber, trace the centerline yc(x) across the full detector width.
    4) Extract a wavelength profile by averaging rows around yc(x).
    5) Compute irradiance-per-count multiplier via lamp calibration.

    Returns:
        n_fibers == 1: single dict, or None if nothing found
        n_fibers > 1 : list of dicts sorted top-to-bottom
    """
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got {img.shape}")

    if img.shape[1] != n_pixels:
        raise ValueError(f"Expected image width {n_pixels}, got {img.shape[1]}")

    x = contrast_clip_rescale(img, low_pct=1.0, high_pct=99.7)

    thr = threshold_sauvola(x, window_size=sauvola_window, k=sauvola_k)
    bw = x > thr
    bw = remove_small_objects(bw, min_size=min_area)

    lab = label(bw)
    props = regionprops(lab, intensity_image=img)
    if not props:
        return None if n_fibers == 1 else []

    props_sorted = sorted(props, key=lambda r: r.area, reverse=True)[:n_fibers]
    props_sorted.sort(key=lambda r: r.centroid[0])  # top -> bottom

    H, W = img.shape
    y_coords = np.arange(H, dtype=np.float64)

    def _trace_centerline(region):
        """
        Compute yc(x): fiber center row as a function of detector column x.
        Uses intensity-weighted centroid within a local vertical window.
        """
        centroid_r, centroid_c = region.centroid
        yc = np.full(W, np.nan, dtype=np.float64)

        # Start with a global seed row from region centroid
        seed_row = int(round(centroid_r))

        for col in range(W):
            r0 = max(0, seed_row - trace_half_height)
            r1 = min(H, seed_row + trace_half_height + 1)

            col_vals = img[r0:r1, col].astype(np.float64)

            # Background remove with local minimum so centroid isn't biased
            col_vals = col_vals - np.min(col_vals)
            s = col_vals.sum()

            if s > 0:
                rows_local = np.arange(r0, r1, dtype=np.float64)
                yc[col] = np.sum(rows_local * col_vals) / s

        # Fill any missing columns by interpolation
        good = np.isfinite(yc)
        if good.sum() < 2:
            # fallback: flat line at centroid
            yc[:] = centroid_r
        else:
            xp = np.flatnonzero(good)
            fp = yc[good]
            yc = np.interp(np.arange(W), xp, fp)

        # Smooth centerline
        if smooth_window > 1:
            yc = uniform_filter1d(yc, size=smooth_window, mode="nearest")

        return yc

    def _extract_profile_from_trace(yc):
        """
        Extract 1D spectrum by averaging rows around traced centerline yc(x).
        """
        profile = np.zeros(W, dtype=np.float64)

        for col in range(W):
            yc_col = yc[col]
            r_center = int(round(yc_col))
            r0 = max(0, r_center - profile_half_height)
            r1 = min(H, r_center + profile_half_height + 1)
            profile[col] = img[r0:r1, col].mean()

        if spectral_smooth_window > 1:
            profile = uniform_filter1d(
                profile,
                size=spectral_smooth_window,
                mode="nearest"
            )

        return profile

    def _build_fiber_dict(region):
        minr, minc, maxr, maxc = region.bbox
        crop = img[minr:maxr, minc:maxc]

        centroid_r, centroid_c = region.centroid
        centroid_index = (int(round(centroid_r)), int(round(centroid_c)))

        # Step 1: trace the fiber centerline yc(x)
        centerline_y = _trace_centerline(region)

        # Step 2: extract along traced centerline
        row_profile_full = _extract_profile_from_trace(centerline_y)

        irr = get_irradiance_multiplier(
            row_profile_full,
            lamp_csv_path=lamp_csv_path,
            n_pixels=n_pixels
        )

        return {
            "bbox": (int(minc), int(minr), int(maxc - minc), int(maxr - minr)),
            "centroid": (float(centroid_r), float(centroid_c)),
            "centroid_index": centroid_index,
            "area": int(region.area),
            "crop": crop,
            "centerline_y": centerline_y,   # yc(x)
            "row_profile_full": row_profile_full,
            "irradiance_multiplier": irr["multiplier_uW_cm2_per_count"],
            "wavelength_nm": irr["wavelength_nm"],
            "irradiance_q_uW_cm2": irr["irradiance_q_uW_cm2"],
        }

    if n_fibers == 1:
        return _build_fiber_dict(props_sorted[0])

    return [_build_fiber_dict(r) for r in props_sorted]


def process_calibration_folder():
    fibers = []
    for idx in CAL_RANGE:
        file_num = f"{idx:03d}"
        fname = f"{CAL_PREFIX}  {file_num}.spe"  # double spaces
        fpath = os.path.join(CAL_FOLDER, fname)

        if not os.path.isfile(fpath):
            print(f"[WARN] Missing: {fpath}")
            continue

        img = load_spe_image(fpath)

        out = extract_fiber_and_calibrate(
            img,
            lamp_csv_path=LAMP_CSV_PATH,
            min_area=70,
            sauvola_window=51,
            sauvola_k=0.2,
            profile_half_height=3,
            smooth_window=30,
            gate_ratio=1.0,   # set to gate_cal / gate_shot if you want that correction here
            n_pixels=1024,
        )

        if out is None:
            print(f"[WARN] No region found in {fname}")
            continue

        out["filename"] = fname
        fibers.append(out)

    print(f"[INFO] Extracted {len(fibers)} fibers")
    return fibers


def show_fiber_boxes(img: np.ndarray, fibers, title="Fiber Regions"):
    fig, ax = plt.subplots()

    vmin = np.percentile(img, 1)
    vmax = np.percentile(img, 99.7)
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.axis("off")

    for i, f in enumerate(fibers, start=1):
        x, y, w, h = f["bbox"]
        ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=2))
        ax.text(x, max(y - 5, 0), f"Fiber {i}", fontsize=8)
        ax.hlines(y,     x, x + w, color="red", linestyle="--", linewidth=1)  # top
        ax.hlines(y + h, x, x + w, color="red", linestyle="--", linewidth=1)  # bottom
        ax.vlines(x,     y, y + h, color="red", linestyle="--", linewidth=1)  # left
        ax.vlines(x + w, y, y + h, color="red", linestyle="--", linewidth=1)  # right

    plt.show() 


def plot_calibration_curves(fiber, title_prefix=""):
    px = np.arange(len(fiber["row_profile_full"]))
    wl = fiber["wavelength_nm"]
    counts = fiber["row_profile_full"]
    irr_q = fiber["irradiance_q_uW_cm2"]
    mult = fiber["irradiance_multiplier"]

    fig, ax = plt.subplots()
    ax.plot(px, counts)
    ax.set_title(f"{title_prefix}Counts profile (smoothed) vs pixel")
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Counts")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(wl, irr_q)
    ax.set_title(f"{title_prefix}Lamp irradiance_q (interp * wavelength) vs wavelength")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Irradiance_q (µW/cm²)")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(px, mult)
    ax.set_title(f"{title_prefix}Irradiance-per-count multiplier vs pixel")
    ax.set_xlabel("Pixel")
    ax.set_ylabel("µW/cm² / count")
    plt.show()


def build_fiber_calibration():
    if not os.path.isfile(LAMP_CSV_PATH):
        print(f"[WARN] Lamp CSV not found at:\n  {LAMP_CSV_PATH}\n"
              f"Edit LAMP_CSV_PATH to the correct location.")

    fibers = process_calibration_folder()
    with open(OUT_FILE, "wb") as f:
        pickle.dump(fibers, f)
    return fibers

import numpy as np
from scipy.ndimage import uniform_filter1d
from skimage.filters import threshold_sauvola
from skimage.morphology import remove_small_objects

def extract_real_image_fiber_intensities(
    img: np.ndarray,
    fibers: list,
    *,
    min_area: int = 20,
    sauvola_window: int = 31,
    sauvola_k: float = 0.2,
    bbox_pad_y: int = 6,
    bbox_pad_x: int = 6,
    trace_half_height: int = 6,
    profile_half_height: int = 3,
    centerline_smooth_window: int = 21,
    spectrum_smooth_window: int = 1,
    use_trapz: bool = True,
):
    """
    For each fiber (defined from calibration image geometry), do the following on a real image:
      1) restrict to the fiber's local bbox region
      2) threshold to find illuminated pixels
      3) background subtract using dark pixels in that same local region
      4) collapse spatial dimension -> 1D intensity vs wavelength
      5) convert counts to uW/cm^2 using irradiance_multiplier
      6) integrate spectral dimension -> one scalar intensity per fiber
      7) return all 20 values in one array

    Parameters
    ----------
    img : 2D ndarray
        Real image.
    fibers : list of dict
        Output from your calibration extraction function, one dict per fiber.
        Each fiber dict must contain:
            - "bbox"
            - "centroid"
            - "irradiance_multiplier"
            - optionally "centerline_y"
            - optionally "wavelength_nm"
    min_area : int
        Minimum illuminated blob area inside local bbox.
    sauvola_window, sauvola_k
        Same threshold logic style as calibration.
    bbox_pad_y, bbox_pad_x
        Extra margin around stored bbox when searching for illuminated region.
    trace_half_height
        Vertical search half-height for local centroid tracing.
    profile_half_height
        Vertical half-height used for spatial collapse around traced centerline.
    centerline_smooth_window
        Smoothing window for traced centerline.
    spectrum_smooth_window
        Optional smoothing of final 1D spectrum.
    use_trapz : bool
        If True and wavelength_nm exists, integrate with np.trapz over wavelength.
        Otherwise sum over pixel index.

    Returns
    -------
    result : dict with keys
        "fiber_intensities_uW_cm2" : (N,) ndarray
        "spectra_counts"           : list of 1D arrays
        "spectra_uW_cm2"           : list of 1D arrays
        "illuminated_masks"        : list of 2D bool arrays (local bbox masks)
        "local_bboxes"             : list of (x, y, w, h)
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got {img.shape}")

    H, W = img.shape
    y_all = np.arange(H, dtype=np.float64)

    intensities = []
    spectra_counts = []
    spectra_uW = []
    illuminated_masks = []
    local_bboxes = []

    for fiber in fibers:
        # --- calibration bbox ---
        bx, by, bw, bh = fiber["bbox"]   # x, y, w, h

        # expand local search box a bit
        x0 = max(0, bx - bbox_pad_x)
        x1 = min(W, bx + bw + bbox_pad_x)
        y0 = max(0, by - bbox_pad_y)
        y1 = min(H, by + bh + bbox_pad_y)

        local = img[y0:y1, x0:x1]

        # --- threshold illuminated region inside this fiber's local area ---
        x_local = contrast_clip_rescale(local, low_pct=1.0, high_pct=99.7)
        thr = threshold_sauvola(x_local, window_size=sauvola_window, k=sauvola_k)
        illum_mask = x_local > thr
        illum_mask = remove_small_objects(illum_mask, min_size=min_area)

        # If nothing survives threshold, return zeros for this fiber
        if not np.any(illum_mask):
            local_bboxes.append((x0, y0, x1 - x0, y1 - y0))
            illuminated_masks.append(illum_mask)
            spectra_counts.append(np.zeros(W, dtype=np.float64))
            spectra_uW.append(np.zeros(W, dtype=np.float64))
            intensities.append(0.0)
            continue

        # --- background estimate from dark pixels in the SAME local bbox ---
        dark_mask = ~illum_mask

        # columnwise background from dark region
        bg_col = np.full(local.shape[1], np.nan, dtype=np.float64)

        # First pass: compute background where dark pixels exist
        for j in range(local.shape[1]):
            dark_vals = local[dark_mask[:, j], j]
            if dark_vals.size > 0:
                bg_col[j] = np.median(dark_vals)

        # Second pass: fill missing columns by interpolation
        good = np.isfinite(bg_col)

        if good.sum() >= 2:
            xp = np.flatnonzero(good)
            fp = bg_col[good]
            bg_col = np.interp(np.arange(local.shape[1]), xp, fp)
        else:
            # fallback if almost everything is illuminated
            bg_col[:] = np.median(local)

        # subtract background column-by-column
        local_bs = local - bg_col[None, :]

        # --- trace centerline only where fiber is illuminated ---
        yc_local = np.full(local.shape[1], np.nan, dtype=np.float64)

        # use calibration centerline if available, otherwise centroid row
        if "centerline_y" in fiber:
            yc_seed_global = np.asarray(fiber["centerline_y"], dtype=np.float64)
            yc_seed_local = yc_seed_global[x0:x1] - y0
        else:
            yc_seed_local = np.full(local.shape[1], fiber["centroid"][0] - y0, dtype=np.float64)

        for j in range(local.shape[1]):
            rows_illum = np.flatnonzero(illum_mask[:, j])

            if rows_illum.size == 0:
                continue

            # restrict centroid calculation around expected fiber location
            seed_row = int(round(yc_seed_local[j]))
            r0 = max(0, seed_row - trace_half_height)
            r1 = min(local.shape[0], seed_row + trace_half_height + 1)

            rows_use = np.arange(r0, r1)
            valid = illum_mask[r0:r1, j]

            if not np.any(valid):
                continue

            vals = local_bs[r0:r1, j].copy()
            vals[~valid] = 0.0
            vals = np.clip(vals, 0.0, None)

            s = vals.sum()
            if s > 0:
                yc_local[j] = np.sum(rows_use * vals) / s

        # fill missing columns by interpolation, fall back to seed
        good = np.isfinite(yc_local)
        if good.sum() >= 2:
            xp = np.flatnonzero(good)
            fp = yc_local[good]
            yc_local = np.interp(np.arange(local.shape[1]), xp, fp)
        else:
            yc_local = yc_seed_local.copy()

        if centerline_smooth_window > 1:
            yc_local = uniform_filter1d(
                yc_local,
                size=centerline_smooth_window,
                mode="nearest"
            )

        # --- collapse spatial dimension to get 1D counts spectrum ---
        spectrum_local = np.zeros(local.shape[1], dtype=np.float64)

        for j in range(local.shape[1]):
            yc_j = int(round(yc_local[j]))
            r0 = max(0, yc_j - profile_half_height)
            r1 = min(local.shape[0], yc_j + profile_half_height + 1)

            vals = local_bs[r0:r1, j]

            # only keep positive, background-subtracted signal
            vals = np.clip(vals, 0.0, None)

            # use SUM, not mean, so this represents total spectral brightness
            spectrum_local[j] = vals.sum()

        if spectrum_smooth_window > 1:
            spectrum_local = uniform_filter1d(
                spectrum_local,
                size=spectrum_smooth_window,
                mode="nearest"
            )

        # --- embed local spectrum into full-width spectrum ---
        spectrum_full = np.zeros(W, dtype=np.float64)
        spectrum_full[x0:x1] = spectrum_local

        # --- convert to uW/cm^2 using irradiance multiplier array ---
        irr_mult = np.asarray(fiber["irradiance_multiplier"], dtype=np.float64)
        if irr_mult.shape[0] != W:
            raise ValueError("irradiance_multiplier length does not match image width")

        spectrum_uW = spectrum_full * irr_mult

        # --- integrate spectral dimension to one scalar ---
        wavelength_nm = fiber.get("wavelength_nm", None)

        if use_trapz and wavelength_nm is not None:
            wavelength_nm = np.asarray(wavelength_nm, dtype=np.float64)
            if wavelength_nm.shape[0] != W:
                raise ValueError("wavelength_nm length does not match image width")
            intensity_val = np.trapezoid(spectrum_uW, wavelength_nm)
        else:
            intensity_val = np.sum(spectrum_uW)

        local_bboxes.append((x0, y0, x1 - x0, y1 - y0))
        illuminated_masks.append(illum_mask)
        spectra_counts.append(spectrum_full)
        spectra_uW.append(spectrum_uW)
        intensities.append(float(intensity_val))

    return {
        "fiber_intensities_uW_cm2": np.array(intensities, dtype=np.float64),
        "spectra_counts": spectra_counts,
        "spectra_uW_cm2": spectra_uW,
        "illuminated_masks": illuminated_masks,
        "local_bboxes": local_bboxes,
    }