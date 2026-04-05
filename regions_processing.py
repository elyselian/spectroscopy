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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def extract_fiber_and_calibrate(
    img: np.ndarray,
    *,
    lamp_csv_path: str,
    min_area: int = 70,
    sauvola_window: int = 51,
    sauvola_k: float = 0.2,
    # Profile parameters (important!)
    profile_half_height: int = 3,   # average rows (centroid-row +/- this many)
    smooth_window: int = 30,
    n_pixels: int = 1024,
    n_fibers: int = 1,
):
    """
    1) Segment fiber regions.
    2) Pick the top *n_fibers* regions by area.
    3) For each, compute a *full-width* (length 1024) row profile from rows
       around the region centroid.
    4) Compute irradiance-per-count multiplier via lamp calibration.

    Returns:
        n_fibers == 1 (default): single dict, or None if nothing found.
        n_fibers  > 1:           list of dicts sorted top-to-bottom by centroid row,
                                 may be shorter than n_fibers if fewer regions exist.
                                 Empty list if nothing found.
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

    # Take top n_fibers regions by area, then sort top-to-bottom
    props_sorted = sorted(props, key=lambda r: r.area, reverse=True)[:n_fibers]
    props_sorted.sort(key=lambda r: r.centroid[0])  # sort by row (top → bottom)

    def _build_fiber_dict(region):
        minr, minc, maxr, maxc = region.bbox
        crop = img[minr:maxr, minc:maxc]

        centroid_r, centroid_c = region.centroid
        centroid_index = (int(round(centroid_r)), int(round(centroid_c)))

        r0 = max(0, centroid_index[0] - profile_half_height)
        r1 = min(img.shape[0], centroid_index[0] + profile_half_height + 1)
        row_profile_full = img[r0:r1, :].mean(axis=0).astype(np.float64)

        row_profile_full_smooth = uniform_filter1d(row_profile_full, size=smooth_window, mode="nearest")

        irr = get_irradiance_multiplier(
            row_profile_full_smooth,
            lamp_csv_path=lamp_csv_path,
            n_pixels=n_pixels
        )

        return {
            "bbox": (int(minc), int(minr), int(maxc - minc), int(maxr - minr)),
            "centroid": (float(centroid_r), float(centroid_c)),
            "centroid_index": centroid_index,
            "area": int(region.area),
            "crop": crop,
            "row_profile_full": row_profile_full_smooth,
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