from pathlib import Path
import hashlib, csv, re, sys
import numpy as np
import h5py
import spe_loader as sl
import json
from typing import Any

# Raw directory storing Ops SPE files
RAW_DIR   = Path("G:\\Shared drives\\Shumlak Lab\\Diagnostics\\Spectroscopy\\S_XB\\Data\\Ops")

# Location of output HDF5 file
OUT_H5    = Path("G:\\Shared drives\\Shumlak Lab\\Users\\Current\\Elyse Lian\\Python Code\\Spectroscopy\\experiment.h5")

# Location of manifest CSV file
MANIFEST  = Path("G:\\Shared drives\\Shumlak Lab\\Users\\Current\\Elyse Lian\\Python Code\\Spectroscopy\\manifest.csv")

# Number of fibers
N_FIBERS  = 20

def load_manifest():
    """
    Load manifest into dict[int, row].

    If manifest does not exist, create an empty one with header
    and return {}.
    """
    rows: dict[int, dict] = {}

    # --- create manifest if missing ---
    if not MANIFEST.exists():
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        with MANIFEST.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["shot_id", "raw_path", "raw_md5", "H", "W"]
            )
            w.writeheader()
        return rows

    # --- load existing manifest ---
    with MANIFEST.open(newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                shot_id = int(r["shot_id"])
            except (KeyError, ValueError, TypeError):
                continue

            r["shot_id"] = shot_id

            # optional int coercion
            for k in ("H", "W"):
                if k in r:
                    try:
                        r[k] = int(r[k])
                    except Exception:
                        pass

            rows[shot_id] = r

    return rows

def get_shot_voltage(c, shot: int):
    """
    Retrieve shot voltages from MDSplus (zaphd tree) for a given shot.

    Raises RuntimeError with context if:
      - connection/tree open fails
      - any signal fetch fails
      - returned data is empty/non-numeric
    """
    if not isinstance(shot, int):
        raise TypeError(f"get_shot_voltage: shot must be int, got {type(shot).__name__}")

    try:
        c.openTree("zaphd", shot)
    except Exception as e:
        raise RuntimeError(f"MDSplus: failed to open tree 'zaphd' for shot={shot}: {e}") from e

    def _max_abs(node: str) -> float:
        try:
            arr = np.array(c.get(node))
        except Exception as e:
            raise RuntimeError(f"MDSplus: failed to get node {node!r} for shot={shot}: {e}") from e

        if arr.size == 0:
            raise RuntimeError(f"MDSplus: node {node!r} returned empty array for shot={shot}")

        # Try numeric conversion
        try:
            arr = arr.astype(float)
        except Exception as e:
            raise RuntimeError(
                f"MDSplus: node {node!r} returned non-numeric data (dtype={arr.dtype}) for shot={shot}: {e}"
            ) from e

        return float(np.max(np.abs(arr)))

    v_c = _max_abs(r"\v_compress")
    v_a = _max_abs(r"\v_accel")
    v_m = _max_abs(r"\v_middle")
    return v_c, v_a, v_m

def parse_shot_id(path: Path) -> int | None:
    """
    Parse a shot_id from an SPE filename.

    Expected filename stem pattern (case-insensitive):
        "<date><spaces><shot>"
    where:
        date = 25xxxxx (e.g., 250219)
        shot = integer (will be zero-padded to 3 digits)

    Returns:
        int(date + shot3)  e.g. date=250219, shot=17 -> 250219017

    Skips:
        stems containing "-raw"
        stems not matching the pattern
    """
    s = path.stem.lower()
    if "-raw" in s:
        return None

    m = re.search(r"(25\d{4})\s+(\d+)", s)
    if not m:
        return None

    date = m.group(1)            # "250219"
    shot = m.group(2).zfill(3)   # "017"
    return int(date + shot)      # 250219017

def manifest_contains_shot_id(shot_id: int, ingested: dict[int, dict]) -> bool:
    """
    Return True if shot_id is already present in the in-memory manifest dict.

    This is the int-only replacement for path+md5 matching.
    """
    if not isinstance(shot_id, int):
        raise TypeError(f"manifest_contains_shot_id: shot_id must be int, got {type(shot_id).__name__}")
    return shot_id in ingested

def read_spe(
    path: Path,
    c,
    shot: int,
    lamb: float = 229.7,
    gate: float = 100,
    electrode_copy: str = "A",
    iccd_delay: float = 100,
):
    """
    Load an SPE file and return (image, metadata).

    - Reads the image as float32 from spe_loader.
    - Tries to extract metadata from the SPE XML footer if present:
        sensor_height, sensor_width, ICCD delay, gate width, central wavelength
      If footer extraction fails, falls back to manual defaults.
    - Queries MDSplus for shot voltages using the given shot (int).
    - Returns:
        img (H x W float32)
        meta (dict)
    """
    if not isinstance(shot, int):
        raise TypeError(f"read_spe: shot must be int, got {type(shot).__name__}")

    f = sl.load_from_files([str(path)])
    img = np.asarray(f.data[0][0], dtype=np.float32)

    meta: dict = {}
    try:
        ft = f.footer
    except Exception as e:
        ft = None
        print(f"Unable to access footer: {e}")

    # Extract known metadata from footer if present
    if ft and (getattr(ft, "attributes", None) or getattr(ft, "children", None) or getattr(ft, "cdata", None)):
        try:
            # sensor dimensions (if available)
            try:
                meta["sensor_height"] = int(ft.SpeFormat.Calibrations.SensorInformation["height"])
                meta["sensor_width"]  = int(ft.SpeFormat.Calibrations.SensorInformation["width"])
            except Exception:
                pass

            # Gate delay and width (ns)
            try:
                gate_delay = np.array([
                    ft.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Cameras.Camera.Gating.RepetitiveGate.Pulse["delay"]
                ]).astype(float)
                gate_width = np.array([
                    ft.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Cameras.Camera.Gating.RepetitiveGate.Pulse["width"]
                ]).astype(float)
                meta["ICCD Delay (ns)"] = float(gate_delay[0])
                meta["Gate Width (ns)"] = float(gate_width[0])
            except Exception:
                print("Failed to extract gate delay/width from footer. Will use manual values.")

            # Center wavelength (nm)
            try:
                cwl = np.array([
                    ft.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Spectrometers.Spectrometer.Grating.CenterWavelength.cdata
                ]).astype(float)
                meta["Central Wavelength (nm)"] = float(cwl[0])
            except Exception:
                print("Failed to extract center wavelength from footer. Will use manual value.")

        except Exception as e:
            print(f"Error while extracting footer metadata: {e}. Falling back to manual values.")
    else:
        print("Footer is empty or missing metadata. Using manual assignments for gate/lambda.")

    # Manual defaults if missing
    meta.setdefault("Gate Width (ns)", gate)
    meta.setdefault("Central Wavelength (nm)", lamb)
    meta.setdefault("ICCD Delay (ns)", iccd_delay)

    # MDSplus voltages
    v_c, v_a, v_m = get_shot_voltage(c, shot)
    meta["Compression Voltage (kV)"] = v_c
    meta["Acceleration Voltage (kV)"] = v_a
    meta["Middle Voltage (kV)"] = v_m
    meta["Electrode Copy"] = electrode_copy

    return img, meta

def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    """
    Compute the MD5 checksum of a file.

    Used to detect whether a raw file’s content has changed (or to uniquely identify it).
    """
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def append_manifest_row(row: dict):
    """
    Append a row to the manifest CSV.
    Enforces shot_id is int in-memory and writes it as digits to CSV.
    """
    if "shot_id" not in row:
        raise KeyError("append_manifest_row: missing 'shot_id'")

    try:
        row_shot = int(row["shot_id"])
    except Exception as e:
        raise ValueError(f"append_manifest_row: shot_id must be int-castable, got {row['shot_id']!r}") from e

    # Ensure canonical in-memory form
    row = dict(row)
    row["shot_id"] = row_shot

    # CSV writer will stringify; that's fine on disk.
    write_header = not MANIFEST.exists()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["shot_id", "raw_path", "raw_md5", "H", "W"])
        if write_header:
            w.writeheader()
        # ensure H/W are ints if provided
        if "H" in row:
            row["H"] = int(row["H"])
        if "W" in row:
            row["W"] = int(row["W"])
        w.writerow(row)

def ensure_group_attrs(grp, meta: dict):
    """
    Add group attributes from meta dict to an HDF5 group, but only if the key does not already exist.

    (This is a conservative 'write-once' behavior for group attrs.)
    """
    for k, v in meta.items():
        try:
            if k not in grp.attrs:
                grp.attrs[k] = v
        except Exception:
            pass  # skip non-serializable


def open_h5(h5_path: Path):
    """
    Open (or create) the HDF5 file at h5_path in append mode.

    Ensures a top-level group 'meta' exists with a couple file-level attributes.
    Returns an h5py.File handle; caller is responsible for closing it.
    """
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(h5_path, "a")
    if "meta" not in f:
        meta = f.create_group("meta")
        meta.attrs["converter_version"] = "v1"
        meta.attrs["n_fibers_nominal"] = N_FIBERS
    return f



def write_fiber_dataset(f, shot_id: int, img, meta, raw_md5: str):
    """
    Write an image for a shot into HDF5 under group shots/<shot_id>/image.
    shot_id is strictly int; HDF5 group key is digits only.
    """
    if not isinstance(shot_id, int):
        raise TypeError(f"write_fiber_dataset: shot_id must be int, got {type(shot_id).__name__}")

    key = str(shot_id)
    grp = f.require_group(f"shots/{key}")

    img = np.asarray(img, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D image, but got shape {img.shape}")
    H, W = int(img.shape[0]), int(img.shape[1])

    chunks = (min(256, H), min(256, W))
    dname = "image"
    if dname in grp:
        try:
            del grp[dname]
        except Exception as e:
            raise RuntimeError(f"Failed to remove existing dataset {grp.name}/{dname}: {e}")

    dset = grp.create_dataset(
        name=dname,
        data=img,
        dtype="float32",
        compression="gzip",
        compression_opts=4,
        chunks=chunks,
        shuffle=True,
        fletcher32=True,
    )
    dset.attrs["raw_md5"] = raw_md5

    # Serialize metadata cleanly
    for k, v in meta.items():
        try:
            if isinstance(v, np.ndarray):
                val = v.tolist()
            else:
                val = v

            if isinstance(val, (str, bytes, int, float, bool, np.integer, np.floating)):
                dset.attrs[k] = val
            else:
                dset.attrs[k] = json.dumps(val)
        except Exception:
            dset.attrs[k] = str(v)

def edit_shot_metadata(shot_id: int | None = None):
    """
    Interactive editor for shot metadata stored in OUT_H5 under shots/<shot_id>.
    shot_id is strictly int.
    """
    if shot_id is None:
        s = input("Enter shot id (integer, e.g. 250219017): ").strip()
        try:
            shot_id = int(s)
        except Exception:
            print("Invalid shot id. Must be an integer.")
            return
    else:
        if not isinstance(shot_id, int):
            try:
                shot_id = int(shot_id)
            except Exception:
                print("Invalid shot id. Must be an integer.")
                return

    group_key = str(shot_id)

    with h5py.File(OUT_H5, "a") as f:
        grp = f.get(f"shots/{group_key}")
        if grp is None:
            print(f"Group shots/{group_key} not found in {OUT_H5}")
            return

        dset = grp.get("image")
        if dset is None:
            dset = next((grp[name] for name in grp if isinstance(grp[name], h5py.Dataset)), None)

        def print_attrs(target, name):
            print(f"\n{name} attributes:")
            for k, v in target.attrs.items():
                val = v.decode() if isinstance(v, (bytes, bytearray)) else v
                try:
                    if isinstance(val, str):
                        val = json.loads(val)
                except Exception:
                    pass
                print(f"  {k}: {val}")

        print(f"Editing metadata for shots/{group_key}")
        print_attrs(grp, "Group")
        if dset is not None:
            print_attrs(dset, "Dataset")
        else:
            print("No dataset found under group.")

        import ast
        while True:
            tgt = input("\nTarget to edit: (g)roup, (d)ataset, (q)uit: ").strip().lower()
            if tgt == "q":
                break
            if tgt not in ("g", "d"):
                print("Choose 'g', 'd' or 'q'.")
                continue
            target_obj = grp if tgt == "g" else dset
            if target_obj is None:
                print("Dataset not available; choose group instead.")
                continue

            action = input("Action: (l)ist, (e)dit/add, (r)emove, (q)back: ").strip().lower()
            if action == "q":
                continue
            if action == "l":
                print_attrs(target_obj, "Group" if target_obj is grp else "Dataset")
                continue
            if action == "r":
                name = input("Attribute name to remove: ").strip()
                if name in target_obj.attrs:
                    try:
                        del target_obj.attrs[name]
                        print(f"Removed attribute '{name}'.")
                    except Exception as e:
                        print(f"Failed to remove attribute: {e}")
                else:
                    print("Attribute not present.")
                continue
            if action == "e":
                name = input("Attribute name to set/edit: ").strip()
                cur = target_obj.attrs.get(name, None)
                cur_display = cur.decode() if isinstance(cur, (bytes, bytearray)) else cur
                print(f"Current value: {cur_display!r}")
                new_str = input("New value (Python literal for lists/dicts/numbers, or plain text): ").strip()
                try:
                    new_val = ast.literal_eval(new_str)
                except Exception:
                    new_val = new_str

                try:
                    store_val = new_val.tolist() if isinstance(new_val, np.ndarray) else new_val
                    if isinstance(store_val, (list, dict)):
                        target_obj.attrs[name] = json.dumps(store_val)
                    else:
                        target_obj.attrs[name] = store_val
                    print(f"Set attribute '{name}' -> {store_val!r}")
                except Exception as e:
                    try:
                        target_obj.attrs[name] = str(new_val)
                        print(f"Set attribute '{name}' as string fallback.")
                    except Exception as ee:
                        print(f"Failed to set attribute: {e} / {ee}")
                continue

            print("Unknown action. Use l/e/r/q.")

    print("Done editing.")


def edit_shot_metadata(shot_id: int | None = None):
    """
    Interactive editor for shot metadata stored in OUT_H5 under shots/<shot_id>.
    shot_id is strictly int. (No tuple/list/underscore formats.)
    """
    if shot_id is None:
        s = input("Enter shot id (integer, e.g. 250219017): ").strip()
        try:
            shot_id = int(s)
        except Exception:
            print("Invalid shot id. Must be an integer.")
            return
    else:
        if not isinstance(shot_id, int):
            try:
                shot_id = int(shot_id)
            except Exception:
                print("Invalid shot id. Must be an integer.")
                return

    group_key = str(shot_id)

    with h5py.File(OUT_H5, "a") as f:
        grp = f.get(f"shots/{group_key}")
        if grp is None:
            print(f"Group shots/{group_key} not found in {OUT_H5}")
            return

        dset = grp.get("image")
        if dset is None:
            dset = next((grp[name] for name in grp if isinstance(grp[name], h5py.Dataset)), None)

        def print_attrs(target, name):
            print(f"\n{name} attributes:")
            for k, v in target.attrs.items():
                val = v.decode() if isinstance(v, (bytes, bytearray)) else v
                try:
                    if isinstance(val, str):
                        val = json.loads(val)
                except Exception:
                    pass
                print(f"  {k}: {val}")

        print(f"Editing metadata for shots/{group_key}")
        print_attrs(grp, "Group")
        if dset is not None:
            print_attrs(dset, "Dataset")
        else:
            print("No dataset found under group.")

        import ast
        while True:
            tgt = input("\nTarget to edit: (g)roup, (d)ataset, (q)uit: ").strip().lower()
            if tgt == "q":
                break
            if tgt not in ("g", "d"):
                print("Choose 'g', 'd' or 'q'.")
                continue
            target_obj = grp if tgt == "g" else dset
            if target_obj is None:
                print("Dataset not available; choose group instead.")
                continue

            action = input("Action: (l)ist, (e)dit/add, (r)emove, (q)back: ").strip().lower()
            if action == "q":
                continue
            if action == "l":
                print_attrs(target_obj, "Group" if target_obj is grp else "Dataset")
                continue
            if action == "r":
                name = input("Attribute name to remove: ").strip()
                if name in target_obj.attrs:
                    try:
                        del target_obj.attrs[name]
                        print(f"Removed attribute '{name}'.")
                    except Exception as e:
                        print(f"Failed to remove attribute: {e}")
                else:
                    print("Attribute not present.")
                continue
            if action == "e":
                name = input("Attribute name to set/edit: ").strip()
                cur = target_obj.attrs.get(name, None)
                cur_display = cur.decode() if isinstance(cur, (bytes, bytearray)) else cur
                print(f"Current value: {cur_display!r}")
                new_str = input("New value (Python literal for lists/dicts/numbers, or plain text): ").strip()
                try:
                    new_val = ast.literal_eval(new_str)
                except Exception:
                    new_val = new_str

                try:
                    store_val = new_val.tolist() if isinstance(new_val, np.ndarray) else new_val
                    if isinstance(store_val, (list, dict)):
                        target_obj.attrs[name] = json.dumps(store_val)
                    else:
                        target_obj.attrs[name] = store_val
                    print(f"Set attribute '{name}' -> {store_val!r}")
                except Exception as e:
                    try:
                        target_obj.attrs[name] = str(new_val)
                        print("Set attribute as string fallback.")
                    except Exception as ee:
                        print(f"Failed to set attribute: {e} / {ee}")
                continue

            print("Unknown action. Use l/e/r/q.")

    print("Done editing.")


def bulk_edit_shot_metadata():
    """
    Bulk edit attributes for multiple shots in the HDF5.

    - Lists available shot groups under /shots
    - User selects shots by index/range or 'all'
    - User enters attributes to apply to group/dataset/both
    - Applies edits across the chosen shots
    """
    import ast

    def parse_indices(sel, names):
        sel = sel.strip()
        if not sel:
            return []
        if sel.lower() in ("all", "*"):
            return list(names)

        picks = []
        for part in sel.split(","):
            p = part.strip()
            if not p:
                continue
            if re.match(r"^\d+(-\d+)?$", p):
                if "-" in p:
                    a, b = map(int, p.split("-", 1))
                    for i in range(a, b + 1):
                        if 1 <= i <= len(names):
                            picks.append(names[i - 1])
                else:
                    i = int(p)
                    if 1 <= i <= len(names):
                        picks.append(names[i - 1])
            else:
                # explicit group key; enforce digits-only for int shot ids
                if p.isdigit() and p in names:
                    picks.append(p)

        seen = set()
        out = []
        for g in picks:
            if g not in seen:
                seen.add(g)
                out.append(g)
        return out

    with h5py.File(OUT_H5, "a") as f:
        if "shots" not in f:
            print("No shots group found in HDF5.")
            return
        shot_names = sorted(list(f["shots"].keys()))
        if not shot_names:
            print("No shots available.")
            return

        print("\nAvailable shots:")
        for i, name in enumerate(shot_names, start=1):
            print(f"  {i:3d}. {name}")

        sel = input("\nSelect shots (indices e.g. 1,3,5-7 or 'all'): ").strip()
        chosen = parse_indices(sel, shot_names)
        if not chosen:
            print("No shots selected. Aborting.")
            return

        print(f"\nSelected {len(chosen)} shots:")
        for g in chosen:
            print("  ", g)

        edits = []  # (attr_name, value, target)
        print("\nEnter attributes to set for all selected shots. Blank name finishes.")
        while True:
            name = input("\nAttribute name (blank to finish): ").strip()
            if not name:
                break
            tgt = input("Target: (g)roup, (d)ataset, (b)oth [g]: ").strip().lower() or "g"
            if tgt not in ("g", "d", "b"):
                print("Invalid target; use g/d/b. Skipping attribute.")
                continue
            val_str = input("Value (Python literal for numbers/lists/dicts; else text): ").strip()
            try:
                val = ast.literal_eval(val_str)
            except Exception:
                val = val_str
            edits.append((name, val, tgt))

        if not edits:
            print("No edits provided. Aborting.")
            return

        print("\nPlanned edits:")
        for name, val, tgt in edits:
            print(f"  {name} -> {val!r}  (target={tgt})")
        ok = input("\nApply to selected shots? (y/N): ").strip().lower()
        if ok != "y":
            print("Aborted.")
            return

        for g in chosen:
            grp = f.get(f"shots/{g}")
            if grp is None:
                print(f"Warning: shots/{g} not found, skipping.")
                continue

            dset = grp.get("image")
            if dset is None:
                dset = next((grp[name] for name in grp if isinstance(grp[name], h5py.Dataset)), None)

            for name, val, tgt in edits:
                targets = []
                if tgt in ("g", "b"):
                    targets.append(grp)
                if tgt in ("d", "b") and dset is not None:
                    targets.append(dset)
                elif tgt in ("d", "b") and dset is None:
                    print(f"  Warning: no dataset for {g}; skipping dataset target for {name}.")
                    continue

                for obj in targets:
                    try:
                        store_val = val.tolist() if isinstance(val, np.ndarray) else val
                        if isinstance(store_val, (str, bytes, int, float, bool, np.integer, np.floating)):
                            obj.attrs[name] = store_val
                        else:
                            obj.attrs[name] = json.dumps(store_val)
                    except Exception as e:
                        try:
                            obj.attrs[name] = str(val)
                        except Exception:
                            print(f"  Failed to set {name} on shots/{g}: {e}")

            print(f"Applied edits to shots/{g}")

    print("Bulk edit complete.")

def _decode_attr_value(v: Any) -> Any:
    """Decode HDF5 attr values: bytes -> str, JSON strings -> python objects."""
    # bytes -> str
    if isinstance(v, (bytes, bytearray)):
        v = v.decode(errors="replace")

    # numpy scalar -> python scalar
    if isinstance(v, np.generic):
        v = v.item()

    # try JSON decode if it's a string
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v

    return v
    
def get_shot_attributes(h5_path: Path, shot_id: int) -> dict:
    """
    Return group attrs, dataset attrs, and the actual image array
    for a given shot_id.
    """

    group_key = str(shot_id)
    group_path = f"shots/{group_key}"

    with h5py.File(h5_path, "r") as hf:
        grp = hf.get(group_path)
        if grp is None:
            raise KeyError(f"Group {group_path} not found in {h5_path}")

        # ---- group attrs
        group_attrs = {
            k: _decode_attr_value(v)
            for k, v in grp.attrs.items()
        }

        # ---- dataset
        dset = grp.get("image")
        if dset is None:
            raise KeyError(f"No dataset 'image' under {group_path}")

        dataset_attrs = {
            k: _decode_attr_value(v)
            for k, v in dset.attrs.items()
        }

        # ---- ACTUAL IMAGE (1024 x 1024)
        image = dset[()]

        image_info = {
            "shape": tuple(image.shape),
            "dtype": str(image.dtype),
        }

        return {
            "shot_id": shot_id,
            "group_attrs": group_attrs,
            "dataset_attrs": dataset_attrs,
            "image_info": image_info,
            "image": image,
        }

