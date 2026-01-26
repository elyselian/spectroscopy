from pathlib import Path
import hashlib, csv, re, sys
import numpy as np
import h5py
import spe_loader as sl
import json

RAW_DIR   = [
    Path("/Users/elyselian/Library/CloudStorage/GoogleDrive-elyse16@uw.edu/Shared drives/Shumlak Lab/Diagnostics/Spectroscopy/S_XB/Data/Ops/250219"),
    Path("/Users/elyselian/Library/CloudStorage/GoogleDrive-elyse16@uw.edu/Shared drives/Shumlak Lab/Diagnostics/Spectroscopy/S_XB/Data/Ops/250416"),
    Path("/Users/elyselian/Library/CloudStorage/GoogleDrive-elyse16@uw.edu/Shared drives/Shumlak Lab/Diagnostics/Spectroscopy/S_XB/Data/Ops/250423"),
    Path("/Users/elyselian/Library/CloudStorage/GoogleDrive-elyse16@uw.edu/Shared drives/Shumlak Lab/Diagnostics/Spectroscopy/S_XB/Data/Ops/250424"),
    Path("/Users/elyselian/Library/CloudStorage/GoogleDrive-elyse16@uw.edu/Shared drives/Shumlak Lab/Diagnostics/Spectroscopy/S_XB/Data/Ops/250428"),
    Path("/Users/elyselian/Library/CloudStorage/GoogleDrive-elyse16@uw.edu/Shared drives/Shumlak Lab/Diagnostics/Spectroscopy/S_XB/Data/Ops/250514"),
    Path("/Users/elyselian/Library/CloudStorage/GoogleDrive-elyse16@uw.edu/Shared drives/Shumlak Lab/Diagnostics/Spectroscopy/S_XB/Data/Ops/250609"),
    Path("/Users/elyselian/Library/CloudStorage/GoogleDrive-elyse16@uw.edu/Shared drives/Shumlak Lab/Diagnostics/Spectroscopy/S_XB/Data/Ops/250616"),
    Path("/Users/elyselian/Library/CloudStorage/GoogleDrive-elyse16@uw.edu/Shared drives/Shumlak Lab/Diagnostics/Spectroscopy/S_XB/Data/Ops/250617"),
    Path("/Users/elyselian/Library/CloudStorage/GoogleDrive-elyse16@uw.edu/Shared drives/Shumlak Lab/Diagnostics/Spectroscopy/S_XB/Data/Ops/250618"),
    Path("/Users/elyselian/Library/CloudStorage/GoogleDrive-elyse16@uw.edu/Shared drives/Shumlak Lab/Diagnostics/Spectroscopy/S_XB/Data/Ops/250620"),
    Path("/Users/elyselian/Library/CloudStorage/GoogleDrive-elyse16@uw.edu/Shared drives/Shumlak Lab/Diagnostics/Spectroscopy/S_XB/Data/Ops/250623"),
    Path("/Users/elyselian/Library/CloudStorage/GoogleDrive-elyse16@uw.edu/Shared drives/Shumlak Lab/Diagnostics/Spectroscopy/S_XB/Data/Ops/250625"),
    Path("/Users/elyselian/Library/CloudStorage/GoogleDrive-elyse16@uw.edu/Shared drives/Shumlak Lab/Diagnostics/Spectroscopy/S_XB/Data/Ops/250714"),
    Path("/Users/elyselian/Library/CloudStorage/GoogleDrive-elyse16@uw.edu/Shared drives/Shumlak Lab/Diagnostics/Spectroscopy/S_XB/Data/Ops/250728")
]

OUT_H5    = Path("curated/experiment.h5")
MANIFEST  = Path("curated/manifest.csv")
N_FIBERS  = 20

def read_spe(path, lamb=229.7, gate=100, 
             v_c=9, v_a=7, electrode_copy='A', iccd_delay=100):
    f = sl.load_from_files([str(path)])
    img = np.asarray(f.data[0][0], dtype=np.float32)

    meta = {}
    try:
        ft = f.footer  # XML footer (untangle object)
    except Exception as e:
        ft = None
        print(f"Unable to access footer: {e}")

    # Try to extract known metadata from the footer if present
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
                    ft.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Cameras.Camera.Gating.RepetitiveGate.Pulse['delay']
                ]).astype(float)
                gate_width = np.array([
                    ft.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Cameras.Camera.Gating.RepetitiveGate.Pulse['width']
                ]).astype(float)
                meta["ICCD Delay (ns)"] = float(gate_delay[0])
                meta["Gate Width (ns)"]  = float(gate_width[0])
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

            # Wavelength calibration (array) if available on object
            try:
                wave_cal = np.asarray(f.wavelength)
                meta["Wavelength Calibration (nm)"] = wave_cal.tolist()
            except Exception:
                pass

        except Exception as e:
            print(f"Error while extracting footer metadata: {e}. Falling back to manual values.")
    else:
        print("Footer is empty or missing metadata. Using manual assignments for gate/lambda.")

    # Ensure manual defaults for missing keys
    meta.setdefault("Gate Width (ns)", gate)
    meta.setdefault("Central Wavelength (nm)", lamb)
    meta.setdefault("ICCD Delay (ns)", iccd_delay)

    meta["Compression Voltage (kV)"] = v_c
    meta["Acceleration Voltage (kV)"] = v_a
    meta["Electrode Copy"] = electrode_copy

    return img, meta

def parse_shot_id(path: Path):
    s = path.stem.lower()
    if "-raw" in s:
        return None
    m = re.search(r"(25\d{4})\s+(\d+)", s)
    return (m.group(1), m.group(2)) if m else None

def md5sum(path: Path, chunk=1<<20):
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def load_manifest():
    rows = {}
    if MANIFEST.exists():
        with MANIFEST.open() as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                # The shot_id in CSV is stored as string representation of tuple
                # So we use it as-is for comparison
                rows[r["shot_id"]] = r
    return rows

def normalize_shot_id_for_comparison(shot_id):
    """Convert shot_id to consistent string format for manifest comparison"""
    if isinstance(shot_id, (list, tuple)):
        return str(shot_id)  # Creates "('250219', '17')" format
    else:
        return str(shot_id)

def normalize_shot_id_for_hdf5(shot_id):
    """Convert shot_id to HDF5 group key format"""
    if isinstance(shot_id, (list, tuple)):
        return "_".join(map(str, shot_id))  # Creates "250219_17" format
    else:
        return str(shot_id)

def clear_manifest(clear_h5: bool = False):
    # remove manifest if present
    try:
        if MANIFEST.exists():
            MANIFEST.unlink()
            print(f"Removed manifest: {MANIFEST}")
    except Exception as e:
        print(f"Failed to remove manifest: {e}")

    # recreate empty manifest with header
    try:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        with MANIFEST.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["shot_id", "raw_path", "raw_md5", "H", "W"])
            w.writeheader()
        print(f"Created empty manifest with header: {MANIFEST}")
    except Exception as e:
        print(f"Failed to create empty manifest: {e}")

    if clear_h5:
        try:
            # opening with 'w' truncates/creates the file
            h5py.File(OUT_H5, "w").close()
            print(f"Truncated/created HDF5: {OUT_H5}")
        except Exception as e:
            print(f"Failed to truncate/create HDF5 ({OUT_H5}): {e}")


def append_manifest_row(row):
    write_header = not MANIFEST.exists()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["shot_id","raw_path","raw_md5","H","W"])
        if write_header:
            w.writeheader()
        w.writerow(row)

def ensure_group_attrs(grp, meta: dict):
    for k, v in meta.items():
        # HDF5 attrs want basic types (str, float, int, np.*)
        try:
            if k not in grp.attrs:
                grp.attrs[k] = v
        except Exception:
            pass  # skip non-serializable

def open_h5():
    OUT_H5.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(OUT_H5, "a")  # append mode
    # Set file-level meta once
    if "meta" not in f:
        meta = f.create_group("meta")
        meta.attrs["converter_version"] = "v1"
        meta.attrs["n_fibers_nominal"]  = N_FIBERS
    return f

def write_fiber_dataset(f, shot_id, img, meta, raw_md5):
    """
    Write an image for a shot into the HDF5 under group shots/<shot_key>/image.
    If an 'image' dataset already exists for that shot it is removed first so
    the function can be re-run without raising "name already exists".
    """
    # normalize shot_id (tuple -> safe string)
    if isinstance(shot_id, (list, tuple)):
        key = "_".join(map(str, shot_id))
    else:
        key = str(shot_id)

    grp = f.require_group(f"shots/{key}")

    # ensure array and shape
    img = np.asarray(img, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D image, but got shape {img.shape}")
    H, W = int(img.shape[0]), int(img.shape[1])

    # choose chunk size and remove existing dataset if present
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

    # write metadata as dataset attrs (serialize non-native types)
    for k, v in meta.items():
        try:
            # numpy arrays -> lists
            if isinstance(v, np.ndarray):
                val = v.tolist()
            else:
                val = v

            # native scalar types: write directly
            if isinstance(val, (str, bytes, int, float, bool, np.integer, np.floating)):
                dset.attrs[k] = val
            else:
                # fallback: JSON-serialize (lists, dicts, etc.)
                dset.attrs[k] = json.dumps(val)
        except Exception:
            # last-resort: convert to string
            dset.attrs[k] = str(v)

def edit_shot_metadata(shot_key: str | tuple | None = None):
    """
    Interactive editor for shot metadata stored in OUT_H5 under shots/<group_key>.
    If shot_key is None the user is prompted. Accepts:
      - "250219_17"
      - "250219 17"
      - "('250219','17')" or "['250219','17']"
    You can list, add, edit, or delete group-level and dataset-level attributes.
    Changes are written immediately to the HDF5 file.
    """
    import ast

    # Resolve group key
    if shot_key is None:
        s = input("Enter shot id (e.g. 250219_17 or ('250219','17')): ").strip()
    else:
        s = str(shot_key)

    try:
        parsed = ast.literal_eval(s)
    except Exception:
        parsed = s

    if isinstance(parsed, (list, tuple)):
        group_key = "_".join(map(str, parsed))
    else:
        group_key = str(parsed).replace(" ", "_")

    with h5py.File(OUT_H5, "a") as f:
        grp = f.get(f"shots/{group_key}")
        if grp is None:
            print(f"Group shots/{group_key} not found in {OUT_H5}")
            return

        # pick dataset (prefer 'image')
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
                new_str = input("New value (enter as Python literal for lists/dicts/numbers, or plain text): ").strip()
                # try to parse python literal
                try:
                    new_val = ast.literal_eval(new_str)
                except Exception:
                    new_val = new_str

                # serialize non-scalar to JSON / lists
                try:
                    if isinstance(new_val, np.ndarray):
                        store_val = new_val.tolist()
                    else:
                        store_val = new_val

                    if isinstance(store_val, (list, dict, tuple)):
                        target_obj.attrs[name] = json.dumps(store_val)
                    else:
                        target_obj.attrs[name] = store_val
                    print(f"Set attribute '{name}' -> {store_val!r}")
                except Exception as e:
                    # fallback to string
                    try:
                        target_obj.attrs[name] = str(new_val)
                        print(f"Set attribute '{name}' as string fallback.")
                    except Exception as ee:
                        print(f"Failed to set attribute: {e} / {ee}")
                continue

            print("Unknown action. Use l/e/r/q.")
    print("Done editing.")

def bulk_edit_shot_metadata():
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
            # numeric index or range (1-based)
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
                # treat as explicit group key
                picks.append(p)
        # dedupe while preserving order
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

        # print list
        print("\nAvailable shots:")
        for i, name in enumerate(shot_names, start=1):
            print(f"  {i:3d}. {name}")

        sel = input("\nSelect shots (indices e.g. 1,3,5-7 or 'all' or names comma-separated): ").strip()
        chosen = parse_indices(sel, shot_names)
        if not chosen:
            print("No shots selected. Aborting.")
            return

        print(f"\nSelected {len(chosen)} shots:")
        for g in chosen:
            print("  ", g)

        # collect edits
        edits = []  # list of tuples (attr_name, value, target) target in {'g','d','b'}
        print("\nNow enter attributes to set for all selected shots.")
        print("Leave attribute name blank to finish.")
        while True:
            name = input("\nAttribute name (blank to finish): ").strip()
            if not name:
                break
            tgt = input("Target: (g)roup, (d)ataset, (b)oth [g]: ").strip().lower() or "g"
            if tgt not in ("g", "d", "b"):
                print("Invalid target; use g/d/b. Skipping attribute.")
                continue
            val_str = input("Value (enter Python literal for numbers, lists, dicts; else text): ").strip()
            try:
                val = ast.literal_eval(val_str)
            except Exception:
                val = val_str
            edits.append((name, val, tgt))

        if not edits:
            print("No edits provided. Aborting.")
            return

        # confirm
        print("\nPlanned edits:")
        for name, val, tgt in edits:
            print(f"  {name} -> {val!r}  (target={tgt})")
        ok = input("\nApply to selected shots? (y/N): ").strip().lower()
        if ok != "y":
            print("Aborted.")
            return

        # apply edits
        for g in chosen:
            grp = f.get(f"shots/{g}")
            if grp is None:
                print(f"Warning: shots/{g} not found, skipping.")
                continue
            # pick dataset
            dset = grp.get("image")
            if dset is None:
                dset = next((grp[name] for name in grp if isinstance(grp[name], h5py.Dataset)), None)

            for name, val, tgt in edits:
                targets = []
                if tgt in ("g", "b"):
                    targets.append(("group", grp))
                if tgt in ("d", "b"):
                    if dset is None:
                        print(f"  Warning: no dataset for {g}; skipping dataset target for {name}.")
                    else:
                        targets.append(("dataset", dset))
                for tname, obj in targets:
                    try:
                        store_val = val
                        if isinstance(store_val, np.ndarray):
                            store_val = store_val.tolist()
                        # native simple types
                        if isinstance(store_val, (str, bytes, int, float, bool, np.integer, np.floating)):
                            obj.attrs[name] = store_val
                        else:
                            # lists/dicts/tuples -> JSON string
                            obj.attrs[name] = json.dumps(store_val)
                    except Exception as e:
                        try:
                            obj.attrs[name] = str(val)
                        except Exception:
                            print(f"  Failed to set {name} on {g}/{tname}: {e}")
            print(f"Applied edits to shots/{g}")

    print("Bulk edit complete.")