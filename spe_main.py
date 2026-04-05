from pathlib import Path
import time

from drive_source import DriveSource
import lightfield_processing as lf
import MDSplus as mds

DRIVE_ROOT = Path(r"G:\Shared drives\Shumlak Lab\Diagnostics\Spectroscopy\S_XB\Data\Ops")
H5_PATH    = Path(r"G:\Shared drives\Shumlak Lab\Users\Current\Elyse Lian\Python Code\Spectroscopy\experiment.h5")

POLL_SECONDS = 10


def scan_once(src: DriveSource, c, ingested: dict[int, dict]):
    files_scanned = 0
    files_already_ingested = 0
    new_files_ingested = 0
    errors = 0

    for spe in src.iter_spe_files():
        files_scanned += 1
        spe_path = spe.path

        # parse shot id
        shot_id = lf.parse_shot_id(spe_path)
        if shot_id is None:
            continue

        # skip if already in manifest
        if shot_id in ingested:
            files_already_ingested += 1
            continue

        try:
            img, meta = lf.read_spe(spe_path, c, shot_id)

            raw_md5 = lf.md5sum(spe_path)

            with lf.open_h5(H5_PATH) as h5f:
                lf.write_fiber_dataset(
                    h5f,
                    shot_id=shot_id,
                    img=img,
                    meta=meta,
                    raw_md5=raw_md5,
                )

            H, W = int(img.shape[0]), int(img.shape[1])

            row = {
                "shot_id": shot_id,
                "raw_path": str(spe_path).replace("\\", "/"),
                "raw_md5": raw_md5,
                "H": H,
                "W": W,
            }

            lf.append_manifest_row(row)

            # update in-memory manifest
            ingested[shot_id] = row
            new_files_ingested += 1

        except Exception as e:
            print(f"[ERROR] {spe_path}: {e}")
            errors += 1
            continue

    return {
        "files_scanned": files_scanned,
        "files_already_ingested": files_already_ingested,
        "new_files_ingested": new_files_ingested,
        "errors": errors,
    }


def main():
    c = mds.Connection("172.25.35.142")
    try:
        c.get("getenv('zaphd_path')")
    except Exception:
        pass

    src = DriveSource(DRIVE_ROOT, recursive=True)

    ingested = lf.load_manifest()

    summary = scan_once(src, c, ingested)

    print("PROCESSING SUMMARY:")
    print(f"  Files scanned: {summary['files_scanned']}")
    print(f"  Files already ingested: {summary['files_already_ingested']}")
    print(f"  New files ingested: {summary['new_files_ingested']}")
    print(f"  Files with errors: {summary['errors']}")
    print(f"  Total in manifest (in-memory): {len(ingested)}")

    time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()


