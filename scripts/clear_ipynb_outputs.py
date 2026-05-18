import json
import sys
from pathlib import Path


def clear_outputs(nb: dict) -> dict:
    for cell in nb.get("cells", []):
        if isinstance(cell, dict) and cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return nb


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: clear_ipynb_outputs.py <notebook.ipynb>")
        return 2

    path = Path(sys.argv[1])
    nb = json.loads(path.read_text(encoding="utf-8"))
    nb = clear_outputs(nb)
    path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Cleared outputs: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
