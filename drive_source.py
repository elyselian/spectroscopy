from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class LocalSPE:
    path: Path


class DriveSource:
    def __init__(self, root: Path, recursive: bool = True):
        self.root = root.expanduser().resolve()
        self.recursive = recursive
        if not self.root.exists():
            raise FileNotFoundError(self.root)

    def iter_spe_files(self) -> Iterator[LocalSPE]:
        pattern = "**/*.spe" if self.recursive else "*.spe"
        for path in sorted(self.root.glob(pattern)):
            if path.is_file():
                yield LocalSPE(path)


