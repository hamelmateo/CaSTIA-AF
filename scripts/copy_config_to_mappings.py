"""
copy_config_to_mappings.py

Usage:
    python copy_config_to_mappings.py /path/to/source_config.json /root/search/directory

This script copies a .json configuration file into all subdirectories named 'cell-mapping'
within a given root directory.
"""

import shutil
import sys
from pathlib import Path

def copy_config_to_cell_mapping_dirs(source_config: Path, root_dir: Path, overwrite: bool = True) -> None:
    if not source_config.exists():
        print(f"[ERROR] Source file not found: {source_config}")
        return

    cell_mapping_dirs = [p for p in root_dir.rglob("cell-mapping") if p.is_dir()]
    print(f"Found {len(cell_mapping_dirs)} 'cell-mapping' folders.")

    for folder in cell_mapping_dirs:
        dest = folder / source_config.name
        if dest.exists() and not overwrite:
            print(f"[SKIP] {dest} already exists.")
            continue
        shutil.copy2(source_config, dest)
        print(f"[OK] Copied to {dest}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python copy_config_to_mappings.py /path/to/source_config.json /root/search/directory")
        return

    source = Path(sys.argv[1])
    root = Path(sys.argv[2])
    copy_config_to_cell_mapping_dirs(source, root)

if __name__ == "__main__":
    main()
