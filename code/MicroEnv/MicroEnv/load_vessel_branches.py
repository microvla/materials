from copy import deepcopy
import json
from pathlib import Path
from typing import Dict, List, Union


BranchLayout = List[Dict]
BRANCH_LAYOUT_DIR = Path(__file__).resolve().parent / "branch_layouts"


def _layout_path(name_or_path: Union[str, Path]) -> Path:
    path = Path(name_or_path).expanduser()
    if path.exists():
        return path
    if path.suffix != ".json":
        path = path.with_suffix(".json")
    candidate = BRANCH_LAYOUT_DIR / path.name
    if candidate.exists():
        return candidate

    # Allow the old unnumbered layout name after files are renamed to NN_name.json.
    matches = sorted(BRANCH_LAYOUT_DIR.glob(f"[0-9][0-9]_{path.stem}.json"))
    if matches:
        return matches[0]
    return candidate


def list_branch_layouts(layout_dir: Union[str, Path] = BRANCH_LAYOUT_DIR) -> List[str]:
    path = Path(layout_dir)
    if not path.exists():
        return []
    return sorted(item.stem for item in path.glob("*.json") if not item.name.startswith("_") and item.name != "index.json")


def load_branch_layout(path: Union[str, Path]) -> BranchLayout:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    branches = data.get("branches", data)
    if not isinstance(branches, list):
        raise ValueError(f"Branch layout JSON must contain a branch list or a 'branches' field: {path}")
    return deepcopy(branches)


def get_branch_layout(name_or_path: Union[str, Path] = "default_2") -> BranchLayout:
    path = _layout_path(name_or_path)
    if not path.exists():
        options = ", ".join(list_branch_layouts())
        raise ValueError(f"Unknown branch layout '{name_or_path}'. Available JSON layouts: {options}")
    return load_branch_layout(path)
