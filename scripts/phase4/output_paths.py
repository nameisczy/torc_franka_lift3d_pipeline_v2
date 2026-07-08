from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_result_root() -> Path:
    raw_root = os.environ.get("TORC_RESULTS_ROOT") or os.environ.get("TORC_PHASE4_RESULTS_ROOT")
    base = Path(raw_root).expanduser() if raw_root else Path("/mnt/ssd/ziyaochen")
    if base.name == PROJECT_ROOT.name:
        return base
    return base / PROJECT_ROOT.name


RESULT_ROOT = get_result_root()


def result_path(*parts: str) -> Path:
    return RESULT_ROOT.joinpath(*parts)


def artifact_path(*parts: str) -> Path:
    return RESULT_ROOT.joinpath("phase4_artifacts", *parts)
