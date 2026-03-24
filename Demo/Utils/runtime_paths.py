from pathlib import Path


DEMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DEMO_ROOT.parent
ETC_ROOT = REPO_ROOT / "ETC"


def demo_path(*parts: str) -> Path:
    return DEMO_ROOT.joinpath(*parts)


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def etc_path(*parts: str) -> Path:
    return ETC_ROOT.joinpath(*parts)
