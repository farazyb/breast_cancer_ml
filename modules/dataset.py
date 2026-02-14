from __future__ import annotations

from pathlib import Path
import hashlib
import requests
import pandas as pd
import matplotlib.pyplot as plt
import re
from googledrivedownloader import download_file_from_google_drive as gdd

# Ensure you add FIGURES_DIR to your modules/config.py!
# e.g., FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
from modules.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, FIGURES_DIR


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def _extract_gdrive_file_id(url: str) -> str:
    m = re.search(r"/file/d/([^/]+)", url)
    if m:
        return m.group(1)
    raise ValueError("Invalid Google Drive file URL. Expected /file/d/<FILE_ID>/...")

def download_dataset2(
    url: str,
    dest: Path = RAW_DATA_DIR / "dataset.csv",
    *,
    force: bool = False,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        return dest

    file_id = _extract_gdrive_file_id(url)

    gdd(
        file_id=file_id,
        dest_path=str(dest),
        unzip=False,   # dataset.csv, not a zip
    )
    return dest

def download_dataset(
    url: str,
    dest: Path = RAW_DATA_DIR / "dataset.csv",
    *,
    force: bool = False,
    timeout: int = 30,
    expected_sha256: str | None = None,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """
    Download dataset from url into RAW_DATA_DIR.
    - Writes atomically via .part then rename.
    - If force=False and dest exists, it won't download again.
    - Optional sha256 check for reproducibility.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        # If file exists and checksum is requested, still validate it.
        if expected_sha256 is not None:
            actual = _sha256_file(dest)
            if actual.lower() != expected_sha256.lower():
                raise ValueError(
                    f"Existing file checksum mismatch: {dest}\n"
                    f"expected: {expected_sha256}\n"
                    f"actual:   {actual}"
                )
        return dest

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()

        # sanity check: many broken links return HTML error pages
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "text/html" in ctype:
            raise ValueError(f"URL returned HTML, not a dataset. Content-Type={ctype}")

        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

        tmp.replace(dest)

    if expected_sha256 is not None:
        actual = _sha256_file(dest)
        if actual.lower() != expected_sha256.lower():
            dest.unlink(missing_ok=True)
            raise ValueError(
                f"Checksum mismatch after download: {dest}\n"
                f"expected: {expected_sha256}\n"
                f"actual:   {actual}"
            )

    return dest


def load(input_path: Path = RAW_DATA_DIR / "dataset.csv") -> pd.DataFrame:
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_info_columns', 1000)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {input_path}. "
            f"Call download_dataset(url) first or pass the correct path."
        )
    return pd.read_csv(input_path)


def write(
    df: pd.DataFrame,
    file_name: str = "dataset.csv",
    out_path: Path = PROCESSED_DATA_DIR,
) -> Path:
    out_path.mkdir(parents=True, exist_ok=True)
    target = out_path / file_name
    df.to_csv(target, encoding="utf-8", index=False, header=True)
    return target


def save_figure(
    fig_id: str,
    fig: plt.Figure | None = None,
    output_dir: Path = FIGURES_DIR,
    tight_layout: bool = True,
    dpi: int = 300,
    fmt: str = "png"
) -> Path:
    """
    Saves a matplotlib figure to the project reports/figures directory.
    
    Parameters:
    - fig_id: The filename (without extension), e.g., "correlation_heatmap".
    - fig: The figure object. If None, uses the current active figure (plt.gcf()).
    - output_dir: Path to save the figure (defaults to config.FIGURES_DIR).
    - tight_layout: Adjusts subplot params for a tight layout.
    - dpi: Resolution (dots per inch).
    - fmt: File format ('png', 'pdf', 'svg', etc.).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    path = output_dir / f"{fig_id}.{fmt}"
    
    target_fig = fig or plt.gcf()
    
    if tight_layout:
        target_fig.tight_layout()
        
    print(f"Saving figure: {fig_id} to {path}...")
    target_fig.savefig(path, format=fmt, dpi=dpi, bbox_inches='tight')
    
    return path