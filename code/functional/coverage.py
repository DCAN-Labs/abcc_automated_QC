"""
Functional QC: Parcel Coverage Module

Implements:
- Parcel-level coverage QC from XCP-D *_stat-coverage_bold.tsv
- PASS/FAIL brain coverage metric
- Percent parcels meeting threshold

Mapping:
- bc_<task>_run-<run> = PASS/FAIL
- bc_<task>_run-<run>_perc_vox = %
"""

from pathlib import Path
import pandas as pd
from code.logger import get_logger
logger = get_logger()


# ------------------------------------------------------------
# Core loader
# ------------------------------------------------------------

def _load_coverage_file(func_dir: Path, task: str = None, atlas: str = None):
    """
    Locate XCP-D coverage file.
    """
    files = list(func_dir.glob(f"*task-{task}_*_atlas-{atlas}*stat-coverage_bold.tsv"))

    if not files:
        raise FileNotFoundError(f"No coverage file in {func_dir}")

    return files


# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------

def compute_brain_coverage(func_dir: Path, task: str, atlas: str, threshold=0.90):
    """
    Compute parcel-based brain coverage QC.

    Parameters
    ----------
    func_dir : Path
    task : str
    atlas : str
    threshold : float

    Returns
    -------
    dict
        { filename: {"QC_pass_fail": "PASS"/"FAIL", "bc_perc_vox": float}, ... }
    """

    files = _load_coverage_file(func_dir, task, atlas)

    results = {}
    for f in files:
        logger.info(f"[Coverage] Loading {f.name}")

        df = pd.read_csv(f, sep="\t")

        # there is only 1 row with all the columns being the parcels
        values = df.iloc[0, :].astype(float)

        total = len(values)
        logger.info(f"Total parcels: {total}")

        good = (values >= threshold).sum()
        logger.info(f"Parcels meeting threshold of {threshold}: {good}")

        percent = (good / total) * 100

        pass_fail = "PASS" if percent >= 90 else "FAIL"
        
        subject=f.name.split('_')[0]
        session=f.name.split('_')[1]

        file_name=f"xcp_d_v26.1.1/{subject}/{session}/func/{f.name}"
        results[file_name] = {
            "QC_pass_fail": pass_fail,
            "bc_perc_vox": round(percent, 3),
        }

    return results
