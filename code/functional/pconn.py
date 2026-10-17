"""
Functional QC: Connectivity File Presence

Checks whether XCP-D produced:
- 5 min pconn
- 10 min pconn
"""

import re
from pathlib import Path
from code.logger import get_logger
logger = get_logger()


# ------------------------------------------------------------
# File existence helper
# ------------------------------------------------------------

def _has_file(func_dir: Path, pattern: str):
    return len(list(func_dir.glob(pattern))) > 0


def _parse_desc_volumes(path: Path):
    match = re.search(r"desc-(\d+)volumes", path.name)
    return int(match.group(1)) if match else None


# ------------------------------------------------------------
# Main API
# ------------------------------------------------------------

def check_pconn_presence(func_dir: Path, task: str, atlas: str):
    """
    Check for 5min / 10min pconn outputs.

    Returns
    -------
    dict
        { filename: {"QC_pass_fail": "PASS"/"FAIL"}, ... }
        Keyed by the actual pconn filename when one is found; falls back
        to an expected-filename placeholder when no matching file exists
        on disk (since there's nothing real to key by in that case).
    """

    results = {}

    all_files = list(func_dir.glob(
        f"*task-{task}*atlas-{atlas}*desc-*volumes*.pconn.nii*"
    ))
    logger.info(f"[pconn] Found {len(all_files)} pconn files for task-{task} atlas-{atlas}")

    # -----------------------------
    # 5 min
    # Between 300 and 740 volumes (inclusive upper bound) is treated
    # as the 5-minute pconn output.
    # -----------------------------
    file_5min = next(
        (p for p in all_files
         if (vol := _parse_desc_volumes(p)) is not None and 300 < vol <= 740),
        None
    )
    #extract subject session from func_dir
    subject = str(func_dir).split("/")[-3]
    session = str(func_dir).split("/")[-2]

    key_5min = f"xcp_d_v26.1.1/{subject}/{session}/func/{file_5min.name}" if file_5min else f"xcp_d_v26.1.1/{subject}/{session}/func/{subject}_{session}_task-{task}_space-fsLR_atlas-{atlas}_den-91k_stat-pearsoncorrelation_desc-374volumes_boldmap.pconn.nii"
    results[key_5min] = {"QC_pass_fail": "PASS" if file_5min else "FAIL"}

    # -----------------------------
    # 10 min
    # Anything above 740 volumes is treated as the 10-minute pconn output.
    # -----------------------------
    file_10min = next(
        (p for p in all_files
         if (vol := _parse_desc_volumes(p)) is not None and vol > 740),
        None
    )

    key_10min = f"xcp_d_v26.1.1/{subject}/{session}/func/{file_10min.name}" if file_10min else f"xcp_d_v26.1.1/{subject}/{session}/func/{subject}_{session}_task-{task}_space-fsLR_atlas-{atlas}_den-91k_stat-pearsoncorrelation_desc-749volumes_boldmap.pconn.nii"
    results[key_10min] = {"QC_pass_fail": "PASS" if file_10min else "FAIL"}

    return results