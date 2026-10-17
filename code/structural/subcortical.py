"""
Structural QC Module (ABCC)

Implements:
1. Subcortical segmentation volume extraction from fMRIPrep *_dseg.nii.gz
2. ROI volume computation (mm³)
3. Standardized ROI dictionary output for cohort aggregation

NOTE:
- Cohort-level outlier detection is handled separately in cohort.py
- This module only extracts raw structural features
"""

from pathlib import Path
import numpy as np
import nibabel as nib
from code.logger import get_logger
logger = get_logger()


# ---------------------------------------------------------------------
# 22 Subcortical + ventricular + brainstem regions (FreeSurfer-style)
# ---------------------------------------------------------------------

SUBCORTICAL_LABELS = {
    2: "Left-Cerebral-White-Matter",
    3: "Left-Cerebral-Cortex",
    4: "Left-Lateral-Ventricle",
    5: "Left-Inf-Lat-Vent",
    7: "Left-Cerebellum-White-Matter",
    8: "Left-Cerebellum-Cortex",
    10: "Left-Thalamus",
    11: "Left-Caudate",
    12: "Left-Putamen",
    13: "Left-Pallidum",
    14: "3rd-Ventricle",
    15: "4th-Ventricle",
    16: "Brain-Stem",
    17: "Left-Hippocampus",
    18: "Left-Amygdala",
    24: "CSF",
    26: "Left-Accumbens-area",
    28: "Left-VentralDC",
    41: "Right-Cerebral-White-Matter",
    42: "Right-Cerebral-Cortex",
    43: "Right-Lateral-Ventricle",
    44: "Right-Inf-Lat-Vent",
    46: "Right-Cerebellum-White-Matter",
    47: "Right-Cerebellum-Cortex",
    49: "Right-Thalamus",
    50: "Right-Caudate",
    51: "Right-Putamen",
    52: "Right-Pallidum",
    53: "Right-Hippocampus",
    54: "Right-Amygdala",
    58: "Right-Accumbens-area",
    60: "Right-VentralDC",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_dseg(dseg_path: Path):
    """
    Load segmentation image.
    """
    img = nib.load(str(dseg_path))
    data = img.get_fdata()
    voxel_volume = np.prod(img.header.get_zooms()[:3])
    return data, voxel_volume


def _find_dseg(anat_dir: Path) -> Path:
    """
    Locate fMRIPrep dseg file.
    """
    candidates = sorted(anat_dir.glob("*_desc-aparcaseg_dseg.nii*"))

    if not candidates:
        candidates = sorted(anat_dir.glob("*_dseg.nii*"))

    if not candidates:
        raise FileNotFoundError(f"No dseg found in {anat_dir}")

    return candidates[0]


# ---------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------

def extract_subcortical_metrics(anat_dir: Path):
    """
    Extract subcortical ROI volumes.

    Returns
    -------
    dict
        {
            "Left-Thalamus": volume_mm3,
            ...
        }
    """

    anat_dir = Path(anat_dir)

    dseg_path = _find_dseg(anat_dir)

    logger.info(f"[Subcortical] Loading: {dseg_path}")

    data, voxel_vol = _load_dseg(dseg_path)

    metrics = {}

    for label, name in SUBCORTICAL_LABELS.items():

        vox_count = np.count_nonzero(data == label)

        metrics[name] = float(vox_count * voxel_vol)

    return metrics


# ---------------------------------------------------------------------
# Optional utility (useful for debugging QC)
# ---------------------------------------------------------------------

def extract_voxel_counts(anat_dir: Path):
    """
    Same as above but returns voxel counts instead of volumes.
    """

    anat_dir = Path(anat_dir)

    dseg_path = _find_dseg(anat_dir)

    data, _ = _load_dseg(dseg_path)

    counts = {}

    for label, name in SUBCORTICAL_LABELS.items():

        counts[name] = int(np.count_nonzero(data == label))

    return counts