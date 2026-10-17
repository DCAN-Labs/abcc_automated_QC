from pathlib import Path
import tarfile
import re
import tempfile
from code.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------
# FreeSurfer LUT-based subcortical labels (CANONICAL)
# ---------------------------------------------------------------------

SUBCORTICAL_LABELS = {
    4:  "Left-Lateral-Ventricle",
    5:  "Left-Inf-Lat-Vent",
    10: "Left-Thalamus",
    11: "Left-Caudate",
    12: "Left-Putamen",
    13: "Left-Pallidum",
    14: "3rd-Ventricle",
    15: "4th-Ventricle",
    16: "Brain-Stem",
    17: "Left-Hippocampus",
    18: "Left-Amygdala",
    19: "Left-Insula",
    26: "Left-Accumbens-area",
    27: "Left-Substancia-Nigra",
    43: "Right-Lateral-Ventricle",
    44: "Right-Inf-Lat-Vent",
    49: "Right-Thalamus",
    50: "Right-Caudate",
    51: "Right-Putamen",
    52: "Right-Pallidum",
    53: "Right-Hippocampus",
    54: "Right-Amygdala",
    55: "Right-Insula",
    58: "Right-Accumbens-area",
    59: "Right-Substancia-Nigra",
    72: "5th-Ventricle",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _extract_tar(tar_path: Path, out_dir: Path):
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=out_dir)


def _find_aseg_stats(extracted_dir: Path) -> Path:
    candidates = list(extracted_dir.rglob("aseg.stats"))
    if not candidates:
        raise FileNotFoundError(f"No aseg.stats found in {extracted_dir}")
    return candidates[0]


def _parse_aseg_stats_by_id(stats_path: Path) -> dict:
    """
    Parse aseg.stats using FreeSurfer numeric structure IDs.

    Format line (typical):
    # Cols: Index SegId NVoxels Volume_mm3 StructureName
    """
    roi = {}

    pattern = re.compile(
        r"\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d\.]+)\s+(.+)"
    )

    with open(stats_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            m = pattern.match(line)
            if m:
                seg_id = int(m.group(2))
                volume = float(m.group(4))

                roi[seg_id] = volume

    return roi


# ---------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------

def extract_subcortical_metrics(freesurfer_tar_path: Path) -> dict:
    """
    Extract subcortical volumes using LUT-based FreeSurfer aseg.stats.
    
    Returns:
        dict: {roi_name: volume_mm3}
    """

    freesurfer_tar_path = Path(freesurfer_tar_path)

    if not freesurfer_tar_path.exists():
        raise FileNotFoundError(f"Missing file: {freesurfer_tar_path}")

    logger.info(f"[FreeSurfer] Processing: {freesurfer_tar_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        _extract_tar(freesurfer_tar_path, tmpdir)
        stats_path = _find_aseg_stats(tmpdir)

        logger.info(f"[FreeSurfer] Found aseg.stats: {stats_path}")

        roi_by_id = _parse_aseg_stats_by_id(stats_path)

    metrics = {}

    for seg_id, name in SUBCORTICAL_LABELS.items():
        metrics[name] = roi_by_id.get(seg_id, 0.0)

    return metrics


# ---------------------------------------------------------------------
# Optional debug utility
# ---------------------------------------------------------------------

def extract_raw_by_id(freesurfer_tar_path: Path) -> dict:
    """
    Return raw {seg_id: volume_mm3} from aseg.stats.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        _extract_tar(freesurfer_tar_path, tmpdir)
        stats_path = _find_aseg_stats(tmpdir)

        return _parse_aseg_stats_by_id(stats_path)