"""
Functional Connectivity QC (ABCC)

Implements:
- Edge-wise ±3σ outlier detection
- Cohort normalization per edge, per atlas
- Sparse memory-safe aggregation
- Robust QC file mapping and metric naming

Outputs:
- rest_pconn_out_5min_<atlas>
- rest_pconn_out_10min_<atlas>

Reference:
  Legacy Metric: Number of functional connectivity edges whose values exceed
  ±3 standard deviations from the cohort mean.

  XCP-D Mapping: Derived from connectivity matrices
  (*_task-<TASK>_space-fsLR_atlas-<ATLAS>_stat-pearsoncorrelation_desc-<374/749>volumes_relmat.tsv)
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
from code.logger import get_logger

logger = get_logger()


# =====================================================================
# Utilities
# =====================================================================

def extract_atlas_from_path(file_path):
    """
    Extract atlas name from relmat file path.

    Pattern: *_atlas-<ATLAS>_*relmat.tsv

    Args:
        file_path (Path or str): Path to relmat file

    Returns:
        str: Atlas name, e.g., 'Schaefer400', or None if not found
    """
    fname = str(file_path)
    match = re.search(r'_atlas-([a-zA-Z0-9]+)_', fname)
    return match.group(1) if match else None


def filter_files_by_atlas(files, atlas):
    """
    Keep only files whose BIDS filename matches the given atlas.

    Args:
        files (list): List of relmat file paths
        atlas (str): Atlas name to filter for (e.g., 'Schaefer400')

    Returns:
        list: Subset of files matching the atlas
    """
    matched = [f for f in files if extract_atlas_from_path(f) == atlas]
    skipped = len(files) - len(matched)
    if skipped:
        logger.debug(f"Filtered out {skipped} files not matching atlas={atlas}")
    return matched


def get_qc_output_path(relmat_file):
    """
    Map relmat file path to corresponding QC output location.

    BIDS structure:
        sub-*/ses-*/func/*_relmat.tsv
        sub-*/ses-*/func/*_desc-abcc-automated-qc.tsv

    Args:
        relmat_file (Path): Path to relmat.tsv file

    Returns:
        Path: Path to QC output TSV
    """
    relmat_path = Path(relmat_file)

    # QC file lives in same directory with fixed naming
    qc_file = relmat_path.parent / f"{relmat_path.stem.split('_relmat')[0]}_desc-abcc-automated-qc.tsv"

    return qc_file


# =====================================================================
# Load connectivity matrix
# =====================================================================

def load_relmat(file: Path):
    """
    Load XCP-D relmat.tsv (upper triangle matrix).

    The file contains a square symmetric connectivity matrix.
    We extract only the upper triangle (excluding diagonal) to get
    unique edge values.

    Args:
        file (Path): Path to relmat.tsv file

    Returns:
        np.ndarray: 1D vector of upper-triangular edge values

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file cannot be parsed
    """
    if not file.exists():
        raise FileNotFoundError(f"Connectivity file not found: {file}")

    try:
        df = pd.read_csv(file, sep="\t")

        # Drop non-numeric columns if present (e.g., index columns)
        df = df.select_dtypes(include=[np.number])

        if df.empty:
            raise ValueError(f"No numeric data in {file}")

        mat = df.values

        # Extract upper triangle (excluding diagonal)
        # This gives us unique edges only
        tri_indices = np.triu_indices_from(mat, k=1)
        edge_vector = mat[tri_indices]

        logger.info(f"Loaded {len(edge_vector)} edges from {file.name}")
        return edge_vector

    except Exception as e:
        logger.error(f"Failed to load {file}: {e}")
        raise


def get_n_regions(file: Path):
    """
    Determine the number of parcels/regions (ROIs) in a connectivity matrix.

    This is the dimension of the square matrix (rows == columns == regions),
    NOT the number of subjects/files in the cohort.

    Args:
        file (Path): Path to a relmat.tsv file for the atlas of interest

    Returns:
        int: Number of regions (ROIs) in the atlas's parcellation

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If the matrix is not square or has no numeric data
    """
    if not file.exists():
        raise FileNotFoundError(f"Connectivity file not found: {file}")

    try:
        df = pd.read_csv(file, sep="\t")
        df = df.select_dtypes(include=[np.number])

        if df.empty:
            raise ValueError(f"No numeric data in {file}")

        n_rows, n_cols = df.shape
        if n_rows != n_cols:
            raise ValueError(
                f"Expected square connectivity matrix, got shape {df.shape} in {file}"
            )

        return n_rows

    except Exception as e:
        logger.error(f"Failed to determine region count for {file}: {e}")
        raise


# =====================================================================
# Build cohort edge statistics (streaming)
# =====================================================================

def accumulate_edges(files, atlas=None):
    """
    Compute cohort-wide mean and std WITHOUT storing full matrix in memory.

    Uses Welford-style streaming accumulation (sum / sum-of-squares) for
    memory efficiency across large cohorts.

    Args:
        files (list): List of relmat file paths, already filtered to a
            single atlas (see filter_files_by_atlas)
        atlas (str, optional): Atlas name, used only for sanity-check
            logging/skipping in case unfiltered files slip through

    Returns:
        tuple: (mean, std) - 1D arrays of edge statistics

    Raises:
        ValueError: If no valid files provided or no edges found
    """
    logger.info("Building cohort edge statistics (streaming)")

    n_edges = None
    sum_edges = None
    sumsq_edges = None
    valid_count = None  # per-edge count of non-NaN contributions
    count = 0
    failed_files = []
    files_with_nans = 0

    for f in files:
        if atlas is not None and extract_atlas_from_path(f) != atlas:
            logger.debug(f"Skipping file {f}: atlas mismatch (expected {atlas})")
            continue

        try:
            vec = load_relmat(f)

            if n_edges is None:
                n_edges = len(vec)
                sum_edges = np.zeros(n_edges, dtype=np.float64)
                sumsq_edges = np.zeros(n_edges, dtype=np.float64)
                valid_count = np.zeros(n_edges, dtype=np.int64)
            elif len(vec) != n_edges:
                logger.warning(
                    f"Skipping {f}: edge count {len(vec)} does not match expected {n_edges}"
                )
                failed_files.append(str(f))
                continue

            # A single NaN edge (e.g. a low-signal/dropped ROI pair) must not
            # poison every other edge's running sum. Accumulate NaN-aware,
            # per-edge, so each edge's mean/std reflects only the subjects
            # that actually had a value for that edge.
            nan_mask = np.isnan(vec)
            if nan_mask.any():
                files_with_nans += 1
                logger.info(f"{f.name}: {int(nan_mask.sum())} NaN edges excluded from accumulation")

            valid_mask = ~nan_mask
            sum_edges[valid_mask] += vec[valid_mask]
            sumsq_edges[valid_mask] += vec[valid_mask] ** 2
            valid_count[valid_mask] += 1
            count += 1

        except Exception as e:
            logger.warning(f"Skipping {f}: {e}")
            failed_files.append(str(f))
            continue

    if count == 0:
        raise ValueError("No valid connectivity files found")

    if failed_files:
        logger.warning(f"Skipped {len(failed_files)} files due to errors")

    if files_with_nans:
        logger.warning(
            f"{files_with_nans}/{count} files contained NaN edge values "
            f"(excluded per-edge, not per-subject)"
        )

    # Edges with zero valid contributions would otherwise divide by zero;
    # guard the division, then mark those edges explicitly as NaN below
    # so downstream code can tell "no data" apart from "computed as zero".
    safe_valid_count = np.maximum(valid_count, 1)
    mean = sum_edges / safe_valid_count

    # Variance: E[X²] - (E[X])²
    var = (sumsq_edges / safe_valid_count) - (mean ** 2)

    # Ensure numerical stability (avoid sqrt of negative due to floating point errors)
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)

    no_data_mask = valid_count == 0
    if no_data_mask.any():
        logger.warning(
            f"{int(no_data_mask.sum())} edges had NO valid (non-NaN) data across "
            f"all {count} subjects; marking as NaN rather than 0"
        )
        mean[no_data_mask] = np.nan
        std[no_data_mask] = np.nan

    logger.info(f"Cohort statistics computed from {count} subjects, {n_edges} edges")
    logger.info(f"Edge means: {np.nanpercentile(mean, [25, 50, 75])}")
    logger.info(f"Edge stds: {np.nanpercentile(std, [25, 50, 75])}")

    return mean, std


# =====================================================================
# Subject-level outlier detection
# =====================================================================

def compute_edge_outliers(file: Path, mean, std, sigma=3):
    """
    Count edges exceeding ±3σ from cohort mean.

    For each edge:
        z = |value - mean| / std
        outlier if z > sigma

    Args:
        file (Path): Subject's relmat file
        mean (np.ndarray): Cohort mean for each edge
        std (np.ndarray): Cohort std for each edge
        sigma (float): Number of standard deviations threshold (default: 3)

    Returns:
        int: Number of outlier edges in this subject

    Raises:
        FileNotFoundError: If file does not exist
    """
    try:
        vec = load_relmat(file)

        # Z-score calculation
        z_scores = np.abs((vec - mean) / std)

        # Count outliers
        n_outliers = int(np.sum(z_scores > sigma))

        logger.debug(f"{file.name}: {n_outliers} outlier edges")
        return n_outliers

    except Exception as e:
        logger.error(f"Failed to compute outliers for {file}: {e}")
        raise


# =====================================================================
# QC file I/O
# =====================================================================

def initialize_qc_file(qc_file):
    """
    Create or ensure QC file exists with proper structure.

    Args:
        qc_file (Path): Path to QC output file

    Returns:
        pd.DataFrame: Existing or empty DataFrame
    """
    if qc_file.exists():
        try:
            df = pd.read_csv(qc_file, sep="\t")
            logger.debug(f"Loaded existing QC file: {qc_file.name}")
            return df
        except Exception as e:
            logger.warning(f"Could not read existing QC file {qc_file}: {e}")
            return pd.DataFrame()
    else:
        logger.debug(f"Creating new QC file: {qc_file.name}")
        qc_file.parent.mkdir(parents=True, exist_ok=True)
        return pd.DataFrame()


def append_qc_metric(qc_file, metric_name, metric_value):
    """
    Append or update a metric in the QC TSV file.

    Args:
        qc_file (Path): Path to QC output file
        metric_name (str): Metric name
        metric_value: Metric value
    """
    try:
        # Load or initialize
        df = initialize_qc_file(qc_file)

        # Add or update metric
        new_row = pd.DataFrame({
            "Metric": [metric_name],
            "Value": [metric_value]
        })

        # Check if metric already exists
        df = df[df["Metric"] != metric_name]
        df = pd.concat([df, new_row], ignore_index=True)

        # Write back
        df.to_csv(qc_file, sep="\t", index=False)
        logger.debug(f"Appended {metric_name}={metric_value} to {qc_file.name}")

    except Exception as e:
        logger.error(f"Failed to update QC file {qc_file}: {e}")
        raise


# =====================================================================
# Per-atlas / per-duration processing helper
# =====================================================================

def process_files(files, stats, duration_label):
    """
    Process a set of connectivity files (already filtered to one atlas
    and one duration) and write QC metrics for each subject.

    Args:
        files (list): Relmat files for this atlas/duration
        stats (dict): {"mean", "std", "atlas", "n_regions", "n_subjects"}
        duration_label (str): "5min" or "10min"

    Returns:
        int: Number of files successfully processed
    """
    if stats["mean"] is None:
        logger.warning(f"Skipping {duration_label} processing (no cohort stats)")
        return 0

    processed = 0
    for f in files:
        try:
            n_outliers = compute_edge_outliers(f, stats["mean"], stats["std"])

            metric_name = f"rest_pconn_out_{duration_label}_{stats['atlas']}"

            qc_file = get_qc_output_path(f)
            append_qc_metric(qc_file, metric_name, n_outliers)

            processed += 1

        except Exception as e:
            logger.error(f"Failed to process {f}: {e}")
            continue

    return processed


# =====================================================================
# Main pipeline
# =====================================================================

def run_connectivity_qc(root: Path, config=None):
    """
    Full cohort functional connectivity QC pipeline, run independently
    for each atlas in config.atlas.

    For each atlas:
      1. Filters the 5-minute and 10-minute connectivity matrices to that atlas
      2. Computes atlas-specific cohort-wide edge statistics (mean, std)
      3. Determines the number of regions (ROIs) from the atlas's own
         parcellation dimension (NOT the number of subject files)
      4. For each subject, counts edges exceeding ±3σ
      5. Writes metrics to subject-level QC files

    Outputs to: *_desc-abcc-automated-qc.tsv
    Metrics: rest_pconn_out_5min_<ATLAS>, rest_pconn_out_10min_<ATLAS>

    Args:
        root (Path): BIDS root directory
        config (dict-like): Must expose `.atlas`, an iterable of atlas names

    Returns:
        dict: Per-atlas summary statistics, e.g.
            {
              "Schaefer400": {
                  "files_5min": ..., "files_10min": ...,
                  "processed_5min": ..., "processed_10min": ...,
                  "n_regions_5min": ..., "n_regions_10min": ...,
              },
              ...
            }
    """
    logger.info("=" * 70)
    logger.info("Starting functional connectivity QC pipeline")
    logger.info("=" * 70)

    root = Path(root)

    # -----------------------------------------------
    # Find connectivity files (all atlases)
    # -----------------------------------------------

    all_files_5min = sorted(root.glob(
        "sub-*/ses-*/func/*desc-374volumes*relmat.tsv"
    ))

    all_files_10min = sorted(root.glob(
        "sub-*/ses-*/func/*desc-749volumes*relmat.tsv"
    ))

    logger.info(f"Found {len(all_files_5min)} 5-minute connectivity files (all atlases)")
    logger.info(f"Found {len(all_files_10min)} 10-minute connectivity files (all atlases)")

    summary = {}

    # -----------------------------------------------
    # Run the full pipeline per atlas
    # -----------------------------------------------

    for atlas in config.atlases:
        logger.info("-" * 70)
        logger.info(f"Running for atlas: {atlas}")

        files_5min = filter_files_by_atlas(all_files_5min, atlas)
        files_10min = filter_files_by_atlas(all_files_10min, atlas)

        logger.info(f"  {len(files_5min)} 5-minute files, {len(files_10min)} 10-minute files")

        stats_5min = {"mean": None, "std": None, "atlas": atlas, "n_regions": None}
        stats_10min = {"mean": None, "std": None, "atlas": atlas, "n_regions": None}

        # --- 5-minute cohort stats ---
        if files_5min:
            try:
                logger.info("  Computing 5-minute cohort statistics...")
                stats_5min["mean"], stats_5min["std"] = accumulate_edges(files_5min, atlas)
                # n is the number of regions (ROIs) in this atlas's parcellation,
                # derived from the connectivity matrix dimension, not file count
                stats_5min["n_regions"] = get_n_regions(files_5min[0])
                logger.info(f"    Atlas: {atlas}, regions: {stats_5min['n_regions']}")
            except Exception as e:
                logger.error(f"  Failed to compute 5-minute stats for {atlas}: {e}")
                stats_5min = {"mean": None, "std": None, "atlas": atlas, "n_regions": None}

        # --- 10-minute cohort stats ---
        if files_10min:
            try:
                logger.info("  Computing 10-minute cohort statistics...")
                stats_10min["mean"], stats_10min["std"] = accumulate_edges(files_10min, atlas)
                stats_10min["n_regions"] = get_n_regions(files_10min[0])
                logger.info(f"    Atlas: {atlas}, regions: {stats_10min['n_regions']}")
            except Exception as e:
                logger.error(f"  Failed to compute 10-minute stats for {atlas}: {e}")
                stats_10min = {"mean": None, "std": None, "atlas": atlas, "n_regions": None}

        # --- Process subjects for this atlas ---
        processed_5min = 0
        processed_10min = 0

        if files_5min and stats_5min["mean"] is not None:
            logger.info(f"  Processing {len(files_5min)} 5-minute subjects for {atlas}...")
            processed_5min = process_files(files_5min, stats_5min, "5min")
            logger.info(f"  Processed {processed_5min}/{len(files_5min)} subjects")

        if files_10min and stats_10min["mean"] is not None:
            logger.info(f"  Processing {len(files_10min)} 10-minute subjects for {atlas}...")
            processed_10min = process_files(files_10min, stats_10min, "10min")
            logger.info(f"  Processed {processed_10min}/{len(files_10min)} subjects")

        summary[atlas] = {
            "files_5min": len(files_5min),
            "files_10min": len(files_10min),
            "processed_5min": processed_5min,
            "processed_10min": processed_10min,
            "n_regions_5min": stats_5min["n_regions"],
            "n_regions_10min": stats_10min["n_regions"],
        }

    logger.info("=" * 70)
    logger.info("Functional connectivity QC completed for all atlases")
    logger.info("=" * 70)

    return summary