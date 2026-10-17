"""
Functional Connectivity QC (ABCC)

Implements:
- Label-to-label edge-wise ±3σ outlier detection
- Cohort normalization per label-to-label connection, per atlas
- Sparse memory-safe aggregation
- Robust QC file mapping and metric naming

Outputs:
- rest_pconn_out_5min_<atlas>
- rest_pconn_out_10min_<atlas>

Reference:
  Legacy Metric: Number of functional connectivity label-to-label edges in
  each subject whose values exceed ±3 standard deviations from the
  corresponding cohort mean for that label-to-label connection.

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
        sub-*/ses-*/*_desc-abcc-automated-qc.tsv

    Args:
        relmat_file (Path): Path to relmat.tsv file

    Returns:
        Path: Path to QC output TSV
    """
    relmat_path = Path(relmat_file)

    subject = relmat_path.name.split("sub-")[1].split("_")[0]
    session = relmat_path.name.split("ses-")[1].split("_")[0]

    # Move from func/ directory to the session directory
    updated_path = relmat_path.parent.parent

    # QC file lives in the session directory with fixed naming
    qc_file = (
        updated_path
        / f"sub-{subject}_ses-{session}_desc-abcc-automated-qc.tsv"
    )

    return qc_file


# =====================================================================
# Load connectivity matrix
# =====================================================================

def load_relmat(file: Path):
    """
    Load an XCP-D connectivity matrix while preserving row/column labels.

    The TSV is expected to contain:
        - labels as the first column (row labels)
        - labels as the column names
        - a square connectivity matrix

    Only the upper triangle (excluding the diagonal) is returned, so each
    label-to-label connection is represented exactly once.

    Args:
        file (Path): Path to a relmat.tsv file

    Returns:
        tuple:
            edge_names (list[tuple[str, str]]):
                Label pairs corresponding to each edge value.
            edge_values (np.ndarray):
                1D vector of upper-triangular connectivity values.

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If the matrix cannot be parsed or is not square
    """
    if not file.exists():
        raise FileNotFoundError(f"Connectivity file not found: {file}")

    try:
        # The first column contains the row labels.
        df = pd.read_csv(file, sep="\t", index_col=0)

        if df.empty:
            raise ValueError(f"No connectivity data in {file}")

        # Preserve the labels before converting the matrix to numeric values.
        row_labels = df.index.astype(str).tolist()
        col_labels = df.columns.astype(str).tolist()

        # Connectivity values should be numeric.
        df = df.apply(pd.to_numeric, errors="coerce")

        if df.shape[0] != df.shape[1]:
            raise ValueError(
                f"Expected square connectivity matrix, got shape "
                f"{df.shape} in {file}"
            )

        if row_labels != col_labels:
            raise ValueError(
                f"Row and column labels do not match in {file}"
            )

        mat = df.values

        # Unique connections only: upper triangle, excluding diagonal.
        tri_indices = np.triu_indices_from(mat, k=1)

        edge_values = mat[tri_indices]

        edge_names = [
            (row_labels[i], col_labels[j])
            for i, j in zip(*tri_indices)
        ]

        logger.debug(
            f"Loaded {len(edge_values)} label-to-label connections "
            f"from {file.name}"
        )

        return edge_names, edge_values

    except Exception as e:
        logger.error(f"Failed to load {file}: {e}")
        raise


def get_n_regions(file: Path):
    """
    Determine the number of parcels/regions in a connectivity matrix.

    This is the dimension of the square matrix (rows == columns == regions),
    NOT the number of subjects/files in the cohort.

    Args:
        file (Path): Path to a relmat.tsv file for the atlas of interest

    Returns:
        int: Number of regions (ROIs) in the atlas's parcellation
    """
    if not file.exists():
        raise FileNotFoundError(f"Connectivity file not found: {file}")

    try:
        df = pd.read_csv(file, sep="\t", index_col=0)

        if df.empty:
            raise ValueError(f"No connectivity data in {file}")

        n_rows, n_cols = df.shape

        if n_rows != n_cols:
            raise ValueError(
                f"Expected square connectivity matrix, got shape "
                f"{df.shape} in {file}"
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
    Compute cohort-wide mean and std separately for every label-to-label
    connection, without storing the full cohort matrix in memory.

    For example, all subjects' values for:
        Label_A -> Label_B

    are accumulated together and receive their own mean/std.

    The resulting arrays use a consistent label-pair ordering established
    from the first valid file. Subsequent files are mapped to that ordering
    using their label pairs, rather than relying only on matrix position.

    NaN values are excluded on a per-edge basis.

    Args:
        files (list): List of relmat file paths for one atlas/duration.
        atlas (str, optional): Atlas name for sanity checking.

    Returns:
        dict:
            {
                "edge_names": list of label pairs,
                "mean": np.ndarray,
                "std": np.ndarray,
                "valid_count": np.ndarray,
                "n_subjects": int,
            }

    Raises:
        ValueError: If no valid connectivity files are found.
    """
    logger.info(
        "       Building cohort statistics for each label-to-label connection"
    )

    edge_names = None
    edge_to_index = None

    sum_edges = None
    sumsq_edges = None
    valid_count = None

    count = 0
    failed_files = []
    files_with_nans = 0

    for f in files:

        if atlas is not None and extract_atlas_from_path(f) != atlas:
            logger.debug(
                f"Skipping file {f}: atlas mismatch (expected {atlas})"
            )
            continue

        try:
            current_edge_names, vec = load_relmat(f)

            # First valid file establishes the canonical label-pair ordering.
            if edge_names is None:
                edge_names = current_edge_names
                edge_to_index = {
                    edge: i for i, edge in enumerate(edge_names)
                }

                n_edges = len(edge_names)

                sum_edges = np.zeros(n_edges, dtype=np.float64)
                sumsq_edges = np.zeros(n_edges, dtype=np.float64)
                valid_count = np.zeros(n_edges, dtype=np.int64)

            # Map this file's connections to the canonical ordering.
            if len(current_edge_names) != len(edge_names):
                raise ValueError(
                    f"Number of connections does not match expected number "
                    f"({len(current_edge_names)} vs {len(edge_names)})"
                )

            try:
                reorder = np.asarray(
                    [edge_to_index[edge] for edge in current_edge_names],
                    dtype=np.int64,
                )
            except KeyError as e:
                raise ValueError(
                    f"Connection {e.args[0]} is missing from the cohort "
                    f"connection set"
                )

            # Put this file's values into the canonical edge order.
            ordered_vec = np.empty_like(vec)
            ordered_vec[reorder] = vec

            nan_mask = np.isnan(ordered_vec)

            if nan_mask.any():
                files_with_nans += 1
                logger.debug(
                    f"{f.name}: {int(nan_mask.sum())} NaN connections "
                    f"excluded from cohort statistics"
                )

            valid_mask = ~nan_mask

            sum_edges[valid_mask] += ordered_vec[valid_mask]
            sumsq_edges[valid_mask] += ordered_vec[valid_mask] ** 2
            valid_count[valid_mask] += 1

            count += 1

        except Exception as e:
            logger.warning(f"Skipping {f}: {e}")
            failed_files.append(str(f))
            continue

    if count == 0:
        raise ValueError("No valid connectivity files found")

    if failed_files:
        logger.warning(
            f"Skipped {len(failed_files)} files due to errors"
        )

    if files_with_nans:
        logger.warning(
            f"{files_with_nans}/{count} files contained NaN connection "
            f"values (excluded per-connection, not per-subject)"
        )

    # Protect against division by zero for connections with no valid data.
    safe_valid_count = np.maximum(valid_count, 1)

    mean = sum_edges / safe_valid_count

    # Preserve the original implementation's population standard deviation:
    # variance = E[X^2] - E[X]^2
    var = (
        sumsq_edges / safe_valid_count
        - mean ** 2
    )

    # Floating-point roundoff can produce tiny negative variances.
    var = np.maximum(var, 0)

    std = np.sqrt(var)

    # Connections with no valid observations have no cohort statistics.
    no_data_mask = valid_count == 0
    mean[no_data_mask] = np.nan
    std[no_data_mask] = np.nan

    logger.info(
        f"      Cohort statistics computed from {count} subjects/files "
        f"      across {len(edge_names)} label-to-label connections"
    )

    logger.info(
        f"      Connection means: {np.nanpercentile(mean, [25, 50, 75])}"
    )
    logger.info(
        f"      Connection stds: {np.nanpercentile(std, [25, 50, 75])}"
    )

    return {
        "edge_names": edge_names,
        "mean": mean,
        "std": std,
        "valid_count": valid_count,
        "n_subjects": count,
    }


# =====================================================================
# Subject-level outlier detection
# =====================================================================

def compute_edge_outliers(file: Path, stats, sigma=3):
    """
    Count label-to-label connections that are outliers for one subject/file.

    IMPORTANT:
        The cohort statistics are calculated independently for each
        label-to-label connection.

        For each connection in this subject:
            z = abs(subject_value - cohort_mean_for_that_connection)
                / cohort_std_for_that_connection

        The subject's outlier count is the number of label-to-label
        connections where z > sigma.

    Thus:
        rest_pconn_out_<duration>_<atlas> = N

    where N is the number of label-to-label connections that are outliers
    within that subject/file.

    Connections with:
        - NaN subject values
        - NaN cohort mean/std
        - zero cohort standard deviation

    are not counted as outliers.

    Args:
        file (Path): Subject's relmat file.
        stats (dict): Output from accumulate_edges().
        sigma (float): Number of standard deviations for the threshold.

    Returns:
        int: Number of outlier label-to-label connections in this subject.
    """
    try:
        edge_names, vec = load_relmat(file)

        cohort_edge_names = stats["edge_names"]
        cohort_mean = stats["mean"]
        cohort_std = stats["std"]

        if len(edge_names) != len(cohort_edge_names):
            raise ValueError(
                f"Number of connections in {file} does not match cohort stats"
            )

        # Match this subject's connections to the cohort statistics by
        # label pair, not merely by array position.
        cohort_edge_to_index = {
            edge: i for i, edge in enumerate(cohort_edge_names)
        }

        try:
            reorder = np.asarray(
                [cohort_edge_to_index[edge] for edge in edge_names],
                dtype=np.int64,
            )
        except KeyError as e:
            raise ValueError(
                f"Connection {e.args[0]} is missing from cohort statistics"
            )

        # Reorder this subject's values into the cohort-statistics order.
        ordered_vec = np.empty_like(vec)
        ordered_vec[reorder] = vec

        # Only connections with a real value and a positive cohort std
        # can be evaluated for outlier status.
        valid = (
            ~np.isnan(ordered_vec)
            & ~np.isnan(cohort_mean)
            & ~np.isnan(cohort_std)
            & (cohort_std > 0)
        )

        z_scores = np.full(
            len(ordered_vec),
            np.nan,
            dtype=np.float64
        )

        z_scores[valid] = np.abs(
            (ordered_vec[valid] - cohort_mean[valid])
            / cohort_std[valid]
        )

        outlier_mask = z_scores > sigma

        # This is N = number of outlier label-to-label connections
        # for THIS subject/file.
        n_outliers = int(np.sum(outlier_mask))

        logger.debug(
            f"{file.name}: {n_outliers} outlier label-to-label connections"
        )

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


def append_qc_metric(qc_file, metric_name, metric_value, edge_names):
    """
    Append or update a metric in the QC TSV file.

    Args:
        qc_file (Path): Path to QC output file
        metric_name (str): Metric name
        metric_value: Number of outlier measurements
        edge_names: List of label-to-label connection names

    Returns:
        None
    """
    try:
        # Load or initialize
        df = initialize_qc_file(qc_file)

        # Make sure required columns exist
        if "outlier_count" not in df.columns:
            df["outlier_count"] = 888.0

        if "total_measurements(N)" not in df.columns:
            df["total_measurements(N)"] = 888.0

        # Check if this metric already exists
        metric_exists = df["Filename"].eq(metric_name).any()

        if metric_exists:
            # Only update the values for this existing metric row
            df.loc[df["Filename"] == metric_name, "outlier_count"] = metric_value
            df.loc[
                df["Filename"] == metric_name,
                "total_measurements(N)"
            ] = float(len(edge_names))

        else:
            # Add a new metric row
            new_row = pd.DataFrame({
                "Filename": [metric_name],
                "outlier_count": [metric_value],
                "total_measurements(N)": [float(len(edge_names))]
            })

            df = pd.concat([df, new_row], ignore_index=True)

        # Write back
        df.to_csv(qc_file, sep="\t", index=False)

        logger.debug(
            f"Updated {metric_name}: "
            f"outlier_count={metric_value}, "
            f"total_measurements(N)={len(edge_names)} "
            f"in {qc_file.name}"
        )

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
        stats (dict): Cohort statistics containing per-connection mean/std.
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
            n_outliers = compute_edge_outliers(f, stats)
            file_name = str(f).split("/")[-1]
            file_name=file_name.replace('_relmat.tsv', '_boldmap.pconn.nii')
            file_name=file_name.replace('_stat-', '_den-91k_stat-')
            subject=file_name.split('_')[0]
            session=file_name.split('_')[1]
            metric_name = f"xcp_d_v26.1.1/{subject}/{session}/func/{file_name}"

            qc_file = get_qc_output_path(f)
            append_qc_metric(qc_file, metric_name, n_outliers, stats["edge_names"])

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
      2. Computes atlas-specific cohort-wide mean/std for every label-to-label connection
      3. Determines the number of regions (ROIs) from the atlas's own
         parcellation dimension (NOT the number of subject files)
      4. For each subject, counts label-to-label connections exceeding ±3σ from their corresponding cohort statistics
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
    for task in config.tasks:
        logger.info(f"Task:{task}")

        # -----------------------------------------------
        # Find connectivity files (all atlases)
        # -----------------------------------------------

        all_files_5min = sorted(root.glob(
            f"sub-*/ses-*/func/*task-{task}*desc-374*volumes*relmat.tsv"
        ))

        all_files_10min = sorted(root.glob(
            f"sub-*/ses-*/func/*task-{task}*desc-749volumes*relmat.tsv"
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
                    logger.info("   Computing 5-minute cohort statistics...")
                    stats_5min = accumulate_edges(files_5min, atlas)
                    # print("stats_5min:", stats_5min)
                    stats_5min["atlas"] = atlas
                    stats_5min["n_regions"] = get_n_regions(files_5min[0])
                    logger.info(f"      Atlas: {atlas}, no. of label to lable conncection: {len(stats_5min['edge_names'])}, regions: {stats_5min['n_regions']}")
                except Exception as e:
                    logger.error(f"  Failed to compute 5-minute stats for {atlas}: {e}")
                    stats_5min = {"mean": None, "std": None, "atlas": atlas, "n_regions": None}

            # --- 10-minute cohort stats ---
            if files_10min:
                try:
                    logger.info("  Computing 10-minute cohort statistics...")
                    stats_10min = accumulate_edges(files_10min, atlas)
                    stats_10min["atlas"] = atlas
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
                "task":task,
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