from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
from code.utils.update_qc_file import update_qc_file

from code.logger import get_logger
logger = get_logger()


# ---------------------------------------------------------------------
# Collect QC files
# ---------------------------------------------------------------------

def collect_qc_files(root: Path):
    logger.info(f"Collecting QC files from {root}")
    return sorted(root.glob("*desc-abcc-automated-qc*.tsv"))


# ---------------------------------------------------------------------
# Load ONLY specific files (memory efficient)
# ---------------------------------------------------------------------

def load_group_files(files, keyword):
    """
    Only keep files relevant to a group (subcortical / sulc / misc)
    """

    if keyword == "misc":
        return [f for f in files if "subcortical" not in f.name and "sulc" not in f.name]

    return [f for f in files if keyword in f.name]


# ---------------------------------------------------------------------
# Parse TSV → grouped metrics
# ---------------------------------------------------------------------

def load_metrics(file: Path):
    df = pd.read_csv(file, sep="\t")

    grouped = {}

    for _, row in df.iterrows():

        key = row["Filename"]
        val = row["Value"]

        # if "_" in key:
        #     prefix, subkey = key.split("_", 1)
        # else:
        #     prefix, subkey = "misc", key

        if key not in grouped:
            grouped[key] = {}

        grouped[key]= val

    return grouped


# ---------------------------------------------------------------------
# Build matrix for one group (streaming)
# ---------------------------------------------------------------------

def build_feature_matrix(files, group_name):
    """
    Builds subject × feature matrix for ONE group only
    """

    data = []
    index = []

    for f in files:

        grouped = load_metrics(f)
        # print("grouped:", grouped)
        metrics = grouped
        # print("metrics:", metrics)

        data.append(metrics)
        index.append(str(f))

    df = pd.DataFrame(data)
    df.index = index

    return df


# ---------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------

def compute_stats(df):
    return df.mean(numeric_only=True), df.std(numeric_only=True)


# ---------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------

def compute_outliers(df, means, stds, config):

    sigma = config.outlier_sigma

    outlier_counts = []

    for _, row in df.iterrows():

        count = 0

        for c in df.columns:

            val = row.get(c, np.nan)
            mu = means.get(c, np.nan)
            sd = stds.get(c, np.nan)

            if pd.isna(val) or pd.isna(mu) or pd.isna(sd) or sd == 0:
                continue

            if abs(val - mu) > sigma * sd:
                count += 1

        outlier_counts.append(count)

    return outlier_counts


# ---------------------------------------------------------------------
# MAIN PIPELINE (STREAMING)
# ---------------------------------------------------------------------

def run_cohort_qc(root: Path, config):

    logger.info("Starting STREAMING cohort QC")

    temp_folder = root / "tmp"
    files = collect_qc_files(temp_folder)
    logger.info(f"files: {len(files)}")

    updates = defaultdict(dict)

    # =========================================================
    # 1. SUBCORTICAL (load only subcortical files)
    # =========================================================

    sub_files = load_group_files(files, "subcortical")

    logger.info(f"Subcortical files: {len(sub_files)}")

    df_sub = build_feature_matrix(sub_files, "subcortical")
    print("df_subcortical:", df_sub)
    means_sub, stds_sub = compute_stats(df_sub)
    logger.info(f"mean_sub: {means_sub}, stds_sub: {stds_sub}")

    sub_counts = compute_outliers(df_sub, means_sub, stds_sub, config)
    for i, file in enumerate(sub_files):

        subject = file.name.split("sub-")[1].split("_")[0]
        session = file.name.split("ses-")[1].split("_")[0]

        updated_path = str(file).replace(
            "tmp/",
            f"sub-{subject}/ses-{session}/"
        )
        updated_path = str(updated_path).replace(
            f"_subcortical.tsv",
            f".tsv"
        )

        filename_sub=f"fmriprep_v25.1.4/sourcedata/freesurfer/sub-{subject}_ses-{session}/aseg.stats"
        updates[updated_path][filename_sub] = {"outlier_count": sub_counts[i],"total_measurements(N)": len(means_sub)}

    del df_sub  # free memory

    # =========================================================
    # 2. SULCAL
    # =========================================================

    sulc_files = load_group_files(files, "sulc")

    logger.info(f"Sulc files: {len(sulc_files)}")

    df_sulc = build_feature_matrix(sulc_files, "sulc")
    print("df_Sulc:", df_sulc)
    means_sulc, stds_sulc = compute_stats(df_sulc)
    logger.info(f"mean_sulc: {means_sulc}")
    logger.info(f"stds_sulc: {stds_sulc}")

    sulc_counts = compute_outliers(df_sulc, means_sulc, stds_sulc, config)
    for i, file in enumerate(sulc_files):

        subject = file.name.split("sub-")[1].split("_")[0]
        session = file.name.split("ses-")[1].split("_")[0]

        updated_path = str(file).replace(
            "tmp/",
            f"sub-{subject}/ses-{session}/"
        )
        updated_path = str(updated_path).replace(
            f"_sulc.tsv",
            f".tsv"
        )
        filename_sulc=f"xcp_d_v26.1.1/sub-{subject}/ses-{session}/anat/sub-{subject}_ses-{session}_rec-norm_run-01_space-fsLR_atlas-Gordon_stat-mean_desc-sulc_morph.tsv"
        updates[updated_path][filename_sulc] = {"outlier_count": sulc_counts[i],"total_measurements(N)": len(means_sulc)}

    del df_sulc  # free memory

    # # =========================================================
    # # 3. MISC (remaining files only)
    # # =========================================================

    # misc_files = load_group_files(files, "misc")

    # logger.info(f"Misc files: {len(misc_files)}")

    # df_misc = build_feature_matrix(misc_files, "misc")
    # means_misc, stds_misc = compute_stats(df_misc)

    # misc_counts = compute_outliers(df_misc, means_misc, stds_misc, config)

    # del df_misc  # free memory

    # =========================================================
    # WRITE FILES
    # =========================================================

    logger.info("Writting the cohort level analysis of outliers")
    print(updates)
    for file, metrics in updates.items():
        update_qc_file(file, metrics)

    logger.info("STREAMING cohort QC completed")

    return