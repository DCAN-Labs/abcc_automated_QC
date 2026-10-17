from pathlib import Path
import pandas as pd


def update_qc_file(file: Path, new_metrics: dict):
    """
    Append or update QC metrics in an existing TSV.

    Args:
        file (Path): Path to QC TSV file.
        new_metrics (dict): Nested dict mapping Metric name -> {column_name: value}.
            Example:
                {
                    "bc_task-rest_run-1": {
                        "QC_pass_fail": "PASS",
                        "bc_perc_vox": 92.5,
                    },
                    "bc_task-rest_run-2": {
                        "QC_pass_fail": "FAIL",
                        "bc_perc_vox": 61.2,
                    },
                }
            Column names (the inner dict keys) must be one of:
                - "QC_pass_fail"
                - "bc_perc_vox"
                - "outlier_count"
                - "total_measurements(N)"

    Behavior:
        - Column names are inferred from the keys of each metric's inner dict
          (no need to pass a separate column_name).
        - If file exists:
            - Update only the specified columns for existing Metric rows.
            - Preserve all other columns and values.
            - Add new Metric rows if they do not already exist.
            - Add new columns if they do not already exist (existing rows
              backfilled with 888.0).
        - If file does not exist:
            - Create a new file with a Metric column plus one column per
              distinct QC field found across all metrics.
    """

    valid_columns = {
        "QC_pass_fail",
        "bc_perc_vox",
        "outlier_count",
        "total_measurements(N)",
    }

    # Figure out every column referenced across all metrics up front,
    # and validate them.
    all_columns = set()
    for metric, cols in new_metrics.items():
        for column_name in cols:
            if column_name not in valid_columns:
                raise ValueError(
                    f"Invalid column_name '{column_name}' for metric '{metric}'. "
                    f"Must be one of: {sorted(valid_columns)}"
                )
            all_columns.add(column_name)

    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # If file exists → load existing QC file
    # ---------------------------------------------------------
    if file.exists():
        df = pd.read_csv(file, sep="\t")

        # Make sure every referenced column exists.
        # Existing rows get 888.0 until explicitly updated.
        for column_name in all_columns:
            if column_name not in df.columns:
                df[column_name] = 888.0

        # Update existing metrics or add new ones
        for metric, cols in new_metrics.items():

            mask = df["Filename"] == metric

            if mask.any():
                # Only update the requested columns for this metric's row.
                for column_name, value in cols.items():
                    df.loc[mask, column_name] = value

            else:
                # Add a new row while preserving existing columns.
                new_row = {col: 888.0 for col in df.columns}
                new_row["Filename"] = metric
                for column_name, value in cols.items():
                    new_row[column_name] = value

                df = pd.concat(
                    [df, pd.DataFrame([new_row])],
                    ignore_index=True
                )

    # ---------------------------------------------------------
    # If file does not exist → create new
    # ---------------------------------------------------------
    else:
        rows = []
        for metric, cols in new_metrics.items():
            row = {col: 888.0 for col in all_columns}
            row["Filename"] = metric
            row.update(cols)
            rows.append(row)

        df = pd.DataFrame(rows, columns=["Filename", *sorted(all_columns)])

    # ---------------------------------------------------------
    # Write back
    # ---------------------------------------------------------
    tmp_file = file.with_suffix(".tmp.tsv")
    df.to_csv(tmp_file, sep="\t", index=False)
    tmp_file.replace(file)