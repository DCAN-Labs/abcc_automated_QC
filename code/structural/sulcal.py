

"""
Sulcal Morphometry QC

Reads XCP-D:
*_desc-sulc_morph.tsv

Outputs raw regional sulcal depth values
for cohort-level z-scoring later.
"""

from pathlib import Path
import pandas as pd
from code.logger import get_logger
logger = get_logger()


def _find_sulc_file(func_dir: Path):
    """
    Locate sulcal morphometry file.
    """
    files = list(func_dir.glob("*desc-sulc_morph*.tsv"))

    if not files:
        raise FileNotFoundError(f"No sulcal file in {func_dir}")

    return files[0]


def extract_sulcal_metrics(func_dir: Path):
    """
    Returns:
        dict: {region: sulcal_depth}
    """

    func_dir = Path(func_dir)

    file = _find_sulc_file(func_dir)

    logger.info(f"[Sulcal] Loading {file}")

    df = pd.read_csv(file, sep="\t")

    if df.empty:
        raise ValueError(f"Sulcal file is empty: {file}")

    if len(df) != 1:
        raise ValueError(
            f"Expected one row in {file}, found {len(df)}"
        )

    return {
        f"{col}": float(df.at[0, col])
        for col in df.columns
    }


