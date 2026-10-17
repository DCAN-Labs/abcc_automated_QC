#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd

from code.config import Config
from code.logger import get_logger

from code.subject import discover_subjects, process_subject
from code.cohort import run_cohort_qc
from code.functional.connectivity_v2 import run_connectivity_qc
from code.utils.update_qc_file import update_qc_file


logger = get_logger()

def write_subject_qc(outfile, metrics):

    df = pd.DataFrame(
        {
            "Filename": metrics.keys(),
            "Value": metrics.values(),
        }
    )

    df.to_csv(
        outfile,
        sep="\t",
        index=False,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="ABCC Automated QC Pipeline"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to config.yaml"
    )

    parser.add_argument(
        "--mode",
        choices=["subject", "cohort", "connectivity", "full"],
        default="full",
        help="Pipeline stage"
    )

    parser.add_argument(
        "--root",
        required=True,
        help="Dataset root (derivatives/xcp_d...)"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    config = Config(args.config)

    root = Path(args.root)

    logger.info(f"Mode: {args.mode}")
    logger.info(f"Root: {root}")

    # =============================================================
    # 1. SUBJECT LEVEL QC
    # =============================================================

    # if args.mode in ["subject", "full"]:

    #     subjects = discover_subjects(root)

    #     logger.info(f"Found {len(subjects)} subjects")

    #     for s in subjects:

    #         subject_results=process_subject(s, config)
    #         # print(f"Subject {s.name} results: {subject_results}")
    #         if subject_results:
    #             for ses, metrics in subject_results.items():
    #                 # print(f"Subject {s.name} Session {ses}: \n")
    #                 # print(f"metrics: {metrics}")
    #                 output_filename = config.output_filename + f"_{s.name}_{ses}.tsv"
    #                 output_file = root / "tmp" / output_filename
    #                 output_file.parent.mkdir(parents=True, exist_ok=True)
    #                 logger.info(f"Writing results to {output_file}")
    #                 write_subject_qc(output_file, metrics)
    #         else: 
    #             logger.warning(f"No results for subject {s.name}")
    #             exit(1)
    
    if args.mode in ["subject", "full"]:

        subjects = discover_subjects(root)
        logger.info(f"Found {len(subjects)} subjects")

        for s in subjects:

            subject_results = process_subject(s, config)

            if subject_results:

                for ses, metrics in subject_results.items():

                    # Separate metrics by type
                    prefix_metrics = {}  # For sulc_ and subcortical_
                    qc_metrics = {}      # For everything else

                    for k, v in metrics.items():
                        if "sulc_" in k or "subcortical_" in k:
                            prefix, subkey = k.split("_", 1)
                            if prefix not in prefix_metrics:
                                prefix_metrics[prefix] = {}
                            prefix_metrics[prefix][subkey] = v
                        else:
                            qc_metrics[k] = v

                    # Write prefix metrics (sulc_ and subcortical_) to TSV files
                    for prefix, metric_block in prefix_metrics.items():

                        output_filename = (
                            f"{s.name}_{ses}_{config.output_filename}_{prefix}.tsv"
                        )

                        output_file = root / "tmp" / output_filename
                        output_file.parent.mkdir(parents=True, exist_ok=True)

                        logger.info(f"Writing {prefix} metrics to {output_file}")

                        write_subject_qc(output_file, metric_block)

                    # Write other metrics to QC file
                    if qc_metrics:
                        qc_filename = f"{s.name}_{ses}_{config.output_filename}.tsv"
                        qc_file = root / s.name / ses / qc_filename
                        logger.info(f"Updating QC file : {qc_file}")
                        # logger.info(f"Updating QC file with metrics: {list(qc_metrics.keys())}")
                        update_qc_file(qc_file, qc_metrics)

            else:
                logger.warning(f"No results for subject {s.name}")

    # =============================================================
    # 2. COHORT QC
    # =============================================================

    if args.mode in ["cohort", "full"]:

        run_cohort_qc(root, config)

    # =============================================================
    # 3. CONNECTIVITY QC
    # =============================================================

    if args.mode in ["connectivity", "full"]:

        pconn_qc=run_connectivity_qc(root, config)
        logger.info('pconn_qc:', pconn_qc)

    # =============================================================
    # 3. Write the results to a tsv file
    # =============================================================




    logger.info("QC pipeline completed successfully")


if __name__ == "__main__":

    main()