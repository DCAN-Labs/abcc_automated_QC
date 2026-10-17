from pathlib import Path

from code.structural.subcortical_v2 import extract_subcortical_metrics
from code.structural.sulcal import extract_sulcal_metrics
from code.functional.coverage import compute_brain_coverage
from code.functional.pconn import check_pconn_presence
from code.logger import get_logger
from code.utils.checkpoint import is_done, mark_done


# ---------------------------------------------------------------------
# Logger (shared across pipeline)
# ---------------------------------------------------------------------

logger = get_logger()


# ---------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------

def discover_subjects(root: Path):
    """
    Find all subjects in dataset.
    """
    return sorted(root.glob("sub-*"))


def discover_sessions(subject_dir: Path):
    """
    Find all sessions within a subject.
    """
    return sorted(subject_dir.glob("ses-*"))


# ---------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------

def process_subject(subject_dir: Path, config):
    """
    Process one subject across all sessions.

    Returns
    -------
    dict
        {
            "ses-01": {metrics},
            "ses-02": {metrics}
        }
    """
    subject_results = {}
    
    logger.info(f"Processing subject: {subject_dir.name}")


    sessions = discover_sessions(subject_dir)

    if not sessions:
        logger.warning(f"No sessions found: {subject_dir}")
        return subject_results

    # -----------------------------------------------------------------
    # Loop sessions
    # -----------------------------------------------------------------

    for session_dir in sessions:
        if is_done(subject_dir.parent, subject_dir.name, session_dir.name):

            logger.info(f"Skipping {subject_dir.name}; already done")
            return subject_results
    
        logger.info(f"Processing session: {session_dir.name}")

        metrics = {}

        anat_dir = session_dir / "anat"
        func_dir = session_dir / "func"

        # =============================================================
        # STRUCTURAL QC
        # =============================================================

        # -----------------------------
        # Subcortical volumes
        # -----------------------------
        try:

            freesurfer_tar_path= anat_dir / f"{subject_dir.name}_{session_dir.name}.tar.gz"
            subcortical = extract_subcortical_metrics(freesurfer_tar_path)

            for roi, volume in subcortical.items():

                metrics[f"subcortical_{roi}_volume"] = volume

        except Exception as e:

            logger.exception(
                f"Subcortical QC failed: {subject_dir.name}/{session_dir.name}"
            )

        # -----------------------------
        # Sulcal morphometry
        # -----------------------------
        try:

            sulcal = extract_sulcal_metrics(anat_dir)

            for region, value in sulcal.items():

                metrics[f"sulc_{region}"] = value

        except Exception as e:

            logger.exception(
                f"Sulcal QC failed: {subject_dir.name}/{session_dir.name}"
            )

        # =============================================================
        # FUNCTIONAL QC
        # =============================================================
        for task in config.tasks:
            logger.info(f"task: {task}")
            for atlas in config.atlases:
                logger.info(f"atlas: {atlas}")
                try:

                    # -----------------------------
                    # Brain coverage
                    # -----------------------------
                    coverage_metrics = compute_brain_coverage(
                        func_dir,
                        task=task,
                        atlas=atlas,
                        threshold=config.coverage_threshold
                    )
                    metrics.update(coverage_metrics)

                    # -----------------------------
                    # pconn presence
                    # -----------------------------
                    pconn_metrics = check_pconn_presence(
                        func_dir,
                        task=task,
                        atlas=atlas
                    )

                    metrics.update(pconn_metrics)

                except Exception as e:

                    logger.exception(
                        f"Functional QC failed: {subject_dir.name}/{session_dir.name}"
                    )

        # =============================================================
        # STORE SESSION RESULT
        # =============================================================

        subject_results[session_dir.name] = metrics

        mark_done(subject_dir.parent, subject_dir.name, session_dir.name)
    
    return subject_results