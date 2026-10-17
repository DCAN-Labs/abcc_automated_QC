from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Checkpoint file
# ---------------------------------------------------------------------

CKPT_FILE = "abcc_qc_checkpoint.json"


# ---------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------

def load_checkpoint(root: Path):

    path = Path(root) / CKPT_FILE

    if not path.exists():

        return {
            "processed_subjects": []
        }

    with open(path, "r") as f:

        return json.load(f)


# ---------------------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------------------

def save_checkpoint(root: Path, data: dict):

    path = Path(root) / CKPT_FILE

    with open(path, "w") as f:

        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------
# Mark subject complete
# ---------------------------------------------------------------------

def mark_done(root: Path, subject: str, session: str):

    ckpt = load_checkpoint(root)
    key=subject+'_'+session
    if key not in ckpt["processed_subjects"]:

        ckpt["processed_subjects"].append(key)

    save_checkpoint(root, ckpt)


# ---------------------------------------------------------------------
# Check if done
# ---------------------------------------------------------------------

def is_done(root: Path, subject: str, session: str):

    ckpt = load_checkpoint(root)
    key = subject+'_'+session
    return key in ckpt.get("processed_subjects", [])