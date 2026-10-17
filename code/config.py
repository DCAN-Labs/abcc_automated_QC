import yaml
from pathlib import Path


class Config:

    def __init__(self, filename):

        with open(filename) as f:
            cfg = yaml.safe_load(f)

        self.output_filename = cfg["output_filename"]

        self.tasks = cfg["tasks"]

        self.atlases = cfg["atlases"]

        self.coverage_threshold = cfg["coverage_threshold"]

        self.coverage_pass = cfg["brain_coverage_pass"]

        self.outlier_sigma = cfg["outlier_sigma"]

        self.n_jobs = cfg["n_jobs"]