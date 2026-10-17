from joblib import Parallel, delayed

from tqdm import tqdm

from .subject import process_subject

from .cohort import process_cohort


class Runner:

    def __init__(self, config, logger):

        self.config = config

        self.logger = logger

    def run(self):

        subjects = process_subject.discover_subjects(
            self.config.xcpd
        )

        self.logger.info(
            f"Found {len(subjects)} subjects"
        )

        Parallel(
            n_jobs=self.config.n_jobs
        )(
            delayed(process_subject)(
                subject,
                self.config
            )
            for subject in tqdm(subjects)
        )

        process_cohort(
            subjects,
            self.config
        )