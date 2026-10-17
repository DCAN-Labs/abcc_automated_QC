from setuptools import setup, find_packages

setup(
    name="abcc_qc",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "nibabel",
        "pyyaml",
        "joblib",
        "tqdm",
    ],
)