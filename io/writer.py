import pandas as pd


def write_subject_qc(outfile, metrics):

    df = pd.DataFrame(
        {
            "Metric": metrics.keys(),
            "Value": metrics.values(),
        }
    )

    df.to_csv(
        outfile,
        sep="\t",
        index=False,
    )