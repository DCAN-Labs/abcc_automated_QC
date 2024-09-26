import nibabel as nib
import numpy as np
import pandas as pd
import os
import glob
import argparse

def extract_labels_from_dlabel(pconn_file):
    """Extract labels from a .dlabel.nii file."""
    label_img = nib.load(pconn_file)
    header = label_img.header

    labels = []
    index = 0
    for index_map in header.get_index_map(1):
        if isinstance(index_map, nib.cifti2.cifti2.Cifti2Parcel):
            label_name = index_map.name
            if isinstance(label_name, str):
                labels.append((index, label_name))
                index += 1
            else:
                print(f"Unexpected label type: {type(label_name)}")
    return [label[1] for label in labels if label[1] != '???']

def map_conn_mat_to_labels(scalar_file, labels):
    """Map connectivity matrix values to labels from the .pconn.nii file."""
    scalar_img = nib.load(scalar_file)
    conn_matrix = scalar_img.get_fdata()

    num_parcels = len(labels)

    if conn_matrix.shape[0] != num_parcels:
        raise ValueError("Number of parcels in scalar data does not match number of labels")
    
    lower_triangle_indices = np.tril_indices_from(conn_matrix, k=-1)
    lower_triangle_values = conn_matrix[lower_triangle_indices]

    index_label_pairs = [
        (f"{labels[i]}_{labels[j]}", value)
        for i, j, value in zip(lower_triangle_indices[0], lower_triangle_indices[1], lower_triangle_values)
    ]

    return index_label_pairs

def count_outliers(outlier_df):
    """Count the number of outliers per subject-session."""
    # Initialize a dictionary to store outlier counts
    outlier_counts = {}
    
    # Iterate over each column (subject-session) to count outliers
    for column in outlier_df.columns:
        outlier_count = outlier_df[column].value_counts().get('outlier', 0)
        outlier_counts[column] = outlier_count
    
    return outlier_counts

def process_files(file_paths, labels):
    """Process .pconn.nii files and return a dictionary of results."""
    pconn_data = {}
    for file_path in file_paths:
        basename = os.path.basename(file_path)
        print(f"Processing file: {basename}")
        subject = basename.split('_')[0]
        session_name = basename.split('_')[1]
        subject_session = f"{subject}_{session_name}"
        
        pconn = map_conn_mat_to_labels(file_path, labels)
        for combined_label, value in pconn:
            if combined_label not in pconn_data:
                pconn_data[combined_label] = {}
            pconn_data[combined_label][subject_session] = value
    
    return pconn_data

def main(data_dir, participants_file):
    """Main function to process NIfTI files and detect outliers."""
    
    # Load participants DataFrame
    participants_df = pd.read_csv(participants_file, sep='\t')
    participants_df['combined_key'] = participants_df['participant_id'] + '_' + participants_df['session_id']

    # Check for the existence of 5-minute and 10-minute files
    file_paths_5min = glob.glob(os.path.join(data_dir, '**', '*_censor-5min_conndata-network_connectivity.pconn.nii'), recursive=True)
    file_paths_10min = glob.glob(os.path.join(data_dir, '**', '*_censor-10min_conndata-network_connectivity.pconn.nii'), recursive=True)

    # Extract subject-session from file paths
    session_keys_5min = set(f"{os.path.basename(fp).split('_')[0]}_{os.path.basename(fp).split('_')[1]}" for fp in file_paths_5min)
    session_keys_10min = set(f"{os.path.basename(fp).split('_')[0]}_{os.path.basename(fp).split('_')[1]}" for fp in file_paths_10min)

    # Initialize columns if not present
    if '5min_pconn' not in participants_df.columns:
        participants_df['5min_pconn'] = 'fail'
    if '10min_pconn' not in participants_df.columns:
        participants_df['10min_pconn'] = 'fail'
    
    # Update columns based on file availability
    participants_df['5min_pconn'] = participants_df['combined_key'].map(lambda k: 'Pass' if k in session_keys_5min else 'Fail')
    participants_df['10min_pconn'] = participants_df['combined_key'].map(lambda k: 'Pass' if k in session_keys_10min else 'Fail')
    
    # Process files and calculate outliers if files are present
    subject_outliers_5min = {}
    subject_outliers_10min = {}

    def is_outlier(value, mean, std, threshold=3):
        return value < (mean - threshold * std) or value > (mean + threshold * std)
    
    if session_keys_5min:
        pconn_data_5min = process_files(file_paths_5min, extract_labels_from_dlabel(file_paths_5min[0]))
        df_5min = pd.DataFrame.from_dict(pconn_data_5min, orient='index')
        df_5min['mean'] = df_5min.mean(axis=1)
        df_5min['std'] = df_5min.std(axis=1)
        
        outlier_df_5min = pd.DataFrame(index=df_5min.index, columns=df_5min.columns[:-2])
        for label in df_5min.index:
            label_mean = df_5min.loc[label, 'mean']
            label_std = df_5min.loc[label, 'std']
            outlier_df_5min.loc[label] = df_5min.loc[label].iloc[:-2].apply(
                lambda x: "outlier" if pd.isna(x) or is_outlier(x, label_mean, label_std) else "no_outlier"
            )
        
        subject_outliers_5min = count_outliers(outlier_df_5min)
    
    if session_keys_10min:
        pconn_data_10min = process_files(file_paths_10min, extract_labels_from_dlabel(file_paths_10min[0]))
        df_10min = pd.DataFrame.from_dict(pconn_data_10min, orient='index')
        df_10min['mean'] = df_10min.mean(axis=1)
        df_10min['std'] = df_10min.std(axis=1)

        outlier_df_10min = pd.DataFrame(index=df_10min.index, columns=df_10min.columns[:-2])
        for label in df_10min.index:
            label_mean = df_10min.loc[label, 'mean']
            label_std = df_10min.loc[label, 'std']
            outlier_df_10min.loc[label] = df_10min.loc[label].iloc[:-2].apply(
                lambda x: "outlier" if pd.isna(x) or is_outlier(x, label_mean, label_std) else "no_outlier"
            )
        
        subject_outliers_10min = count_outliers(outlier_df_10min)
    
    # Update outlier count columns
    participants_df['#pconn_out_5min'] = participants_df['combined_key'].map(subject_outliers_5min)
    participants_df['#pconn_out_10min'] = participants_df['combined_key'].map(subject_outliers_10min)

    # Drop the combined_key column
    participants_df = participants_df.drop(columns=['combined_key'])

    # Save the updated DataFrame
    participants_df.to_csv(participants_file, sep='\t', index=False)
    if session_keys_5min:
        df_5min.to_csv('/home/midb-ig/shared/projects/ABCD/Automated_QC/brain_coverage/pconn_outlier_5min_df.csv', sep=',', index=True)
    if session_keys_10min:
        df_10min.to_csv('/home/midb-ig/shared/projects/ABCD/Automated_QC/brain_coverage/pconn_outlier_10min_df.csv', sep=',', index=True)

    print("Updated participants.tsv with pass/fail and outlier counts:")
    print(participants_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process desg NIfTI files to calculate volumes and detect outliers.")
    parser.add_argument('data_dir', type=str, help="Directory containing NIfTI files ending with *pconn.nii")
    parser.add_argument('participants_file', type=str, help="Participants TSV file to update with outlier counts")
    
    args = parser.parse_args()
    
    print("Running pconn outlier detection...")
    main(args.data_dir, args.participants_file)
