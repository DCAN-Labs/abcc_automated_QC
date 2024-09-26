import nibabel as nib
import numpy as np
import pandas as pd
import os
import glob
import argparse
import re

def calculate_volumes(segmentation_img):
    """Calculate volumes of segmented regions from a NIfTI image."""
    data = segmentation_img.get_fdata()
    header = segmentation_img.header
    pixdim = header.get_zooms()
    # print(pixdim)

    # Extract voxel dimensions (in mm) from pixdim
    voxel_size_mm = pixdim[1:4]  
    voxel_size_mm3 = np.prod(voxel_size_mm)  # Voxel size in mm³

    
    unique_labels = np.unique(data)
    
    volumes = {}
    for label in unique_labels:
        if label == 0:
            continue
        voxel_count = np.sum(data == label)
        volume_ml = voxel_count * voxel_size_mm3
        volumes[label] = volume_ml
    
    return volumes

def read_label_mapping(mapping_file):
    """Read the label mapping from a text file and return a dictionary."""
    label_mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            # Skip lines that are comments or do not contain valid data
            if line.startswith('#') or not line.strip():
                continue
            
            # Use regular expressions to match lines with label and name
            match = re.match(r'(\d+)\s+(.+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', line)
            if match:
                label = int(match.group(1))
                name = match.group(2).strip()
                label_mapping[label] = name
    print("Label-mapping:", label_mapping)
    return label_mapping

def process_session(file_path):
    """Process a single NIfTI file and return a dictionary of volumes."""
    segmentation_img = nib.load(file_path)
    # print(segmentation_img)
    volumes = calculate_volumes(segmentation_img)
    return volumes

def main(data_dir, mapping_file, participants_file):
    """Main function to process NIfTI files and detect outliers."""

    label_mapping = read_label_mapping(mapping_file)
    file_paths = glob.glob(os.path.join(data_dir, '**', '*ACPC_dseg.nii.gz'), recursive=True)

    volume_data = {}

    # Process each file and store results
    for file_path in file_paths:
        basename = os.path.basename(file_path)
        print(f"Processing file: {basename}")
        subject = basename.split('_')[0] 
        session_name = basename.split('_')[1]
        subject_session = f"{subject}_{session_name}"
        
        volumes = process_session(file_path)
        
        for label, volume in volumes.items():
            if label not in volume_data:
                volume_data[label] = {}
            volume_data[label][subject_session] = volume

    # Convert the volume_data dictionary to a DataFrame
    df = pd.DataFrame.from_dict(volume_data, orient='index')

#    # Convert numeric labels to strings using the mapping file
#     df.index = df.index.map(lambda x: label_mapping.get(x, str(x)))  # Default to numeric if not in mapping

    # Create a set of labels present in the label_mapping
    valid_labels = set(label_mapping.keys())

    # Filter df to include only those rows where the index is in label_mapping
    df = df[df.index.isin(valid_labels)]
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # Convert numeric labels to strings using the mapping file
    df.index = df.index.map(lambda x: label_mapping.get(x, str(x)))  # Default to numeric if not in mapping

    print(type(df))

    # Calculate row-wise mean and standard deviation
    df['mean'] = df.mean(axis=1)
    df['std'] = df.std(axis=1)

    # Define outlier detection
    def is_outlier(value, mean, std, threshold=3):
        return value < (mean - threshold * std) or value > (mean + threshold * std)

    # Create outlier_df
    outlier_df = pd.DataFrame(index=df.index, columns=df.columns[:-2])  # Exclude 'mean' and 'std' columns

    for label in df.index:
        label_mean = df.loc[label, 'mean']
        label_std = df.loc[label, 'std']
        
        # Apply outlier detection for each value in the row (excluding 'mean' and 'std' columns)
        outlier_df.loc[label] = df.loc[label].iloc[:-2].apply(
            lambda x: "outlier" if pd.isna(x) or is_outlier(x, label_mean, label_std) else "no_outlier"
        )

    # Count outliers per subject
    subject_outliers = {}
    for subject_session in df.columns[:-2]:  # Exclude 'mean' and 'std' columns
        outlier_count = outlier_df[subject_session].value_counts().get('outlier', 0)
        if subject_session not in subject_outliers:
            subject_outliers[subject_session] = 0
        subject_outliers[subject_session] += outlier_count

    participants_df = pd.read_csv(participants_file, sep='\t')
    participants_df['combined_key'] = participants_df['participant_id'] + '_' + participants_df['session_id']

    
    participants_df['#subcortical_segmentation_vol_out'] = participants_df['combined_key'].map(subject_outliers)

    
    participants_df = participants_df.drop(columns=['combined_key'])

    participants_df.to_csv(participants_file, sep='\t', index=False)
    df.to_csv('/home/midb-ig/shared/projects/ABCD/Automated_QC/brain_coverage/subcortical_segmentation_vol_outlier_df_in_mm3_filtered.csv', sep=',', index=True)
    outlier_df.to_csv('/home/midb-ig/shared/projects/ABCD/Automated_QC/brain_coverage/subcortical_segmentation_vol_outlier_df.csv', sep=',', index=True)

    # Print the DataFrame and outlier_df
    print("Original DataFrame with Mean and Standard Deviation:")
    print(df)

    print("\nOutlier DataFrame:")
    print(outlier_df)

    print("\nUpdated participants.tsv with outlier counts:")
    print(participants_df)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process desg NIfTI files to calculate volumes and detect outliers.")
    parser.add_argument('data_dir', type=str, help="Directory containing NIfTI files ending with *ACPC_dseg.nii.gz")
    parser.add_argument('mapping_file', type=str, help="Text file containing label mappings")
    parser.add_argument('participants_file', type=str, help="Participants TSV file to update with outlier counts")
    
    args = parser.parse_args()

    print("Runing Subcortical segmention Vol outlier detection...........")
    main(args.data_dir, args.mapping_file, args.participants_file)
