import nibabel as nib
import numpy as np
import pandas as pd
import os
import glob
import argparse

def extract_labels_from_dlabel(dlabel_file):
    """Extract labels from a .dlabel.nii file."""
    label_img = nib.load(dlabel_file)
    header = label_img.header

    labels = []
    for index_map in header.get_index_map(0):
        if hasattr(index_map, 'label_table'):
            label_table = index_map.label_table
            
            # Extract labels and names
            for index, label in label_table._labels.items():
                label_name = label.label
                
                # Check type of label object
                if isinstance(label_name, str):
                    labels.append((index, label_name))
                else:
                    print(f"Unexpected label type: {type(label)}")
    # Return only the label names in order
    return [label[1] for label in labels if label[1] != '???']
    #return [label for label in labels if label[1] != '???']

def map_scalar_to_labels(scalar_file, labels):
    """Map scalar values to labels from the .pscalar.nii file."""
    scalar_img = nib.load(scalar_file)
    scalar_data = scalar_img.get_fdata()

    # Flatten the scalar data
    scalar_data_flat = scalar_data.flatten()
    num_parcels = len(labels)

    if scalar_data_flat.shape[0] != num_parcels:
        raise ValueError("Number of parcels in scalar data does not match number of labels")


   # Create a dictionary to map labels to scalar values
    label_value_dict = {labels[i]: scalar_data_flat[i] for i in range(num_parcels)}

    return label_value_dict

def main(data_dir, label_file, participants_file):
    """Main function to process NIfTI files and detect outliers."""

    # Extract labels from the dlabel file
    labels = extract_labels_from_dlabel(label_file)

    # # Print labels to verify
    # print("Labels extracted:")
    # for label in labels:
    #     print(label)
    file_paths = glob.glob(os.path.join(data_dir, '**', '*_space-fsLR32k_sulc.pscalar.nii'), recursive=True)

    pscalar_data = {}

    for file_path in file_paths:
        basename = os.path.basename(file_path)
        print(f"Processing file: {basename}")
        subject = basename.split('_')[0] 
        session_name = basename.split('_')[1]
        subject_session = f"{subject}_{session_name}"
        
        pscalar = map_scalar_to_labels(file_path, labels)
        
        for label, pscalar in pscalar.items():
            if label not in pscalar_data:
                pscalar_data[label] = {}
            pscalar_data[label][subject_session] = pscalar

    # Convert the pscalar_data dictionary to a DataFrame
    df = pd.DataFrame.from_dict(pscalar_data, orient='index')

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
        # subject = subject_session.split('_')[0]
        if subject_session not in subject_outliers:
            subject_outliers[subject_session] = 0
        subject_outliers[subject_session] += outlier_count   

    participants_df = pd.read_csv(participants_file, sep='\t')
    participants_df['combined_key'] = participants_df['participant_id'] + '_' + participants_df['session_id']

    participants_df['#cortical_morphometry_sulc_out'] = participants_df['combined_key'].map(subject_outliers)
    participants_df = participants_df.drop(columns=['combined_key'])

    participants_df.to_csv(participants_file, sep='\t', index=False)
    df.to_csv('/home/midb-ig/shared/projects/ABCD/Automated_QC/brain_coverage/cortical_morphometry_sulc_outlier_df_with_mean_SD.csv', sep=',', index=True)
    outlier_df.to_csv('/home/midb-ig/shared/projects/ABCD/Automated_QC/brain_coverage/cortical_morphometry_sulc_outlier_df.csv', sep=',', index=True)


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
    parser.add_argument('data_dir', type=str, help="Directory containing NIfTI files ending with *pscalar.nii")
    parser.add_argument('mapping_file', type=str, help="dlabel file containing label mappings")
    parser.add_argument('participants_file', type=str, help="Participants TSV file to update with outlier counts")
    
    args = parser.parse_args()
    
    print("Runing Cortical Morphometry surface outlier detection...........")
    main(args.data_dir, args.mapping_file, args.participants_file)



