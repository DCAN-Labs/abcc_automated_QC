#!/bin/bash -l
#SBATCH --job-name=copy_results_back_abcc_automated_qc
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH -t 76:00:00
#SBATCH -A midb-ig
#SBATCH -p msismall
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pandh015@umn.edu
#SBATCH -o /projects/standard/midb_abcd/shared/ABCC/code/abcc_automated_QC/abcc-qc_copy_results_back.out
#SBATCH -e /projects/standard/midb_abcd/shared/ABCC/code/abcc_automated_QC/abcc-qc_copy_results_back.err

# Load the environment
source /projects/standard/midb-ig/shared/projects/ABCD/Automated_QC/brain_coverage/autoQC/bin/activate

# Setting file paths
csv_file="/projects/standard/midb-ig/shared/projects/ABCD/fully_processed_abcd-hcp-pipeline_example/sub_ses_lists/prod_buc/fmriprep_03-30-26_final.csv"
final_bucket="s3://midb-abcd-main-pr"

# Create a directory for processed data
# output_base_dir="/projects/standard/midb-ig/shared/projects/ABCD/Automated_QC/brain_coverage/tmp_desg"
output_base_dir="/scratch.global/pandh015/abcc_autoqc"

# # Loop through the csv file to download dseg and dscalar files
# while IFS=',' read -r subject_id session_id _; do
#     if [[ "${subject_id}" == "subject" ]]; then
#         echo "skipping header"
#         continue
#     fi

#     if [ -d ${output_base_dir}/${subject_id}/${session_id} ]; then
#         echo "removing directory: ${output_base_dir}/${subject_id}/${session_id}"
#         rm -R ${output_base_dir}/${subject_id}/${session_id}
#     fi
#     mkdir -p ${output_base_dir}/${subject_id}/${session_id}/anat
#     mkdir -p ${output_base_dir}/${subject_id}/${session_id}/func

#     echo "Downloading ${subject_id} - ${session_id}"
#     # Sync the data from S3
#     ## For subcortical segmentation vol QC
#     s3cmd sync -F -v ${final_bucket}/derivatives/fmriprep_v25.1.4/sourcedata/freesurfer/${subject_id}_${session_id}.tar.gz ${output_base_dir}/${subject_id}/${session_id}/anat/${subject_id}_${session_id}.tar.gz -c /projects/standard/midb_abcd/shared/ABCC/code/s3cfgs/msi_loris_abcd_midb_s3.s3cfg

#     # ## For Cortical morphometry sulc depth QC 
#     s3cmd ls -c /projects/standard/midb_abcd/shared/ABCC/code/s3cfgs/msi_loris_abcd_midb_s3.s3cfg \
#     ${final_bucket}/derivatives/xcp_d_v26.1.1/${subject_id}/${session_id}/anat/ \
#     | grep '_desc-sulc_morph.tsv' \
#     | awk '{print $4}' \
#     | while read -r s3path; do
#         s3cmd get -F -v "$s3path" \
#           "${output_base_dir}/${subject_id}/${session_id}/anat/$(basename "$s3path")" \
#           -c /projects/standard/midb_abcd/shared/ABCC/code/s3cfgs/msi_loris_abcd_midb_s3.s3cfg
#       done

#     # ## For Brain Coverage 
#     s3cmd ls -c /projects/standard/midb_abcd/shared/ABCC/code/s3cfgs/msi_loris_abcd_midb_s3.s3cfg \
#     ${final_bucket}/derivatives/xcp_d_v26.1.1/${subject_id}/${session_id}/func/ \
#     | grep '_stat-coverage_bold.tsv' \
#     | awk '{print $4}' \
#     | while read -r s3path; do
#         s3cmd get -F -v "$s3path" \
#           "${output_base_dir}/${subject_id}/${session_id}/func/$(basename "$s3path")" \
#           -c /projects/standard/midb_abcd/shared/ABCC/code/s3cfgs/msi_loris_abcd_midb_s3.s3cfg
#       done

#     # For pconn QC
#     s3cmd ls -c /projects/standard/midb_abcd/shared/ABCC/code/s3cfgs/msi_loris_abcd_midb_s3.s3cfg \
#     ${final_bucket}/derivatives/xcp_d_v26.1.1/${subject_id}/${session_id}/func/ \
#     | grep 'volumes_boldmap.pconn.nii' \
#     | awk '{print $4}' \
#     | while read -r s3path; do
#         s3cmd get -F -v "$s3path" \
#           "${output_base_dir}/${subject_id}/${session_id}/func/$(basename "$s3path")" \
#           -c /projects/standard/midb_abcd/shared/ABCC/code/s3cfgs/msi_loris_abcd_midb_s3.s3cfg
#       done

#     s3cmd ls -c /projects/standard/midb_abcd/shared/ABCC/code/s3cfgs/msi_loris_abcd_midb_s3.s3cfg \
#     ${final_bucket}/derivatives/xcp_d_v26.1.1/${subject_id}/${session_id}/func/ \
#     | grep 'relmat.tsv' \
#     | awk '{print $4}' \
#     | while read -r s3path; do
#         s3cmd get -F -v "$s3path" \
#           "${output_base_dir}/${subject_id}/${session_id}/func/$(basename "$s3path")" \
#           -c /projects/standard/midb_abcd/shared/ABCC/code/s3cfgs/msi_loris_abcd_midb_s3.s3cfg
#       done

# done < "${csv_file}"


# Run subcortical segmentation vol
# python3 /projects/standard/midb_abcd/shared/ABCC/code/abcc_automated_QC/run_qc.py --config /projects/standard/midb_abcd/shared/ABCC/code/abcc_automated_QC/config/config.yaml --root ${output_base_dir} 

#copy the output to s3 bucket
# Loop through the csv file to download dseg and dscalar files
while IFS=',' read -r subject_id session_id _; do
    if [[ "${subject_id}" == "subject" ]]; then
        echo "skipping header"
        continue
    fi

    s3cmd sync -F -v ${output_base_dir}/${subject_id}/${session_id}/${subject_id}_${session_id}_desc-abcc-automated-qc.tsv ${final_bucket}/derivatives/abcc_qc/${subject_id}/${session_id}/${subject_id}_${session_id}_desc-abcc-automated-qc.tsv -c /projects/standard/midb_abcd/shared/ABCC/code/s3cfgs/msi_loris_abcd_midb_s3.s3cfg
    
done < "${csv_file}"