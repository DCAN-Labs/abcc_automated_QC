# abcc_automated_QC
This repo has the code for automated_QC (structural QC as well as functional QC) performed on abcc data processed using the [abcd-hcp-pipeline](https://github.com/DCAN-Labs/abcd-hcp-pipeline/) and file-mapped using [DCAN Lab's file-mapper](https://github.com/DCAN-Labs/file-mapper/)<br>
# Structural QC
The following are automated metrics for evaluating processed structural data quality:<br>

## Population Metrics from outlier detection:<br>
  * Subcortical segmentation volumes size – outliers based on 3 SD from the mean<br>
  <b>Usage:</b><br>
  `python3 ./subcortical_vol_outlier_detection.py ${output_base_dir} ${dseg_label_file} ${tsv_file}` <br> <br>
  <b>Returns:</b><br>
  An updated tsv file with an additional column `#subcortical_segmentation_vol_out`, showing the number of outliers for each subject and session
 
  
  * Cortical morphometry, split by region of interest – outliers based on 3 SD from them mean<br>
    <b>Usage:</b><br>
      `python3 ./Cortical_morphometry_surface_outlier_detection.py ${output_base_dir} ${pscalar_label_file} ${tsv_file}`<br><br>
    <b>Returns:</b><br>
       An updated tsv file with an additional column `#cortical_morphometry_sulc_out`, showing the number of outliers for each subject and session<br><br>
 <b> Parameters:</b><br>
  * output_base_dir: directory where you have all *_space-fsLR32k_sulc.pscalar.nii, *_space-ACPC_dseg.nii.gz files stored<br>
  * dseg_label_file: ./dseg_label.txt <br>
  * pscalat_label_file: Gordon.32k_fs_LR.dlabel.nii<br>
  * tsv_file: participants.tsv<br>

# Functional QC 
### Pconn File detection using pconn_outlier_detection.py<br>
  * Column for passing 5 minutes of data:(based on whether the pconns get produced)<br> 
  * Column for passing 10 minutes of data:(based on whether the pconns get produced)<br> 10min_pconn
  
### Population metrics from outlier detection<br>
  * Connectivity matrix for 5 min and 10 min pconns – outliers based on 3 SD from the mean
  Usage:<br>
 `python3 ./pconn_outlier_detection.py ${output_base_dir} ${tsv_file}`<br><br>
 <b>Returns:</b><br>
 An updated TSV file with the following additional columns:
 
 * `5min_pconn`: Pass/Fail based on file presence.
 * `10min_pconn`: Pass/Fail based on file presence.
 * `#pconn_out_5min`: Number of outliers for each subject and session for the 5-minute connectivity matrix.
 * `#pconn_out_10min`: Number of outliers for each subject and session for the 10-minute connectivity matrix. <br><br>
<b> Parameters:</b><br>
* output_base_dir: directory where you have all *_pconn.nii.gz files stored <br>
* tsv_file: participants.tsv<br>


*Note* - Any column that is nan/empty signifies that particular file did not exist.
