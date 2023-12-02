
subjects_dir="data/ADNI_out_mgz_copy/CN/" 

for subject in ${subjects_dir}/*/; do
    echo "Converting files for subject: ${subject}"
    for mgz_file in ${subject}/mri/*.mgz; do
        nii_file="${mgz_file%.mgz}.nii" # Replace .mgz with .nii in the filename
        mri_convert "$mgz_file" "$nii_file"
    done
done
