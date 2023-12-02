base_dir="/mnt/data_lab513/tramy/4CAD/data/ADNI_nii/Mod"

for subject_dir in "$base_dir"/*; do
    if [ -d "$subject_dir" ]; then  # Check if it's a directory
        subject_id=$(basename "$subject_dir")

        # Find the .nii file and rename it
        for nii_file in "$subject_dir"/*.nii; do
            if [ -f "$nii_file" ]; then  # Check if .nii file exists
                mv "$nii_file" "${subject_dir}/${subject_id}.nii"
                echo "Renamed $nii_file to ${subject_dir}/${subject_id}.nii"
            fi
        done
    fi
done

