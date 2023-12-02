for i in /mnt/data_lab513/tramy/4CAD/data/ADNI_nii/Mod_dcm/*; do
    subject_id=$(basename $i)
    mkdir -p /mnt/data_lab513/tramy/4CAD/data/ADNI_nii/Mod/$subject_id
    dcm_dir=$(find $i -type d | awk -F'/' '{print NF-1, $0}' | sort -nr | head -1 | cut -d" " -f2-)
    #echo $dcm_dir
    dcm2niix -z n -f %p_%s -o /mnt/data_lab513/tramy/4CAD/data/ADNI_nii/Mod/$subject_id $dcm_dir
done
