export FREESURFER_HOME=/mnt/data_lab513/tramy/freesurfer7/freesurfer
export FUNCTIONALS_DIR=$FREESURFER_HOME/sessions
source $FREESURFER_HOME/SetUpFreeSurfer.sh
classes=("Mod")

for class in "${classes[@]}"; do
    output_dir="/mnt/data_lab513/tramy/4CAD/data/ADNI_out_mgz/${class}"
    parent_folder="/mnt/data_lab513/tramy/4CAD/data/ADNI_nii/${class}"
    export SUBJECTS_DIR="${output_dir}"
    find "${parent_folder}" -name "*.nii" -print0 | parallel -0 --jobs 16 'recon-all -i {} -s {.} -all'
done
