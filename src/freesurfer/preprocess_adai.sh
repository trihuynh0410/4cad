export FREESURFER_HOME=/mnt/data_lab513/tramy/freesurfer7/freesurfer
export FUNCTIONALS_DIR=$FREESURFER_HOME/sessions
source $FREESURFER_HOME/SetUpFreeSurfer.sh
classes=("CN" "MCI" "Mild" "Mod")

for class in "${classes[@]}"; do
    output_dir="ADAI_out/${class}"
    parent_folder="ADAI_data/${class}"
    export SUBJECTS_DIR="${output_dir}"
    find "${parent_folder}" -name "*.nii" -print0 | parallel -0 --jobs 16 'recon-all -i {} -s {.} -all'
done
