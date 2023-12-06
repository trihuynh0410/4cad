#!/bin/bash
classes=("CN" "MCI" "Mild")
exception="fsaverage" 

for class in "${classes[@]}"; do
    for dir in data/ADNI_out_mgz/${class}/*; do
        if [ -d "$dir" ]; then
            base_dir=$(basename "$dir")
            if [ "$base_dir" == "$exception" ]; then
                echo "Skipping $dir"
                continue
            fi
            newdir=$(echo "$dir" | awk -F'/' '{print $1"/"$2"/"$3"/"substr($4,6,10)}')
            mv "$dir" "$newdir"
        fi
    done
done


