#!/bin/bash

root_dir="/workspace/jmwang/protein/gmps/typephage"
output_dir="/workspace/jmwang/protein/deer/result/gmps/typephage"

# iterate over all the directories in the root dir
for dir in "${root_dir}"/*; do
    # check if it is a directory
    if [ -d "${dir}" ]; then
        # get the name of the directory
        dir_name=$(basename "${dir}")
        echo "processing ${dir_name}..."

        fasta_dir="${dir}/protein/"

        # check if there is only one fasta file in the directory
        # fasta_paths=("${fasta_dir}"*.FA)
        
        # select the only fasta file
        # fasta_path="${fasta_paths[0]}"
        # echo "fasta_path: ${fasta_path}"

        csv_path="${dir}/${dir_name}.csv"

        # overwrite the test_fasta_path, test_esm_input_pkl_dir, test_saprot_input_pkl_dir, test_meta_path, and save_result_dir from the config file

        python test.py \
            --test_fasta_dir "${fasta_dir}" \
            --test_esm_input_pkl_dir "${output_dir}/${dir_name}/" \
            --test_saprot_input_pkl_dir "${output_dir}/${dir_name}/" \
            --test_meta_path "${csv_path}" \
            --save_result_dir "${output_dir}/${dir_name}/"

            

        # break
        
        
    fi
done
