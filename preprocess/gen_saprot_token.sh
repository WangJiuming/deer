#!/bin/bash

pdb_dir="../data/pdb"
save_path="../data/test_3di_tokens.txt"

foldseek structureto3didescriptor --chain-name-mode 1 \
                                  "${pdb_dir}" \
                                  "${save_path}"

rm "${save_path}.dbtype"

echo "3Di tokens saved to ${save_path}"


