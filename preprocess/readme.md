## Data preprocessing

The raw input contains three parts:
1. A `fasta` file containing all the sequences for both the human template and the bacterial database.
2. A directory of `pdb` files for each sequence. 
3. A `csv` file containing the metadata for the input sequences and structures. The two most important columns are `seq_id` and `structure_id`, which maps the names for the sequences in the `fasta` file and the structures in the `pdb` file. For the `EC` column, if no information is availble, a placeholder EC (like `1.1.1.1`) can be used for all entries.

The raw sequences and structures should be preprocessed before inputting to the DEER model for generating the embeddings. To do that, follow the below steps.

**Step 1. Generate 3Di tokens**

Modify the paths in the below file and run it.
```bash
chmod +x gen_saprot_token.sh
./gen_saprot_token.sh
```

**Step 2. Generate SaProt embeddings**

Open `gen_saprot_token.py` and modify the following variables.
- `fasta_path`: the path to the `fasta` file with all the sequences.
- `metadata_path`: the path to the `csv` file with the metadata.
- `fs_data_path`: the path of the `txt` file outputted from Step 1.
- `save_pkl_path`: the path of the `pkl` file for output in this step.

```bash
python gen_saprot_token.py
```

**Step 3. Generate ESM embeddings**

Open `gen_faesm_token.py` and modify the following variables.
- `fasta_path`: the path to the `fasta` file with all the sequences.
- `save_pkl_path`: the path of the `pkl` file for output in this step.

```bash
python gen_faesm_token.py
```

Next, users may run `predict.py` in the main directory to generate the embeddings for the template and database entries.
