import pickle
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from transformers import EsmTokenizer

# --------------------------------------
# configure the paths and parameters
# --------------------------------------

device = 'cuda:0'

model_arch = 'saprot_35m'
model_path = '../ckpt/saprot_35m'

fasta_path = Path('../data/test.fasta')
metadata_path = Path('../data/test_metadata.csv')
fs_data_path = Path('../data/test_3di_tokens.txt')

save_pkl_path = Path(f'../data/test_{model_arch}_inputs.pkl')

# --------------------------------------
# process the data
# --------------------------------------

# 1. get the sequences
records = list(SeqIO.parse(fasta_path, 'fasta'))

seq_dict = {r.id: str(r.seq) for r in records}
seq_ids = set(r.id for r in records)

print(f'Number of input sequences: {len(seq_ids)}')

# 2. get the metadata
metadata_df = pd.read_csv(metadata_path)
print(metadata_df.head())

structure_id2seq_id = {s_id: uniprot_id for s_id, uniprot_id in
                           zip(metadata_df['structure_id'], metadata_df['seq_id'])}

# 3. get the 3di tokens
fs_data = {}

with open(fs_data_path, 'r') as f:
    for line in tqdm(f):
        items = line.strip().split('\t')

        struct_id = Path(items[0].split()[0]).stem
        seq_id = structure_id2seq_id.get(struct_id, None)
        # print(name)
        # input('...')

        if seq_id not in seq_ids:
            print(seq_id)
            continue

        seq = items[1]
        fs_seq = items[2]

        combined_seq = ''.join(a + b.lower() for a, b in zip(seq, fs_seq))

        fs_data[seq_id] = combined_seq

print(f'Number of sequences with tokens: {len(fs_data)}')

# generate the tokens
tokenizer = EsmTokenizer.from_pretrained(model_path)


def generate_tokens_in_batches(seq_dict, tokenizer):
    """
    Generate tokens for sequences in batches

    Args:
        data: List of (id, sequence) tuples
        batch_converter: ESM batch converter
        batch_size: Number of sequences to process at once

    Returns:
        Dictionary mapping sequence IDs to tokens
    """
    tokens_dict = {}

    # Process sequences in batches
    for seq_id, seq in tqdm(seq_dict.items(), total=len(seq_dict)):
        tokens = tokenizer(seq, return_tensors='pt')

        tokens_dict[seq_id] = tokens

    return tokens_dict


print('Generating tokens...')
tokens_dict = generate_tokens_in_batches(fs_data, tokenizer)

with open(save_pkl_path, 'wb') as f:
    pickle.dump(tokens_dict, f)

print(f'Saved tokens to {save_pkl_path}')
