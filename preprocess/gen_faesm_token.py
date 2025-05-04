import pickle
from pathlib import Path

from Bio import SeqIO
from faesm.esm import FAEsmForMaskedLM
from tqdm import tqdm

# --------------------------------------
# configure the paths and parameters
# --------------------------------------
device = 'cuda:0'

esm_arch = 'esm2_35m'

fasta_path = Path('./data/test.fasta')
save_pkl_path = Path(f'./data/test_{esm_arch}_inputs.pkl')

# --------------------------------------
# process the data
# --------------------------------------

records = list(SeqIO.parse(fasta_path, 'fasta'))
seq_data = [(record.id, str(record.seq)) for record in records]

match esm_arch:
    case 'esm2_35m':
        esm_model = FAEsmForMaskedLM.from_pretrained(f'facebook/esm2_t12_35M_UR50D',
                                                     use_fa=True)
    case _:
        raise NotImplementedError(f'Unknown model architecture: {esm_arch}')


def generate_tokens_in_batches(data):
    """
    Generate tokens for sequences in batches

    Args:
        data (list): List of (id, sequence) tuples

    Returns:
        Dictionary mapping sequence IDs to tokens
    """
    tokens_dict = {}

    # process sequences in batches
    for i in tqdm(range(0, len(data))):
        seq_id, seq = data[i]

        # this is compatible with ESM2 model
        inputs = esm_model.tokenizer(seq, return_tensors='pt')

        tokens_dict[seq_id] = inputs

    return tokens_dict


print(f'Using {esm_arch} model for token generation')

print(f'Generating tokens for {len(seq_data):,} sequences')
tokens_dict = generate_tokens_in_batches(seq_data)

with open(save_pkl_path, 'wb') as f:
    pickle.dump(tokens_dict, f)
