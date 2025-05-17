import argparse
import pickle
import subprocess
from pathlib import Path

import lightning as pl
import numpy as np
import pandas as pd
import torch
from dataset import GMPSDataModule
from faesm.esm import FAEsmForMaskedLM
from model import GMPSModel
from tqdm import tqdm
from transformers import EsmTokenizer


def generate_3di_tokens(template_pdb_dir, tmp_dir, verbose=True):
    """Generate 3di tokens with foldseek and parse them."""
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    cmd = f'foldseek structureto3didescriptor --chain-name-mode 1 {template_pdb_dir} {tmp_dir}/3di_tokens.txt'

    if verbose:
        print('[INFO] Generating 3di tokens...')
        print(f'[INFO] Running command: {cmd}')

    subprocess.run(cmd, shell=True, check=True)

    fs_3di_dict = {}
    seq_dict = {}

    if verbose:
        print('[INFO] Generating SaProt tokens...')
        print(f'[INFO] Parsing 3di tokens from {tmp_dir}/3di_tokens.txt')

    with open(f'{tmp_dir}/3di_tokens.txt', 'r') as f:
        for line in tqdm(f):
            items = line.strip().split('\t')

            template_id = Path(items[0].split()[0]).stem

            fs_3di_seq = items[1]
            fs_seq = items[2]

            combined_seq = ''.join(a + b.lower() for a, b in zip(fs_3di_seq, fs_seq))

            fs_3di_dict[template_id] = combined_seq
            seq_dict[template_id] = fs_3di_seq

    if verbose:
        print(f'[INFO] Number of tokenized sequences: {len(fs_3di_dict)}')

    return fs_3di_dict, seq_dict


def generate_saprot_tokens(fs_3di_dict, saprot_model_dir, tmp_dir, verbose=True):
    """Generate SaProt tokens for sequences."""
    saprot_tokenizer = EsmTokenizer.from_pretrained(saprot_model_dir)

    saprot_token_dict = {}

    if verbose:
        print('[INFO] Generating SaProt tokens for sequences...')

    for seq_id, fs_3di_seq in tqdm(fs_3di_dict.items(), total=len(fs_3di_dict)):
        try:
            tokens = saprot_tokenizer(fs_3di_seq.replace('X', '#'), return_tensors='pt')
        except ValueError as e:
            raise ValueError(f'Error processing sequence {seq_id}: {e}')

        saprot_token_dict[seq_id] = tokens

    saprot_pkl_path = Path(tmp_dir) / 'saprot_tokens.pkl'
    with open(saprot_pkl_path, 'wb') as f:
        pickle.dump(saprot_token_dict, f)

    if verbose:
        print(f'[INFO] Generated SaProt tokens for {len(saprot_token_dict)} sequences')
        print(f'[INFO] Saved SaProt tokens to {saprot_pkl_path}')

    return saprot_token_dict, saprot_pkl_path


def setup_device(use_fa, verbose=True):
    """Setup device (CPU/GPU) for model execution."""
    if not torch.cuda.is_available():
        use_fa = False
        device = 'cpu'
        device_list = None
        if verbose:
            print('[INFO] CUDA is not available, disabling Flash Attention by force and using CPU')
    else:
        device = 'cuda'
        device_list = [i for i in range(torch.cuda.device_count())]
        if verbose:
            print('[INFO] CUDA is available, using GPU')

    return device, device_list, use_fa


def generate_esm_tokens(seq_dict, esm_model_dir, use_fa, tmp_dir, verbose=True):
    """Generate ESM tokens for sequences."""
    esm_model = FAEsmForMaskedLM.from_pretrained(esm_model_dir, use_fa=use_fa)

    esm_seq_data = [(seq_id, seq) for seq_id, seq in seq_dict.items()]

    if verbose:
        print('[INFO] Generating ESM tokens for sequences...')

    esm_token_dict = {}

    for seq_id, seq in tqdm(esm_seq_data):
        try:
            inputs = esm_model.tokenizer(seq, return_tensors='pt')
        except ValueError as e:
            raise ValueError(f'Error processing sequence {seq_id}: {e}')

        esm_token_dict[seq_id] = inputs

    esm_pkl_path = Path(tmp_dir) / 'esm_tokens.pkl'
    with open(esm_pkl_path, 'wb') as f:
        pickle.dump(esm_token_dict, f)

    if verbose:
        print(f'[INFO] Generated ESM tokens for {len(esm_token_dict)} sequences')
        print(f'[INFO] Saved ESM tokens to {esm_pkl_path}')

    return esm_token_dict, esm_seq_data, esm_pkl_path


def write_fasta_file(esm_seq_data, tmp_dir):
    """Write a temporary fasta file for the sequences."""
    fasta_path = f'{tmp_dir}/seq.fasta'
    
    with open(fasta_path, 'w') as f:
        for seq_id, seq in esm_seq_data:
            f.write(f'>{seq_id}\n{seq}\n')

    return fasta_path


def load_embeddings(emb_path):
    """Load the embeddings for the species."""
    if not Path(emb_path).exists():
        raise ValueError(f'Error: {emb_path} not found')

    with open(emb_path, 'rb') as f:
        bacteria_emb_dict = pickle.load(f)

    return bacteria_emb_dict


def get_embeddings(fasta_path, esm_pkl_path, saprot_pkl_path,
                   deer_model_path, esm_model_dir, saprot_model_dir,
                   device, device_list, use_fa, tmp_dir, verbose=True):
    """Get embeddings for templates using the GMPS model."""
    data_module = GMPSDataModule(fasta_path, esm_pkl_path, saprot_pkl_path, device)

    model = GMPSModel.load_from_checkpoint(deer_model_path, esm_model_dir=esm_model_dir,
                                           saprot_model_dir=saprot_model_dir,
                                           save_repr_path=f'{tmp_dir}/reprs.pkl', use_fa=use_fa)
    model.eval()

    if device == 'cpu':
        trainer = pl.Trainer(
            num_sanity_val_steps=0,
            logger=None,
            precision='bf16-mixed',
            accelerator='cpu',
            devices='auto',
            strategy='auto',
            deterministic=True,
        )
    else:
        trainer = pl.Trainer(
            num_sanity_val_steps=0,
            logger=None,
            precision='bf16-mixed',
            accelerator='gpu',
            devices=device_list,
            strategy='ddp',
            deterministic=True,
        )

    _ = trainer.predict(model=model, datamodule=data_module)

    with open(f'{tmp_dir}/reprs.pkl', 'rb') as f:
        emb_dict = pickle.load(f)

    if verbose:
        print(
            f'[INFO] Generated embeddings for {len(emb_dict)} templates, '
            f'saved at {tmp_dir}/reprs.pkl'
        )

    return emb_dict


def do_retrieval(emb_dict, bacteria_emb_dict, output_dir, top_k=-1, verbose=True):
    """Perform retrieval using embeddings and calculate similarity scores."""
    similarity_list = []

    bacteria_embs = np.array([bacteria_emb for bacteria_emb in bacteria_emb_dict.values()])

    for template_id, template_emb in tqdm(emb_dict.items()):
        template_emb = np.array(template_emb)
        # calculate the distance between the template representation and all bacteria representations
        distance = np.linalg.norm(bacteria_embs - template_emb, axis=1)

        for i, bacteria_id in enumerate(bacteria_emb_dict.keys()):
            similarity_list.append([template_id, bacteria_id, distance[i]])

    df = pd.DataFrame(similarity_list, columns=['template_id', 'bacteria_id', 'distance'])

    df.sort_values(by='distance', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if top_k > 0:
        # select the top k rows
        if verbose:
            print(f'[INFO] Selecting top {top_k} results')
            
        df = df.head(top_k)

    df.to_csv(f'{output_dir}/similarity.csv', index=False)

    if verbose:
        print(
            f'[INFO] Generated similarity scores for {len(similarity_list):,} pairs, '
            f'saved at {output_dir}/similarity.csv'
        )

    return df


def main(
        template_pdb_dir,
        database_pdb_dir,
        output_dir='./result',
        deer_model_path='./ckpt/deer_checkpoint.ckpt',
        saprot_model_dir='./ckpt/saprot_35m/',
        esm_model_dir='./ckpt/esm2_t12_35M_UR50D/',
        use_fa=True,
        tmp_dir='./tmp',
        top_k=-1,
        verbose=True
):
    """
    Args:
        template_pdb_dir (str): Path to the directory containing the template PDB files.
        database_pdb_dir (str): the path to the directory of the database containing the PDB files.
        output_dir (str): the path to the directory to save the results (CSV file), default is './result'.
        deer_model_path (str): the path to the DEER model checkpoint, default is './ckpt/deer_checkpoint.ckpt'.
        saprot_model_dir (str): the path to the directory of the SaProt model config and checkpoint, default is '../ckpt/saprot_35m'.
        esm_model_dir (str): the path to the directory of the ESM model config and checkpoint, default is './ckpt/esm2_t12_35M_UR50D/'.
        use_fa (bool): whether to use the Flash Attention in ESM, default is True.
        tmp_dir (str): the temporary directory to store intermediate files, default is './tmp'.
        top_k (int): the number of top results to keep, default is -1 (keep all).
        verbose (bool): whether to print verbose output, default is True.

    Returns:
        list: List of ranking results for each template PDB file.
    """

    pl.seed_everything(42, workers=True)

    # Step 1. generate the input tokens for the template PDB files
    # 1.1 generate the 3di tokens with foldseek
    # human template
    template_tmp_dir = Path(tmp_dir) / 'templates'
    template_tmp_dir.mkdir(parents=True, exist_ok=True)
    template_fs_3di_dict, template_seq_dict = generate_3di_tokens(template_pdb_dir, template_tmp_dir, verbose)

    # bacteria database
    database_tmp_dir = Path(tmp_dir) / 'database'
    database_tmp_dir.mkdir(parents=True, exist_ok=True)
    database_fs_3di_dict, database_seq_dict = generate_3di_tokens(database_pdb_dir, database_tmp_dir, verbose)

    # 1.2 generate the SaProt tokens
    # template
    _, template_saprot_pkl_path = generate_saprot_tokens(template_fs_3di_dict, saprot_model_dir, template_tmp_dir,
                                                         verbose)

    # database
    _, database_saprot_pkl_path = generate_saprot_tokens(database_fs_3di_dict, saprot_model_dir, database_tmp_dir,
                                                         verbose)

    # 1.3 setup device (CPU/GPU)
    device, device_list, use_fa = setup_device(use_fa, verbose)

    # 1.4 generate the ESM tokens
    # template
    _, template_esm_seq_data, template_esm_pkl_path = generate_esm_tokens(template_seq_dict, esm_model_dir, use_fa,
                                                                          template_tmp_dir, verbose)

    # database
    _, database_esm_seq_data, database_esm_pkl_path = generate_esm_tokens(database_seq_dict, esm_model_dir, use_fa,
                                                                          database_tmp_dir, verbose)

    # Step 2. generate the embeddings for the input tokens
    # 2.1 write a temporary fasta file for the sequences
    # template
    template_fasta_path = write_fasta_file(template_esm_seq_data, template_tmp_dir)

    # database
    database_fasta_path = write_fasta_file(database_esm_seq_data, database_tmp_dir)

    # Step 3. perform retrieval using the embeddings
    # 3.2 get the embeddings for the templates and databases
    template_emb_dict = get_embeddings(
        template_fasta_path, template_esm_pkl_path, template_saprot_pkl_path,
        deer_model_path, esm_model_dir, saprot_model_dir,
        device, device_list, use_fa, tmp_dir, verbose
    )

    database_emb_dict = get_embeddings(
        database_fasta_path, database_esm_pkl_path, database_saprot_pkl_path,
        deer_model_path, esm_model_dir, saprot_model_dir,
        device, device_list, use_fa, tmp_dir, verbose
    )

    # 3.3 perform retrieval using the embeddings
    results_df = do_retrieval(template_emb_dict, database_emb_dict, output_dir, top_k, verbose)
    
    return results_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DEER retrieval')

    parser.add_argument('--template_pdb_dir', type=str, required=True, help='Path to the template PDB directory')
    parser.add_argument('--database_pdb_dir', type=str, required=True, help='Path to the database PDB directory')
    parser.add_argument('--output_dir', type=str, default='./result', help='Path to the output directory')

    parser.add_argument('--deer_model_path', type=str, default='./ckpt/deer_checkpoint.ckpt',
                        help='Path to the DEER model checkpoint')
    parser.add_argument('--saprot_model_dir', type=str, default='./ckpt/saprot_35m/',
                        help='Path to the SaProt model directory')
    parser.add_argument('--esm_model_dir', type=str, default='./ckpt/esm2_t12_35M_UR50D/',
                        help='Path to the ESM model directory')
    parser.add_argument('--use_fa', action='store_true', help='Use Flash Attention in ESM')
    parser.add_argument('--tmp_dir', type=str, default='./tmp', help='Path to the temporary directory')
    parser.add_argument('--top_k', type=int, default=-1, help='Number of top results to keep (-1 to keep all)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    main(args.template_pdb_dir, args.database_pdb_dir, args.output_dir, args.deer_model_path, args.saprot_model_dir,
         args.esm_model_dir, args.use_fa, args.tmp_dir, args.top_k, args.verbose)
