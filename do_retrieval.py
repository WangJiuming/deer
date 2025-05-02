import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def do_retrieval(metadata_path, repr_pkl_path, save_sim_pkl_path):
    """
    Perform dense retrieval of enzyme representations and calculate pairwise similarities.

    Args:
        metadata_path (str or Path): Path to the metadata CSV file containing enzyme information
        repr_pkl_path (str or Path): Path to the pickle file containing protein representations
        save_sim_pkl_path (str or Path): Path to the output PKL file for saving the pairwise similarities
    """
    # load the metadata
    metadata_df = pd.read_csv(metadata_path)

    # load representations
    with open(repr_pkl_path, 'rb') as f:
        repr_dict = pickle.load(f)  # seq id to representation

    print('Number of representations:', len(repr_dict))

    bacteria_ids = metadata_df[metadata_df['organism'] == 'bacteria']['uniprot_id'].values
    eukaryota_ids = metadata_df[metadata_df['organism'] != 'bacteria']['uniprot_id'].values  # not bacteria is eukaryota

    print(f'Number of bacteria IDs: {len(bacteria_ids)}')
    print(f'Number of eukaryota IDs: {len(eukaryota_ids)}')

    # get the similarity between the bacteria enzymes and eukaryota enzymes (templates)
    similarity_list = []

    # get an array of the bacteria representations (pre-index)
    bacteria_reprs = np.array([repr_dict[bacteria_id] for bacteria_id in bacteria_ids])

    # get an array of the eukaryota representations
    for eukaryota_id in eukaryota_ids:
        eukaryota_repr = np.array(repr_dict[eukaryota_id])

        # calculate the distance between the eukaryota representation and all bacteria representations
        distance = np.linalg.norm(bacteria_reprs - eukaryota_repr, axis=1)

        for i, bacteria_id in enumerate(bacteria_ids):
            similarity_list.append([eukaryota_id, bacteria_id, distance[i]])

    df = pd.DataFrame(similarity_list, columns=['eukaryota_id', 'bacteria_id', 'distance'])

    # save as pickle
    df.to_pickle(save_sim_pkl_path)

    print('Retrieval completed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--metadata_path', type=str, default='./data/test_metadata.csv', required=False)
    parser.add_argument('--repr_pkl_path', type=str, default='./result/reprs.pkl', required=False)
    parser.add_argument('--save_sim_pkl_path', type=str, default='./result/similarity.pkl', required=False)

    args = parser.parse_args()

    metadata_path = Path(args.metadata_path)
    repr_pkl_path = Path(args.repr_pkl_path)
    save_sim_pkl_path = Path(args.save_sim_pkl_path)

    do_retrieval(metadata_path, repr_pkl_path, save_sim_pkl_path)
