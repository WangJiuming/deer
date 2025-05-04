import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import lightning as pl
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader


class GMPSDataset(Dataset):
    def __init__(self, config, device='cuda', mode='train'):
        """
        Args:
            config (dict): configuration dictionary
            device (str): device to use
            mode (str): either 'train', 'val', or 'test',
                determines whether to sample from clusters or traverse all the data points
        """

        self.mode = mode

        self.device = device
        self.fasta_path = config['data'][f'{mode}_fasta_path']
        self.meta_path = config['data'][f'{mode}_meta_path']

        ec2label_dict_path = config['data']['ec2num_dict_path']

        records = list(SeqIO.parse(self.fasta_path, 'fasta'))
        self.seq_dict = {record.id: str(record.seq) for record in records}

        meta_df = pd.read_csv(self.meta_path)
        self.label_dict = {seq_id: ec for seq_id, ec in zip(meta_df['seq_id'], meta_df['ec_number'])}

        self.ec2seq_ids = defaultdict(list)
        for seq_id, ec in self.label_dict.items():
            self.ec2seq_ids[ec].append(seq_id)

        with open(ec2label_dict_path, 'r') as f:
            self.ec2label = json.load(f)  # seq id -> EC number

        print(f'Number of labels: {len(self.ec2label)}')

        print('Loading ESM tokens...')

        self.esm_arch = config['model']['esm_arch']

        esm_inputs_path = Path(config['data'][f'{mode}_esm_input_pkl_dir'],
                               f'{mode}_{self.esm_arch}_inputs.pkl')

        print(f'Loading ESM inputs from {esm_inputs_path}')
        with open(esm_inputs_path, 'rb') as f:
            self.esm_input_dict = pickle.load(f)

        # load the SaProt tokens
        self.saprot_arch = config['model']['saprot_arch']

        saprot_inputs_path = Path(config['data'][f'{mode}_saprot_input_pkl_dir'],
                                  f'{mode}_{self.saprot_arch}_inputs.pkl')
        print(f'Loading SaProt inputs from {saprot_inputs_path}')
        with open(saprot_inputs_path, 'rb') as f:
            self.saprot_input_dict = pickle.load(f)

    def __len__(self):
        """
        Return the number of data points in the dataset.
        """
        return len(self.seq_dict)

    def __getitem__(self, idx):
        """
        Return a data point and its corresponding positive and negative samples.
        Each data point is a tuple of (seq: str, EC label: int, seq ID: str, ESM tokens: array, SaProt tokens: array).
        Note that train and val/test scenarios should be distinguished here
        since there could be no EC from another family during testing.

        Args:
            idx (int): index of the EC class for the anchor data point.

        Returns:
            dict: dictionary containing the data point and
                its corresponding positive and negative samples.
        """

        # anchor data point
        seq_id = list(self.seq_dict.keys())[idx]
        seq = self.seq_dict[seq_id]
        ec = str(self.label_dict[seq_id])
        label = self.ec2label[ec]

        esm_input = self.esm_input_dict[seq_id]
        saprot_input = self.saprot_input_dict[seq_id]

        anchor_data = {
            'seq': seq,
            'label': label,
            'seq_id': seq_id,

            'esm_input': esm_input,
            'saprot_input': saprot_input
        }

        # positive data point from the same EC family
        pos_seq_id = random.choice(self.ec2seq_ids[ec])
        pos_seq = self.seq_dict[pos_seq_id]
        pos_label = self.ec2label[ec]

        pos_esm_input = self.esm_input_dict[pos_seq_id]
        pos_saprot_input = self.saprot_input_dict[pos_seq_id]

        pos_data = {
            'seq': pos_seq,
            'label': pos_label,
            'seq_id': pos_seq_id,

            'esm_input': pos_esm_input,
            'saprot_input': pos_saprot_input
        }

        # negative data point from a different EC family
        neg_ec = random.choice(list(self.ec2seq_ids.keys()))

        # this negative sample must be different from the anchor EC if performing training (i.e., during optimization)
        # if doing testing, we can sample from the same EC family since it is not used
        if self.mode == 'train':
            while neg_ec == ec:
                neg_ec = random.choice(list(self.ec2seq_ids.keys()))

        neg_seq_id = random.choice(self.ec2seq_ids[neg_ec])
        neg_seq = self.seq_dict[neg_seq_id]
        neg_label = self.ec2label[neg_ec]
        neg_esm_input = self.esm_input_dict[neg_seq_id]
        neg_saprot_input = self.saprot_input_dict[neg_seq_id]

        neg_data = {
            'seq': neg_seq,
            'label': neg_label,
            'seq_id': neg_seq_id,

            'esm_input': neg_esm_input,
            'saprot_input': neg_saprot_input
        }

        return anchor_data, pos_data, neg_data


class GMPSCollator:
    def __init__(self, config):
        """
        Args:
            config (dict): configuration dictionary.
        """
        self.config = config

    def __call__(self, batch):
        # process triplets into tensors
        anchors, positives, negatives = zip(*batch)

        # process anchors
        anchor_data = self._process_batch_items(anchors)

        # process positives
        positive_data = self._process_batch_items(positives)

        # process negatives
        negative_data = self._process_batch_items(negatives)

        return {
            'anchor': anchor_data,
            'positive': positive_data,
            'negative': negative_data
        }

    def _process_batch_items(self, items):
        """
        Process a batch of items into tensor format and pad them.

        Args:
            items (list): list of items to process.
        Returns:
            dict: dictionary containing the processed items.
        """
        # extract sequences and find max length
        seqs = [item['seq'] for item in items]
        max_seq_len = max(len(seq) for seq in seqs)

        # convert sequences to one-hot encoding
        seqs = np.array([self.seq_to_onehot(seq, max_seq_len) for seq in seqs])
        seqs_one_hot = torch.from_numpy(seqs)

        # extract labels
        ec_labels = [item['label'] for item in items]
        ec_labels = torch.tensor(ec_labels, dtype=torch.long)

        # extract sequence IDs
        seq_id_list = [item['seq_id'] for item in items]

        # process ESM tokens
        # need to pad the input_ids with the pad token (1) and attention mask with 0
        esm_input_ids = [item['esm_input']['input_ids'].squeeze() for item in items]
        esm_attention_mask = [item['esm_input']['attention_mask'].squeeze() for item in items]

        # then pad
        esm_input_ids = torch.nn.utils.rnn.pad_sequence(
            esm_input_ids, batch_first=True, padding_value=1).long()
        esm_attention_mask = torch.nn.utils.rnn.pad_sequence(
            esm_attention_mask, batch_first=True, padding_value=0).long()

        # process SaProt tokens
        saprot_input_ids = [item['saprot_input']['input_ids'].squeeze() for item in items]
        saprot_attention_mask = [item['saprot_input']['attention_mask'].squeeze() for item in items]

        # pad SaProt tokens
        saprot_input_ids = torch.nn.utils.rnn.pad_sequence(
            saprot_input_ids, batch_first=True, padding_value=1).long()
        saprot_attention_mask = torch.nn.utils.rnn.pad_sequence(
            saprot_attention_mask, batch_first=True, padding_value=0).long()

        return {
            'seq_one_hots': seqs_one_hot,
            'ec_labels': ec_labels,
            'seq_ids': seq_id_list,

            'esm_input_ids': esm_input_ids,
            'esm_attention_mask': esm_attention_mask,

            'saprot_input_ids': saprot_input_ids,
            'saprot_attention_mask': saprot_attention_mask,
        }

    @staticmethod
    def seq_to_onehot(seq, pad_len):
        """
        Converts a sequence to one-hot encoding.

        Args:
            seq (str): sequence to convert.
            pad_len (int): length to pad the sequence to.
        Returns:
            torch.Tensor: one-hot encoding of the sequence.
        """
        uncommon_aa_set = {'U', 'O', 'B', 'Z', 'J'}

        # remove uncommon amino acids
        seq = ''.join(aa if aa not in uncommon_aa_set else 'X' for aa in seq)

        aa_set = {
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
            'X'
        }
        aa_dict = {aa: i for i, aa in enumerate(aa_set)}

        # convert sequence to one-hot encoding
        one_hot = np.zeros((pad_len, len(aa_set)))
        for i, aa in enumerate(seq):
            if aa in aa_dict:
                one_hot[i, aa_dict[aa]] = 1
            else:
                print(f'Invalid amino acid: {aa}')

        # convert one-hot to 1D numeric representation using index
        one_hot = np.argmax(one_hot, axis=1)

        return one_hot


class GMPSDataModule(pl.LightningDataModule):
    def __init__(self, config):
        """
        Args:
            config (dict): configuration dictionary
        """
        super().__init__()
        self.config = config

        self.ec_cls_num = self.config['data']['cls_num']

        self.num_workers = self.config['data']['num_workers']
        self.batch_size = self.config['train']['batch_size']
        self.device = self.config['experiment']['device']

        self.collate_fn = GMPSCollator(self.config)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == 'fit':

            self.train_dataset = GMPSDataset(
                config=self.config,
                mode='train'
            )

            self.val_dataset = GMPSDataset(
                config=self.config,
                mode='val'
            )

            print(f'Train dataset length: {len(self.train_dataset)}')
            print(f'Validation dataset length: {len(self.val_dataset)}')

        if stage == 'test' or stage == 'predict':
            self.test_dataset = GMPSDataset(
                config=self.config,
                mode='test'
            )

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=True
        )

    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=True
        )

    def test_dataloader(self):

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=True
        )

    def predict_dataloader(self):

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            pin_memory=True
        )
