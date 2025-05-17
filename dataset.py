import pickle

import lightning as pl
import numpy as np
import torch
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader


class GMPSDataset(Dataset):
    def __init__(self, fasta_path, esm_input_path, saprot_input_path, device='cuda', mode='train'):
        """
        Args:
            config (dict): configuration dictionary
            device (str): device to use
            mode (str): either 'train', 'val', or 'test',
                determines whether to sample from clusters or traverse all the data points
        """

        self.mode = mode
        self.device = device

        records = list(SeqIO.parse(fasta_path, 'fasta'))

        self.seq_dict = {record.id: str(record.seq) for record in records}
        print(f'[INFO] Dataloader Number of sequences: {len(self.seq_dict)}')

        print(f'[INFO] Dataloader: Loading ESM tokens from {esm_input_path}')
        with open(esm_input_path, 'rb') as f:
            self.esm_input_dict = pickle.load(f)

        print(f'[INFO] Dataloader: Loading SaProt tokens from {saprot_input_path}')
        with open(saprot_input_path, 'rb') as f:
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
            idx (int): index of the sequence.

        Returns:
            dict: dictionary containing the data point only
        """

        # anchor data point
        seq_id = list(self.seq_dict.keys())[idx]
        seq = self.seq_dict[seq_id]

        esm_input = self.esm_input_dict[seq_id]
        saprot_input = self.saprot_input_dict[seq_id]

        anchor_data = {
            'seq': seq,
            'label': None,
            'seq_id': seq_id,

            'esm_input': esm_input,
            'saprot_input': saprot_input
        }

        return anchor_data, None, None


class GMPSCollator:
    def __init__(self):
        
        pass

    def __call__(self, batch):
        # process triplets into tensors
        anchors, _, _ = zip(*batch)

        # process anchors
        anchor_data = self._process_batch_items(anchors)

        return {
            'anchor': anchor_data,
            'positive': None,
            'negative': None
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
            'ec_labels': None,
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
    def __init__(self, fasta_path, esm_input_path, saprot_input_path, device, num_workers=64, batch_size=64):
        """
        Args:
            fasta_path (str): path to the FASTA file containing sequences.
            esm_input_path (str): path to the ESM input file.
            saprot_input_path (str): path to the SaProt input file.
            device (str): device to use for processing.
            num_workers (int): number of workers for data loading.
            batch_size (int): batch size for data loading.
        """
        super().__init__()

        self.fasta_path = fasta_path
        self.esm_input_path = esm_input_path
        self.saprot_input_path = saprot_input_path

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.device = device

        self.collate_fn = GMPSCollator()

        self.test_dataset = None

    def setup(self, stage=None):
        if stage == 'fit':
            raise NotImplementedError('Training is not available for the server version.')

        if stage == 'predict':
            self.test_dataset = GMPSDataset(
                fasta_path=self.fasta_path,
                esm_input_path=self.esm_input_path,
                saprot_input_path=self.saprot_input_path,
                device=self.device,
                mode='test'
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
