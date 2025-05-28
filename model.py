import pickle
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
from faesm.esm import FAEsmForMaskedLM

from huggingface_hub import snapshot_download


class GMPSModel(pl.LightningModule):
    def __init__(self, ckpt_dir, save_repr_path, use_fa, download_ckpt=True):
        """
        Args:
            ckpt_dir (str): Path to the checkpoint directory.
            save_repr_path (str): Path to save the extracted representations.
            use_fa (bool): Whether to use Flash Attention in ESM.
            download_ckpt (bool): Whether to newly download the checkpoint from Hugging Face Hub.
        """
        super().__init__()

        self.use_fa = use_fa
        print(f'[INFO] Model: Using flash-attention: {self.use_fa}')

        self.save_repr_path = Path(save_repr_path)
        self.ckpt_dir = Path(ckpt_dir)
        
        esm_model_dir = Path(ckpt_dir) / 'esm2_t12_35M_UR50D'
        saprot_model_dir = Path(ckpt_dir) / 'saprot_35m'
        
        if download_ckpt:
            self.download_checkpoint()

        self.esm_model = FAEsmForMaskedLM.from_pretrained(
            esm_model_dir, use_fa=self.use_fa
        )
        self.esm_dim = 480

        self.saprot_model = FAEsmForMaskedLM.from_pretrained(
            saprot_model_dir, use_fa=self.use_fa
        )
        self.saprot_dim = 480

        # backbone model
        self.model_dim = self.esm_dim + self.saprot_dim

        hidden_dim1 = 512
        hidden_dim2 = 256
        hidden_dim3 = 128
        proj_dim = 128

        self.fc1 = nn.Linear(self.model_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)

        self.ln1 = nn.LayerNorm(self.model_dim)
        self.ln2 = nn.LayerNorm(hidden_dim1)
        self.ln3 = nn.LayerNorm(hidden_dim2)

        # classification head (auxiliary task)
        self.ln_cls = nn.LayerNorm(hidden_dim3)
        self.cls = nn.Linear(hidden_dim3, 4443)

        # contrastive head
        self.ln_contrastive = nn.LayerNorm(hidden_dim3)
        self.contrastive = nn.Linear(hidden_dim3, proj_dim)

        self.dropout = nn.Dropout(0.1)

        self.gelu = nn.GELU()

        self.test_reprs = {}

        self.save_hyperparameters()
        
    def download_checkpoint(self):
        repo_id = 'cuhkaih/deer'
        
        try:
            download_ckpt_dir = snapshot_download(repo_id=repo_id, local_dir=self.ckpt_dir)
            print(f'[INFO] checkpoint downloaded to: {download_ckpt_dir}')
        except Exception as e:
            print(f'[ERROR] Could not download checkpoint from {repo_id}. Error: {e}')
            raise e
            
    def backbone(self, data_point):
        """
        process the data with the backbone model before passing to the classification and contrastive heads

        Args:
            data_point (dict): Data for a single type of example (anchor, positive, or negative)

        Returns:
            torch.Tensor: Extracted features before classification layer
        """
        # get ESM and SaProt embeddings
        # dictionary containing input_ids and attention_mask
        x = {
            'input_ids': data_point['esm_input_ids'],
            'attention_mask': data_point['esm_attention_mask']
        }
        x_esm = self.esm_model(**x)['last_hidden_state']

        x = {
            'input_ids': data_point['saprot_input_ids'],
            'attention_mask': data_point['saprot_attention_mask']
        }
        x_saprot = self.saprot_model(**x)['last_hidden_state']

        # extract the BOS representation (first token) for classification
        # concatenate the BOS representations from both models
        x = torch.cat((x_esm[:, 0], x_saprot[:, 0]), dim=-1)  # (B, esm_dim + saprot_dim)

        # pass through fully connected layers later in the classifier
        x = self.gelu(self.fc1(self.ln1(x)))
        x = self.dropout(x)

        x = self.gelu(self.fc2(self.ln2(x)))
        x = self.dropout(x)

        x = self.gelu(self.fc3(self.ln3(x)))
        x = self.dropout(x)

        return x

    def extract_features(self, x):
        """
        Extract features from the input data using the backbone model

        Returns:
            torch.Tensor: Extracted features
        """
        # apply the contrastive head
        x = self.contrastive(self.ln_contrastive(x))

        return x

    def forward(self, batch):
        """
        Forward pass for single batch of data

        Args:
            batch (dict): Batch containing input data
        """
        # for triplet-based contrastive learning
        x = self.backbone(batch['anchor'])
        anchor_features = self.extract_features(x)

        return {
            'anchor': {'logits': None, 'features': anchor_features},
            'positive': None,
            'negative': None
        }

    def on_predict_start(self):
        self.predict_reprs = {}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # 1. forward
        outputs = self(batch)
        anchor_feats = outputs['anchor']['features']  # (B, H)

        # 2. move all reprs to CPU together
        reprs_cpu = anchor_feats.float().detach().cpu().numpy()  # (B, H)

        # 3. iterate per sample
        seq_ids = batch['anchor']['seq_ids']
        for i, seq_id in enumerate(seq_ids):
            self.predict_reprs[seq_id] = reprs_cpu[i]

    def on_predict_epoch_end(self):
        # if performing distributed inference, gather into rank-0
        if self.trainer.world_size > 1:
            all_reprs = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(all_reprs, self.predict_reprs)
            if self.trainer.is_global_zero:
                # merge
                merged_repr = {}

                for r in all_reprs:
                    merged_repr.update(r)
                self.predict_reprs = merged_repr
        # else: single-node, nothing to do

        # only the main process writes out
        if self.trainer.is_global_zero:
            self.save_repr_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.save_repr_path, 'wb') as f:
                pickle.dump(self.predict_reprs, f)

            return self.predict_reprs
