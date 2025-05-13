import json
import pickle
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from faesm.esm import FAEsmForMaskedLM


class GMPSModel(pl.LightningModule):
    def __init__(self, config):
        """
        Args:
            config (dict): dictionary containing the configuration
        """
        super().__init__()

        self.config = config

        # general training parameters
        self.lr = config['train']['lr']
        self.weight_decay = config['train']['weight_decay']

        # contrastive learning parameters
        self.contrastive_weight = config['train'].get('contrastive_weight', 1.0)
        self.ec_weight = config['train'].get('ec_weight', 1.0)
        self.supcon_weight = config['train'].get('supcon_weight', 0.0)

        # -----------------------------------------
        # define the DEER model
        # -----------------------------------------
        self.cls_num = config['data']['cls_num']  # int, how many EC classes in the training dataset

        self.esm_arch = config['model']['esm_arch']
        self.saprot_arch = config['model']['saprot_arch']

        self.use_fa = bool(config['model']['use_fa'])
        print(f'Using flash-attention: {self.use_fa}')

        match self.esm_arch:
            case 'esm2_35m':
                self.esm_model = FAEsmForMaskedLM.from_pretrained(
                    f'./ckpt/esm2_t12_35M_UR50D/', use_fa=self.use_fa
                )
                self.esm_dim = 480

            # otherwise
            case _:
                raise NotImplementedError(f'Unknown ESM architecture: {self.esm_arch}')

        # get the saprot model
        match self.saprot_arch:
            case 'saprot_35m':
                self.saprot_model = FAEsmForMaskedLM.from_pretrained(
                    self.config['model']['saprot_model_dir'], use_fa=self.use_fa
                )
                self.saprot_dim = 480

            case _:
                raise NotImplementedError(f'Unknown SaProt architecture: {self.saprot_arch}')

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
        self.cls = nn.Linear(hidden_dim3, self.cls_num)

        # contrastive head
        self.ln_contrastive = nn.LayerNorm(hidden_dim3)
        self.contrastive = nn.Linear(hidden_dim3, proj_dim)

        self.dropout = nn.Dropout(0.1)

        self.gelu = nn.GELU()

        # -----------------------------------------
        # loss settings
        # -----------------------------------------
        self.criterion = nn.TripletMarginLoss(margin=1.0, p=2)

        # -----------------------------------------
        # prepare the EC labels
        # -----------------------------------------

        with open(config['data']['ec2num_dict_path'], 'r') as f:
            self.ec2num = json.load(f)  # this is the mapping from a string EC number to a number

        self.num2ec = {v: k for k, v in
                       self.ec2num.items()}  # this is the reverse mapping from a number to a string EC number

        self.test_results = {}
        self.test_reprs = {}

        self.save_hyperparameters()

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

    def classifier(self, x):
        """
        Process the <CLS> token through the classification head
        """

        # cls head
        x = self.cls(self.ln_cls(x))

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
        anchor_logits = self.classifier(x)

        x = self.backbone(batch['positive'])
        positive_features = self.extract_features(x)
        positive_logits = self.classifier(x)

        x = self.backbone(batch['negative'])
        negative_features = self.extract_features(x)
        negative_logits = self.classifier(x)

        return {
            'anchor': {'logits': anchor_logits, 'features': anchor_features},
            'positive': {'logits': positive_logits, 'features': positive_features},
            'negative': {'logits': negative_logits, 'features': negative_features}
        }

    def compute_triplet_loss(self, anchor_features, positive_features, negative_features):
        """
        Compute triplet contrastive loss

        Args:
            anchor_features (torch.Tensor): Features from anchor samples
            positive_features (torch.Tensor): Features from positive samples
            negative_features (torch.Tensor): Features from negative samples

        Returns:
            torch.Tensor: Triplet loss value
        """

        return self.criterion(anchor_features, positive_features, negative_features)

    def compute_supcon_hard_loss(self, model_emb, temp=0.1, n_pos=1):
        """
        Calculate the Supervised Contrastive-Hard loss from CLEAN (Science, 2023).
        This was not switched on but monitored during training due to similar effectiveness to the triplet loss.

        Args:
            model_emb (tensor): shape = (B, N, D),
                                where B is batch size,
                                N is the number of examples (1 anchor + n_pos + n_neg),
                                D is the embedding dimension
            temp (float): temperature parameter
            n_pos (int): number of positive examples per anchor

        Returns:
            loss (tensor): computed loss value
        """

        # l2 normalize every embedding
        features = F.normalize(model_emb, dim=-1, p=2)
        # print(features.shape)
        # features_T is [bsz, outdim, n_all], for performing batch dot product
        features_T = torch.transpose(features, 1, 2)
        # print(features_T.shape)
        # anchor is the first embedding
        anchor = features[:, 0]
        # print(anchor.shape)
        # anchor is the first embedding
        anchor_dot_features = torch.bmm(anchor.unsqueeze(1), features_T) / temp
        # anchor_dot_features now [bsz, n_all], contains
        anchor_dot_features = anchor_dot_features.squeeze(1)
        # deduct by max logits, which will be 1/temp since features are L2 normalized
        logits = anchor_dot_features - 1 / temp
        # the exp(z_i dot z_a) excludes the dot product between itself
        # exp_logits is of size [bsz, n_pos+n_neg]
        exp_logits = torch.exp(logits[:, 1:])
        exp_logits_sum = n_pos * torch.log(exp_logits.sum(1))  # size [bsz], scale by n_pos
        pos_logits_sum = logits[:, 1:n_pos + 1].sum(1)  # sum over all (anchor dot pos)
        log_prob = (pos_logits_sum - exp_logits_sum) / n_pos
        loss = - log_prob.mean()

        return loss

    def compute_accuracy(self, probs, target):
        """
        Args:
            probs: (B, N), predicted probabilities for each class (i.e., normalized logits)
            target: (B,) target labels.
        """
        # Get the predicted class by finding the index of the max probability
        preds = torch.argmax(probs, dim=1)

        # Calculate accuracy
        acc = (preds == target).float().mean()

        return acc, (preds == target).float().tolist()

    def compute_contrastive_accuracy(self, anchor_features, positive_features, negative_features):
        """
        Compute contrastive accuracy based on the triplet feature distances

        Args:
            anchor_features (torch.Tensor): Features from anchor samples, shape (B, D)
            positive_features (torch.Tensor): Features from positive samples, shape (B, D)
            negative_features (torch.Tensor): Features from negative samples, shape (B, D)

        Returns:
            acc (float): Contrastive accuracy
        """

        pdist = nn.PairwiseDistance(p=2)

        pos_dist = pdist(anchor_features, positive_features)
        neg_dist = pdist(anchor_features, negative_features)

        # Compute the accuracy based on the distances
        pos_correct = (pos_dist < neg_dist).sum()

        acc = pos_correct.float() / anchor_features.size(0)

        return acc

    def training_step(self, batch, batch_idx):
        # Process the triplet batch
        outputs = self(batch)

        # Extract anchor data for classification
        anchor_logits = outputs['anchor']['logits']
        anchor_features = outputs['anchor']['features']
        anchor_probs = F.softmax(anchor_logits, dim=1)

        pos_logits = outputs['positive']['logits']
        pos_features = outputs['positive']['features']
        pos_probs = F.softmax(pos_logits, dim=1)
        pos_labels = batch['positive']['ec_labels']

        neg_logits = outputs['negative']['logits']
        neg_features = outputs['negative']['features']
        neg_probs = F.softmax(neg_logits, dim=1)
        neg_labels = batch['negative']['ec_labels']

        ec_labels = batch['anchor']['ec_labels']

        # --------------------------------------
        # compute the loss
        # --------------------------------------
        # 1. loss for the EC number prediction head
        ec_loss = F.cross_entropy(anchor_logits, ec_labels, reduction='mean')
        self.log('train_loss_ec', ec_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(ec_labels),
                 sync_dist=True)

        # 2. loss for the contrastive learning (cl)
        contrastive_loss = self.compute_triplet_loss(
            anchor_features,
            pos_features,
            neg_features
        )
        self.log('train_loss_cl', contrastive_loss, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=len(ec_labels), sync_dist=True)

        # 3. monitor the supervised contrastive loss
        # first, convert the (B, D) features to (B, N, D)
        features = torch.stack([anchor_features, pos_features, neg_features], dim=1)
        supcon_loss = self.compute_supcon_hard_loss(features)
        self.log('train_loss_supcon', supcon_loss, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=len(ec_labels), sync_dist=True)

        # 3. total loss
        total_loss = (self.ec_weight * ec_loss +
                      self.contrastive_weight * contrastive_loss +
                      self.supcon_weight * supcon_loss)

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(ec_labels),
                 sync_dist=True)

        # --------------------------------------
        # compute the accuracy
        # --------------------------------------
        # 1. compute and accuracy of whether the anchor is classified correctly
        acc, _ = self.compute_accuracy(anchor_probs, ec_labels)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(ec_labels),
                 sync_dist=True)

        # 2. compute the accuracy for all three samples
        all_probs = torch.cat([anchor_probs, pos_probs, neg_probs], dim=0)
        all_labels = torch.cat([ec_labels, pos_labels, neg_labels], dim=0)
        all_acc, _ = self.compute_accuracy(all_probs, all_labels)
        self.log('train_acc_triplet', all_acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(all_labels),
                 sync_dist=True)

        # 3. compute the accuracy for whether the anchor is closer to the positive than the negative
        # this is defined as the contrastive accuracy
        contrastive_acc = self.compute_contrastive_accuracy(anchor_features, pos_features, neg_features)
        self.log('train_acc_cl', contrastive_acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(ec_labels),
                 sync_dist=True)

        return total_loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        # process the triplet batch
        outputs = self(batch)

        # extract anchor data for classification
        anchor_logits = outputs['anchor']['logits']
        anchor_features = outputs['anchor']['features']
        anchor_probs = F.softmax(anchor_logits, dim=1)
        ec_labels = batch['anchor']['ec_labels']

        pos_features = outputs['positive']['features']
        neg_features = outputs['negative']['features']

        # --------------------------------------
        # compute the loss
        # --------------------------------------
        # 1. loss for the EC number prediction head
        ec_loss = F.cross_entropy(anchor_logits, ec_labels, reduction='mean')
        self.log('val_loss_ec', ec_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(ec_labels),
                 sync_dist=True)

        # 2. loss for the contrastive learning (cl)
        contrastive_loss = self.compute_triplet_loss(
            anchor_features,
            outputs['positive']['features'],
            outputs['negative']['features']
        )
        self.log('val_loss_cl', contrastive_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(ec_labels),
                 sync_dist=True)
        # self.val_contrastive_loss.append(contrastive_loss.item())

        # 3. monitor the supervised contrastive loss
        # first, convert the (B, D) features to (B, N, D)
        features = torch.stack([anchor_features, pos_features, neg_features], dim=1)
        supcon_loss = self.compute_supcon_hard_loss(features)
        self.log('val_loss_supcon', supcon_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(ec_labels),
                 sync_dist=True)

        # 3. total loss
        total_loss = self.ec_weight * ec_loss + self.contrastive_weight * contrastive_loss + self.supcon_weight * supcon_loss
        self.log('val_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(ec_labels),
                 sync_dist=True)

        # --------------------------------------
        # compute the accuracy
        # --------------------------------------
        # 1. compute and accuracy of whether the anchor is classified correctly
        acc, correctness_list = self.compute_accuracy(anchor_probs, ec_labels)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(ec_labels), sync_dist=True)

        # 2. compute the accuracy for all three samples in a triplet
        contrastive_acc = self.compute_contrastive_accuracy(anchor_features, pos_features, neg_features)
        self.log('val_acc_cl', contrastive_acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(ec_labels),
                 sync_dist=True)

        return total_loss

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        # process the batch (which may contain multiple samples)
        outputs = self(batch)
        anchor_logits = outputs['anchor']['logits']

        # apply softmax
        anchor_probs = F.softmax(anchor_logits, dim=1)

        # get the top5 predictions for all samples in batch
        top5_idx = torch.topk(anchor_probs, 5, dim=1)

        # also retrieve the anchor representations
        anchor_reprs = outputs['anchor']['features']

        # process each sample in the batch and add to the results
        batch_size = len(batch['anchor']['seq_ids'])
        for i in range(batch_size):
            seq_id = batch['anchor']['seq_ids'][i]

            # get top5 predictions and probabilities for this sample
            top5_pred = [self.num2ec[top5_idx.indices[i, j].item()] for j in range(5)]
            top5_prob = [top5_idx.values[i, j].item() for j in range(5)]

            # store results for this sequence
            self.test_results[seq_id] = {'top5_pred': top5_pred, 'top5_prob': top5_prob}

            # save the anchor representation
            self.test_reprs[seq_id] = anchor_reprs[i].float().detach().cpu().numpy()

    def on_test_epoch_end(self):
        save_path = Path(self.config['train']['save_result_dir']) / f'ec_preds.json'

        print(f'Saving test results to {save_path}')
        print(f'Number of test results: {len(self.test_results)}')

        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(self.test_results, f, indent=4)

        self.test_results = {}

        # Save the anchor representations as pickle
        reprs_path = Path(self.config['train']['save_result_dir']) / f'reprs.pkl'
        print(f'Saving test representations to {reprs_path}')

        with open(reprs_path, 'wb') as f:
            pickle.dump(self.test_reprs, f)

        self.test_reprs = {}

    def on_predict_start(self):
        # pre‐allocate result containers
        self.predict_results = {}
        self.predict_reprs = {}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # 1. forward
        outputs = self(batch)
        anchor_logits = outputs['anchor']['logits']
        anchor_feats = outputs['anchor']['features']  # (B, H)

        # 2. softmax + top-5
        anchor_probs = F.softmax(anchor_logits, dim=1)
        top5 = torch.topk(anchor_probs, k=5, dim=1)
        top5_vals, top5_idx = top5.values, top5.indices  # both (B, 5)

        # 3. move all reprs to CPU together
        reprs_cpu = anchor_feats.float().detach().cpu().numpy()  # (B, H)

        # 4. iterate per sample
        seq_ids = batch['anchor']['seq_ids']
        for i, seq_id in enumerate(seq_ids):
            # map indices → labels
            preds = [self.num2ec[idx.item()] for idx in top5_idx[i]]
            probs = top5_vals[i].tolist()  # already floats

            self.predict_results[seq_id] = {
                'top5_pred': preds,
                'top5_prob': probs
            }
            self.predict_reprs[seq_id] = reprs_cpu[i]

    def on_predict_epoch_end(self):
        # if performing distributed inference, gather into rank-0
        if self.trainer.world_size > 1:
            all_results = [None] * self.trainer.world_size
            all_reprs = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(all_results, self.predict_results)
            torch.distributed.all_gather_object(all_reprs, self.predict_reprs)
            if self.trainer.is_global_zero:
                # merge
                merged_res = {}
                merged_repr = {}
                for r in all_results:
                    merged_res.update(r)
                for r in all_reprs:
                    merged_repr.update(r)
                self.predict_results = merged_res
                self.predict_reprs = merged_repr
        # else: single-node, nothing to do

        # only the main process writes out
        if self.trainer.is_global_zero:
            save_dir = Path(self.config['test']['save_result_dir'])
            save_dir.mkdir(parents=True, exist_ok=True)

            # JSON without indent for speed/compactness
            with open(save_dir / 'ec_preds.json', 'w') as f:
                json.dump(self.predict_results, f)

            with open(save_dir / 'reprs.pkl', 'wb') as f:
                pickle.dump(self.predict_reprs, f)

            print(f'Saved {len(self.predict_results)} predictions to {save_dir}')

        # free memory
        self.predict_results.clear()
        self.predict_reprs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        return optimizer
