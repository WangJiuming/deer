import argparse
import datetime
from pathlib import Path

import lightning as pl
import torch
import yaml
from dataset import GMPSDataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from model import GMPSModel


def main():
    parser = argparse.ArgumentParser(description='Train DEER')
    parser.add_argument('--config', type=str, default='./config/config_train.yaml', help='path to the config file')

    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')

    # parse the config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config['experiment']['seed'], workers=True)

    # set up the loggers
    run_id = f'{datetime.datetime.now():%Y%m%d}_{config["experiment"]["name"]}'

    # make model checkpoint directory
    ckpt_dir = Path(config['train']['save_ckpt_dir'], run_id)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # update the save result directory with the run id and make the directory
    config['train']['save_result_dir'] = Path(config['train']['save_result_dir'], run_id)
    config['train']['save_result_dir'].mkdir(parents=True, exist_ok=True)

    data_module = GMPSDataModule(config)

    model = GMPSModel(config)
    print(model)

    # devices are expected to be inputted as a string of int separated by commas without any spacing
    devices = [int(d) for d in config['experiment']['device'].split(',')]
    print(f'Using device: {devices}')

    # callback for saving the last epoch
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),  # directory to save checkpoints
        filename='last_checkpoint_{train_loss:.4f}',  # include train_loss in filename
        monitor='train_loss',
        save_top_k=0,  # only save the last model
        mode='min',  # save the best model based on the validation loss
        save_last=True  # the other callback will save the last model
    )

    trainer = pl.Trainer(
        max_epochs=config['train']['epochs'],
        callbacks=[last_checkpoint_callback],
        num_sanity_val_steps=0,
        check_val_every_n_epoch=config['train']['val_interval'],
        logger=None,
        precision='bf16-mixed',
        accelerator='gpu',
        devices=devices,
        strategy='ddp_find_unused_parameters_true',
        deterministic=True,
    )

    trainer.fit(model=model, datamodule=data_module)


if __name__ == '__main__':
    main()
