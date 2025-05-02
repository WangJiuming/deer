import argparse
import yaml

import torch
import lightning as pl

from dataset import GMPSDataModule
from model import GMPSModel


def main():
    parser = argparse.ArgumentParser(description='Test DEER model')
    parser.add_argument('--config', type=str, default='./config/config_test.yaml',
                        help='path to the config file')

    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')

    # parse the config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config['experiment']['seed'], workers=True)

    devices = [int(d) for d in config['experiment']['device'].split(',')]
    print(f'Using device: {devices}')

    data_module = GMPSDataModule(config)

    model = GMPSModel.load_from_checkpoint(config['test']['ckpt_path'], config=config)
    model.eval()

    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        logger=None,
        precision='bf16-mixed',
        accelerator='gpu',
        devices=devices,
        strategy='ddp',
        deterministic=True,
    )

    trainer.predict(model=model, datamodule=data_module)


if __name__ == '__main__':
    main()
