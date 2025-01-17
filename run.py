import argparse
import os

import wandb

import datasets
import models
from omegaconf import OmegaConf
import pytorch_lightning as pl

from utils.utils import deep_update, todict


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str)
    parser.add_argument('-n', '--name', type=str)
    parser.add_argument('-p', '--ckpt', type=str)
    parser.add_argument('-m', '--mode', type=str, default='train')
    parser.add_argument('-pr', '--project', type=str, default='M2Mixer')

    parser.add_argument('--disable-wandb', action='store_true', default=False)
    args, unknown = parser.parse_known_args()
    return args, unknown


if __name__ == '__main__':
    args, unknown = parse_args()
    cfg = OmegaConf.load(args.cfg)
    train_cfg = cfg.train
    dataset_cfg = cfg.dataset
    model_cfg = cfg.model
    pl.seed_everything(train_cfg.seed)
    unknown = [u.replace('--', '') for u in unknown]
    ucfg = OmegaConf.from_cli(unknown)
    if 'model' in ucfg:
        deep_update(model_cfg, ucfg.model)
    if 'train' in ucfg:
        deep_update(train_cfg, ucfg.train)
    if 'dataset' in ucfg:
        deep_update(dataset_cfg, ucfg.dataset)

    if args.disable_wandb:
        wandb.init(project=args.project, name=args.name, config=todict(cfg), mode='disabled')
    else:
        wandb.init(project=args.project, name=args.name, config=todict(cfg))

    model = models.get_model(model_cfg.type)
    if args.ckpt:
        train_module = model.load_from_checkpoint(args.ckpt, optimizer_cfg=train_cfg.optimizer,
                                                  model_cfg=model_cfg)
    else:
        train_module = model(model_cfg, train_cfg.optimizer)
    wandb.watch(train_module)
    data_module = datasets.get_data_module(dataset_cfg.type)
    if dataset_cfg.params.num_workers == -1:
        dataset_cfg.params.num_workers = os.cpu_count()
    data_module = data_module(**dataset_cfg.params)

    trainer = pl.Trainer(
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min'),
            pl.callbacks.ModelCheckpoint(
                monitor=train_cfg.monitor,
                save_last=True,
                save_top_k=5,
                mode=train_cfg.monitor_mode
            )
        ],
        accelerator='gpu',
        devices=-1,
        log_every_n_steps=train_cfg.log_interval_steps,
        logger=pl.loggers.TensorBoardLogger(train_cfg.tensorboard_path, args.name),
        max_epochs=train_cfg.epochs
    )
    wandb.config.update({"run_version": trainer.logger.version})
    if args.mode == 'train':
        try:
            trainer.fit(train_module, data_module)
        except KeyboardInterrupt:
            print('KeyboardInterrupt: Trying to test with the current best model')
        trainer.test(train_module, data_module, ckpt_path='best')
    if args.mode == 'test':
        trainer.test(train_module, data_module)
