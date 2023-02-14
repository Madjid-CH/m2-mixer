import abc
from typing import List, Dict, Any

import numpy as np
import torch
import wandb
from omegaconf import DictConfig
import pytorch_lightning as pl
from torchmetrics import Metric


class AbstractTrainTestModule(pl.LightningModule, abc.ABC):
    def __init__(self, optimizer_cfg: DictConfig, log_confusion_matrix: bool = False, **kwargs):
        self.optimizer_cfg = optimizer_cfg
        self.log_confusion_matrix = log_confusion_matrix
        self.loss_pos_weight = torch.tensor(optimizer_cfg.loss_pos_weight) if 'loss_pos_weight' in optimizer_cfg \
            else None
        if 'loss_pos_weight' in optimizer_cfg:
            self.optimizer_cfg.pop('loss_pos_weight')

        super(AbstractTrainTestModule, self).__init__(**kwargs)
        self.criterion = self.setup_criterion()
        self.train_scores, self.val_scores, self.test_scores = self.setup_scores()
        if any([isinstance(self.train_scores, list), isinstance(self.val_scores, list), isinstance(self.test_scores, list)]):
            raise ValueError('Scores must be a dict')
        self.best_epochs = {'val_loss': None}
        if self.val_scores is not None:
            for metric in self.val_scores:
                self.best_epochs[metric] = None

        self.logged_n_paarameters = False

    @abc.abstractmethod
    def setup_criterion(self) -> torch.nn.Module:
        raise NotImplementedError

    @abc.abstractmethod
    def setup_scores(self) -> List[Dict[str, Metric]]:
        raise NotImplementedError

    @abc.abstractmethod
    def shared_step(self, batch) -> Dict[str, Any]:
        raise NotImplementedError

    def log_n_parameters(self):
        if not self.logged_n_paarameters:
            trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total_parameters = sum(p.numel() for p in self.parameters())

            wandb.run.summary['trainable_parameters'] = trainable_parameters
            wandb.run.summary['total_parameters'] = total_parameters
            self.logged_n_paarameters = True

    def training_step(self, batch, batch_idx):
        self.log_n_parameters()
        if self.train_scores is not None:
            for metric in self.train_scores:
                self.train_scores[metric].to(self.device)
        results = self.shared_step(batch)
        self.log('', results['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.train_scores is not None:
            for metric in self.train_scores:
                score = self.train_scores[metric](results['preds'].to(self.device), results['labels'].to(self.device))
                self.log(f'train_{metric}_step', score, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        wandb.log({'train_loss_step': results['loss'].cpu().item()})
        return results

    def training_epoch_end(self, outputs):
        if self.train_scores is not None:
            for metric in self.train_scores:
                train_score = self.train_scores[metric].compute()
                wandb.log({f'train_{metric}': train_score})
                self.log(f'train_{metric}', train_score, prog_bar=True, logger=True)
        wandb.log({'train_loss': np.mean([output['loss'].cpu().item() for output in outputs])})

    def validation_step(self, batch, batch_idx):
        if self.val_scores is not None:
            for metric in self.val_scores:
                self.val_scores[metric].to(self.device)
        results = self.shared_step(batch)
        self.log('val_loss', results['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.val_scores is not None:
            for metric in self.val_scores:
                self.val_scores[metric](results['preds'].to(self.device), results['labels'].to(self.device))
        return results

    def validation_epoch_end(self, outputs):
        if self.val_scores is not None:
            for metric in self.val_scores:
                val_score = self.val_scores[metric].compute()
                wandb.log({f'val_{metric}': val_score})
                self.log(f'val_{metric}', val_score, prog_bar=True, logger=True)
        val_loss = np.mean([output['loss'].cpu().item() for output in outputs])
        wandb.log({'val_loss': val_loss})
        if self.best_epochs['val_loss'] is None or (val_loss < self.best_epochs['val_loss'][1]):
            self.best_epochs['val_loss'] = (self.current_epoch, val_loss)
            wandb.run.summary['best_val_loss'] = val_loss
            wandb.run.summary['best_val_loss_epoch'] = self.current_epoch
            if self.val_scores is not None:
                for metric in self.val_scores:
                    val_score = self.val_scores[metric].compute()
                    wandb.run.summary[f'best_val_{metric}'] = val_score
        if self.log_confusion_matrix:
            preds = torch.cat([output['preds'].cpu() for output in outputs])
            labs = torch.cat([output['labels'].cpu() for output in outputs])
            wandb.log({'conf_mat': wandb.plot.confusion_matrix(
                probs=None,
                y_true=labs.long().cpu().numpy(),
                preds=preds.long().cpu().numpy(),
                class_names=range(max(preds.max().item(), labs.max().item())+1),
            )})

    def test_step(self, batch, batch_idx):
        self.log_n_parameters()
        if self.test_scores is not None:
            for metric in self.test_scores:
                self.test_scores[metric].to(self.device)
        results = self.shared_step(batch)
        self.log('test_loss', results['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.test_scores is not None:
            for metric in self.test_scores:
                self.test_scores[metric](results['preds'].to(self.device), results['labels'].to(self.device))
        return results

    def test_epoch_end(self, outputs):
        if self.test_scores is not None:
            for metric in self.test_scores:
                test_score = self.test_scores[metric].compute()
                wandb.log({f'test_{metric}': test_score})
                self.log(f'test_{metric}', test_score, prog_bar=True, logger=True)
        if self.log_confusion_matrix:
            preds = torch.cat([output['preds'].cpu() for output in outputs])
            labs = torch.cat([output['labels'].cpu() for output in outputs])
            wandb.log({'conf_mat': wandb.plot.confusion_matrix(
                probs=None,
                y_true=labs.long().cpu().numpy(),
                preds=preds.long().cpu().numpy(),
                class_names=range(max(preds.max().item(), labs.max().item())+1),
            )})

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), **optimizer_cfg)
        return optimizer