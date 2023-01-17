from argparse import Namespace

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.alexnet import AlexNet
from models.improved_lenet import ImprovedLeNet
from models.lenet import LeNet


class Classifier(pl.LightningModule):
    def __init__(self, args: Namespace, classes):
        super().__init__()
        self.args = args
        self.labels = np.linspace(0, args.num_classes - 1, args.num_classes, dtype=int)
        self.classes = classes

        if args.model_name == 'lenet':
            self.model = LeNet(len(classes))
        elif args.model_name == 'improved_lenet':
            self.model = ImprovedLeNet(len(classes))
        elif args.model_name == 'alexnet':
            self.model = AlexNet(len(classes))
        else:
            raise ValueError(f'Unknown model name: {args.model_name}')

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, report = self._shared_eval_step(batch, batch_idx)
        metrics = {"validation_accuracy": report['micro avg']['f1-score'],
                   "validation_weighted_f1": report['weighted avg']['f1-score'], "validation_loss": loss}
        self.log_dict(metrics)
        return {"loss": loss, "report": report}

    def test_step(self, batch, batch_idx):
        loss, report = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_accuracy": report['micro avg']['f1-score'],
                   "test_weighted_f1": report['weighted avg']['f1-score'], "test_loss": loss}
        self.log_dict(metrics)
        return {"loss": loss, "report": report}

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        report = self.__metrics(y_hat, y)
        return loss, report

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True),
                "monitor": "validation_loss",
                "frequency": 1
            },
        }

    def __metrics(self, y_hat, y):
        y = torch.Tensor.cpu(y)
        y_hat = torch.Tensor.cpu(y_hat)
        y_hat = torch.argmax(y_hat, dim=1)
        report = classification_report(y, y_hat, output_dict=True, zero_division=0, labels=self.labels,
                                       target_names=self.classes)
        return report


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.LazyLinear):
        nn.init.kaiming_normal_(m.weight)
