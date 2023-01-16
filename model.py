from argparse import Namespace

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import accuracy


class LeNet(pl.LightningModule):
    def __init__(self, args: Namespace):
        super().__init__()
        self.args = args
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # (32, 159, 99)
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),  # (32, 79, 49)
            nn.Conv2d(32, 64, kernel_size=3),  # (64, 77, 47)
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),  # (64, 38, 23)
            nn.Flatten(),
            nn.Linear(55936, 1000),
            nn.Dropout(0.5),
            nn.Linear(1000, args.num_classes),
            nn.ReLU(),
            nn.LogSoftmax(dim=1)
            )

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self._shared_step(x, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch)
        metrics = {"validation_acc": acc, "validation_loss": loss}
        self.log_dict(metrics)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return loss

    def _shared_step(self, x, y):
        y_hat = self.model(x)
        loss = nn.functional.nll_loss(y_hat, y)
        return y_hat, loss

    def _shared_eval_step(self, batch):
        x, y = batch
        y_hat, loss = self._shared_step(x, y)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.args.num_classes)
        return loss, acc

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


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
