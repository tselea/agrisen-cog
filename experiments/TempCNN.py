#code adapted from https://github.com/Orion-AI-Lab/S4A-Models


from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchmetrics import MetricCollection, Accuracy, Precision, Recall,ConfusionMatrix
from torchmetrics import F1Score as F1

import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import os

import torch
import torch.nn as nn
import torch.utils.data

"""
Pytorch re-implementation of Pelletier et al. 2019
https://github.com/charlotte-pel/temporalCNN
https://www.mdpi.com/2072-4292/11/5/523
"""


class Conv1D_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=5, drop_probability=0.5):
        super(Conv1D_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims, kernel_size, padding=(kernel_size // 2)),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class FC_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, drop_probability=0.5):
        super(FC_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class TempCNN(pl.LightningModule):
    def __init__(
            self,
            run_path,
            input_size: int = 13,
            num_classes: int = 9,
            sequencelength: int = 45,
            kernel_size: int = 7,
            hidden_size: int = 128,
            dropout: float = 0.2,
            batch_size: int = 4098,
            name_decoder: dict = None,
            id_decoder: dict = None,
            lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_size = 1
        self.num_classes = num_classes
        self.sequencelength = 28
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.batch_size = batch_size
        self.name_decoder = name_decoder
        self.id_decoder = id_decoder
        self.lr = lr

        self.modelname = f"TempCNN_input-dim={self.input_size}_" \
                         f"num-classes={self.num_classes}_" \
                         f"sequencelenght={self.sequencelength}_" \
                         f"kernelsize={self.kernel_size}_" \
                         f"hidden-dims={self.hidden_size}_" \
                         f"dropout={self.dropout}"

        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(
            self.input_size,
            self.hidden_size,
            kernel_size=self.kernel_size,
            drop_probability=self.dropout
        )

        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(
            self.hidden_size,
            self.hidden_size,
            kernel_size=self.kernel_size,
            drop_probability=self.dropout
        )

        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(
            self.hidden_size,
            self.hidden_size,
            kernel_size=self.kernel_size,
            drop_probability=self.dropout
        )

        self.flatten = Flatten()
        self.dense = FC_BatchNorm_Relu_Dropout(
            self.hidden_size * self.sequencelength,
            4 * self.hidden_size,
            drop_probability=self.dropout
        )

        self.logsoftmax = nn.Sequential(
            nn.Linear(4 * self.hidden_size, self.num_classes),
            nn.LogSoftmax(dim=-1)
        )

        # Each cost function is modeled through categorical cross entropy
        self.criterion = nn.CrossEntropyLoss()

        metrics = MetricCollection({
            'accuracy': Accuracy(task='multiclass', num_classes=self.num_classes),
            'precision': Precision(task='multiclass',num_classes=self.num_classes),
            'precision_weighted': Precision(task='multiclass',num_classes=self.num_classes, average='weighted'),
            'recall': Recall(task='multiclass',num_classes=self.num_classes),
            'recall_weighted': Recall(task='multiclass',num_classes=self.num_classes, average='weighted'),
            'f1-score:': F1(task='multiclass',num_classes=self.num_classes),
            'f1-score_weighted:': F1(task='multiclass',num_classes=self.num_classes, average='weighted')
        })

        self.run_path = Path(run_path)

        # Only accuracy for training
        self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_confusion_matrix = ConfusionMatrix(task='multiclass',num_classes=self.num_classes, compute_on_step=False, normalize='true')
        self.test_confusion_matrix = ConfusionMatrix(task='multiclass',num_classes=self.num_classes, compute_on_step=False, normalize='true')

        self.test_confusion_matrix_org = ConfusionMatrix(task='multiclass', num_classes=num_classes, compute_on_step=False, normalize='none')


        self.metrics_val = metrics.clone(prefix='val_')
        self.metrics_test = metrics.clone(prefix='test_')

        self.test_f1_score_class = F1(task='multiclass', num_classes=self.num_classes, average='none')


    def forward(self, x):
        # require NxTxD
        x = x.transpose(1, 2)
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.logsoftmax(x)

    def training_step(self, batch, batch_idx):
        data, target = batch

        # Forward
        preds = self(data)

        # Calculate loss
        loss = self.criterion(preds, target)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc(preds, target), on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch

        # Forward
        preds = self(data)

        # Calculate loss
        loss = self.criterion(preds, target)
        self.log('val_loss', loss)

        self.val_confusion_matrix(preds, target)

        self.metrics_val(preds, target)
        self.log_dict(self.metrics_val, on_step=False, on_epoch=True)

        return {'loss': loss, 'preds': preds, 'target': target}

    def test_step(self, batch, batch_idx):
        data, target = batch

        # Forward
        preds = self(data)

        # Calculate loss
        loss = self.criterion(preds, target)
        self.log('test_loss', loss)

        self.test_confusion_matrix(preds, target)
        self.test_confusion_matrix_org(preds, target)


        self.metrics_test(preds, target)
        self.log_dict(self.metrics_test, on_step=False, on_epoch=True)

        return {'loss': loss, 'preds': preds, 'target': target}

    def training_step_end(self, outputs) -> None:
        return None

    def validation_step_end(self, outputs) -> None:
        return None

    def test_step_end(self, outputs) -> None:
        return None

    def training_epoch_end(self, outputs):
        return None

    def validation_epoch_end(self, outputs):
        sns.set(font_scale=1.25)
        confusion_matrix = self.val_confusion_matrix.compute().cpu().numpy()

        indices = [f'{self.name_decoder[str(i)]} ({self.id_decoder[str(i)]})' for i in range(self.num_classes)]

        df_cm = pd.DataFrame(confusion_matrix, index=indices, columns=indices)

        plt.figure(figsize=(15, 15))
        fig = sns.heatmap(df_cm, annot=False, cmap='rocket_r', fmt='.3').get_figure()
        plt.close(fig)

        self.logger.experiment.add_figure('val_confusion_matrix', fig, self.current_epoch)
        self.val_confusion_matrix.reset()

    def test_epoch_end(self, outputs):
        sns.set(font_scale=1.25)
        confusion_matrix = self.test_confusion_matrix.compute().cpu().numpy()
        confusion_matrix_org = self.test_confusion_matrix_org.compute().cpu().numpy()


        indices = [f'{self.name_decoder[str(i)]} ({self.id_decoder[str(i)]})' for i in range(self.num_classes)]

        df_cm = pd.DataFrame(confusion_matrix, index=indices, columns=indices)
        plt.figure(figsize=(15, 15))
        #fig = sns.heatmap(df_cm, annot=False, cmap='rocket_r', fmt='.3').get_figure()
        fig = sns.heatmap(df_cm, annot=False, cmap='terrain_r', fmt='.3').get_figure()
        plt.savefig(self.run_path / f'confusion_matrix_epoch_norm.png', dpi=fig.dpi, bbox_inches='tight',
                    pad_inches=0.5)

        plt.close(fig)

        # Export metrics in text file
        metrics_file = self.run_path / f"confusion_matrix_norm.csv"
        # Delete file if present
        metrics_file.unlink(missing_ok=True)
        df_cm.to_csv(metrics_file)

        # Replace invalid values with 0
        confusion_matrix_org = np.nan_to_num(confusion_matrix_org, nan=0.0, posinf=0.0, neginf=0.0)

        indices = [f'{self.name_decoder[str(i)]} ({self.id_decoder[str(i)]})' for i in range(self.num_classes)]

        df_cm_org = pd.DataFrame(confusion_matrix_org, index=indices, columns=indices)
        # m = self.metrics_test.compute()
        # print(m.keys())
        f1_class_score = self.test_f1_score_class.compute().cpu().numpy()

        # Export metrics in text file
        metrics_file = self.run_path / f"confusion_matrix.csv"
        # Delete file if present
        metrics_file.unlink(missing_ok=True)
        df_cm_org.to_csv(metrics_file)
        with open(self.run_path / "f1_scores_class.npy", 'wb') as f:
            np.save(f, f1_class_score)

        metrics = self.metrics_test.compute()
        for k, v in metrics.items():
            metrics[k] = float(metrics[k].cpu().numpy())
        import pickle

        with open(self.run_path / 'metrics_dictionary.pkl', 'wb') as f:
            pickle.dump(metrics, f)

        self.logger.experiment.add_figure('test_confusion_matrix', fig, self.current_epoch)
        self.test_confusion_matrix.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.batch_size * 100 * 5, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to " + path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state, **kwargs), path)

    def load(self, path):
        print("loading model from " + path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot
