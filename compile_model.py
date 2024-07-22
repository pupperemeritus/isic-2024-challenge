from safetensors.torch import save_file

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torchmetrics import Metric

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

pl.seed_everything(42, workers=True)
# Import your model architecture
# from your_model_file import YourModelClass
"""
2024 ISIC Challenge primary prize scoring metric

Given a list of binary labels, an associated list of prediction 
scores ranging from [0,1], this function produces, as a single value, 
the partial area under the receiver operating characteristic (pAUC) 
above a given true positive rate (TPR).
https://en.wikipedia.org/wiki/Partial_Area_Under_the_ROC_Curve.

(c) 2024 Nicholas R Kurtansky, MSKCC
"""


import numpy as np
import pandas as pd


class PartialAUROC(Metric):
    def __init__(
        self,
        min_tpr: float = 0.80,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.min_tpr = min_tpr
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        preds = torch.cat(self.preds)
        target = torch.cat(self.target)
        return self._partial_auroc(target, preds, self.min_tpr)

    def _partial_auroc(
        self, y_true: torch.Tensor, y_score: torch.Tensor, min_tpr: float
    ) -> float:
        y_true = torch.abs(y_true - 1)
        y_score = -y_score

        fpr, tpr, _ = self._roc_curve(y_true, y_score)
        max_fpr = 1.0 - min_tpr

        # print(f"Computed FPR: {fpr}")
        # print(f"Computed TPR: {tpr}")

        if max_fpr == 1:
            return self._auc(fpr, tpr)
        if max_fpr <= 0 or max_fpr > 1:
            raise ValueError(f"Expected min_tpr in range [0, 1), got: {min_tpr}")

        stop = torch.searchsorted(fpr, torch.tensor(max_fpr), right=True)
        x_interp = fpr[stop - 1 : stop + 1]
        y_interp = tpr[stop - 1 : stop + 1]

        # print(f"x_interp: {x_interp}")
        # print(f"y_interp: {y_interp}")

        if len(x_interp) == 1:
            interp_tpr = y_interp[0]
        else:
            interp_tpr = y_interp[0] + (max_fpr - x_interp[0]) * (
                y_interp[1] - y_interp[0]
            ) / (x_interp[1] - x_interp[0])

        tpr = torch.cat([tpr[:stop], torch.tensor([interp_tpr])])
        fpr = torch.cat([fpr[:stop], torch.tensor([max_fpr])])

        partial_auc = self._auc(fpr, tpr)
        return partial_auc

    def _roc_curve(self, y_true: torch.Tensor, y_score: torch.Tensor):
        desc_score_indices = torch.argsort(y_score, descending=True)
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]

        distinct_value_indices = torch.where(torch.diff(y_score))[0]
        threshold_idxs = torch.cat(
            [distinct_value_indices, torch.tensor([y_true.numel() - 1])]
        )

        tps = torch.cumsum(y_true, dim=0)[threshold_idxs]
        fps = 1 + threshold_idxs - tps

        # Handle the case where there are no positive samples
        if tps[-1] == 0:
            tpr = torch.zeros_like(tps)
        else:
            tpr = tps / tps[-1]

        fpr = fps / fps[-1]
        thresholds = y_score[threshold_idxs]

        # print(f"tps: {tps}")
        # print(f"fps: {fps}")
        # print(f"tpr: {tpr}")
        # print(f"fpr: {fpr}")
        # print(f"thresholds: {thresholds}")

        return fpr, tpr, thresholds

    def _auc(self, x: torch.Tensor, y: torch.Tensor) -> float:
        if torch.all(y == 0):
            print("Warning: All TPR values are zero. AUC is undefined.")
            return 0.0

        direction = 1
        dx = torch.diff(x)
        if torch.any(dx < 0):
            if torch.all(dx <= 0):
                direction = -1
            else:
                raise ValueError("x is neither increasing nor decreasing")
        auc_value = direction * torch.trapz(y, x).item()
        # print(f"Computed AUC: {auc_value}")
        return auc_value


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNActivation(in_channels, hidden_dim, kernel_size=1))
        layers.extend(
            [
                ConvBNActivation(
                    hidden_dim, hidden_dim, stride=stride, groups=hidden_dim
                ),
                nn.Conv2d(hidden_dim, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBNActivation(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Mish(),
        )


class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate, dropout_rate=0.2):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList(
            [
                DenseLayer(in_channels + i * growth_rate, growth_rate, dropout_rate)
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = torch.cat([x, layer(x)], 1)
        return x


class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, dropout_rate):
        super(DenseLayer, self).__init__(
            nn.BatchNorm2d(in_channels),
            nn.Mish(),
            nn.Conv2d(in_channels, 4 * growth_rate, 1, bias=True),
            nn.BatchNorm2d(4 * growth_rate),
            nn.Mish(),
            nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1, bias=True),
            nn.Dropout2d(dropout_rate),
        )


class TransitionLayer(nn.Sequential):
    def __init__(self, in_channels, compression_factor=0.5):
        out_channels = int(in_channels * compression_factor)
        super(TransitionLayer, self).__init__(
            nn.BatchNorm2d(in_channels),
            nn.Mish(),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.AvgPool2d(2, stride=2),
        )


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, 1, 1)
        self.bn3 = nn.BatchNorm2d(1)

    def forward(self, x):
        g = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        att = nn.Hardswish()(g + x)
        att = nn.Sigmoid()(self.bn3(self.conv3(att)))
        return x * att


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(InceptionBlock, self).__init__()
        f1, f2, f3 = filters
        self.branch1 = ConvBNActivation(in_channels, f1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBNActivation(in_channels, f2[0], kernel_size=1),
            ConvBNActivation(f2[0], f2[1], kernel_size=3),
        )
        self.branch3 = nn.Sequential(
            ConvBNActivation(in_channels, f3[0], kernel_size=1),
            ConvBNActivation(f3[0], f3[1], kernel_size=5),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNActivation(in_channels, f1, kernel_size=1),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class GatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides):
        super(GatedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=strides,
            padding=kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.activation = nn.Mish()

        # Add a shortcut connection if input and output dimensions don't match
        self.shortcut = nn.Sequential()
        if strides != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=strides, bias=True
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        gate = nn.Sigmoid()(self.bn3(self.conv3(x)))
        x = x * gate
        x += residual
        return self.activation(x)


class GuruNet(pl.LightningModule):
    def __init__(
        self,
        input_shape=(139, 139, 3),
        metadata_shape=None,
        classes=2,
    ):
        super(GuruNet, self).__init__()
        self.input_shape = input_shape
        self.metadata_shape = metadata_shape

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 256, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.activation = nn.Hardswish()

        # Inverted Residual Blocks
        self.inv_res_blocks = nn.ModuleList()
        block_params = [
            # expand_ratio, filters, strides, repeats
            (6, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 40, 2, 2),
            (6, 80, 2, 3),
            (6, 112, 1, 3),
            (6, 128, 2, 4),
            (6, 196, 1, 1),
        ]

        in_channels = 256
        for i, (expand_ratio, filters, strides, repeats) in enumerate(block_params):
            for j in range(repeats):
                if j > 0:
                    strides = 1
                self.inv_res_blocks.append(
                    InvertedResidualBlock(in_channels, filters, expand_ratio, strides)
                )
                in_channels = filters

        # Dense Block
        self.dense_block = DenseBlock(in_channels, num_layers=20, growth_rate=32)
        in_channels += 20 * 32  # Update in_channels after dense block

        # Transition Layer
        self.transition = TransitionLayer(in_channels, compression_factor=0.5)
        in_channels = int(in_channels * 0.5)

        # Attention Block
        self.attention = AttentionBlock(in_channels, 256)
        in_channels = 256

        # Average Pooling
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # Inception Block
        self.inception = InceptionBlock(in_channels, [128, (128, 192), (32, 96)])
        in_channels = 128 + 192 + 96 + 128

        self.inception2 = InceptionBlock(in_channels, [128, (128, 192), (32, 96)])
        in_channels = 128 + 192 + 96 + 128

        # Attention Block
        self.attention2 = AttentionBlock(in_channels, 256)
        in_channels = 256

        # Gated Residual Block
        self.gated_res = GatedResidualBlock(in_channels, 512, kernel_size=3, strides=2)
        in_channels = 512

        # Attention Block
        self.attention3 = AttentionBlock(in_channels, 256)
        in_channels = 256

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(in_channels, 4096)
        self.bn_fc1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.bn_fc2 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 512)
        self.bn_fc3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 128)
        self.bn_fc4 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)

        self.metadata_fc1 = nn.Linear(41, 4096)
        self.metadata_bn1 = nn.BatchNorm1d(4096)
        self.metadata_fc2 = nn.Linear(4096, 1024)
        self.metadata_bn2 = nn.BatchNorm1d(1024)
        self.metadata_fc3 = nn.Linear(1024, 512)
        self.metadata_bn3 = nn.BatchNorm1d(512)
        self.metadata_fc4 = nn.Linear(512, 128)
        self.metadata_bn4 = nn.BatchNorm1d(128)
        self.final_fc = nn.Linear(128 + 128, classes)
        self.final_activation = nn.Sigmoid()
        self.scaler = GradScaler()
        self.loss = self.loss = nn.CrossEntropyLoss()
        self.auroc = PartialAUROC(min_tpr=0.8)

    def forward(self, x, metadata):
        x = self.activation(self.bn1(self.conv1(x)))

        # Inverted Residual Blocks
        for block in self.inv_res_blocks:
            x = block(x)

        # Dense Block
        x = self.dense_block(x)

        # Transition Layer
        x = self.transition(x)

        # Attention Block
        x = self.attention(x)

        # Average Pooling
        x = self.avg_pool(x)

        # Inception Block
        x = self.inception(x)
        x = self.inception2(x)

        # Attention Block
        x = self.attention2(x)

        # Gated Residual Block
        x = self.gated_res(x)

        # Attention Block
        x = self.attention3(x)

        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.activation(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.activation(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.activation(self.bn_fc3(self.fc3(x)))
        x = self.dropout(x)
        x = self.activation(self.bn_fc4(self.fc4(x)))

        metadata = self.activation(self.metadata_bn1(self.metadata_fc1(metadata)))
        metadata = self.dropout(metadata)
        metadata = self.activation(self.metadata_bn2(self.metadata_fc2(metadata)))
        metadata = self.dropout(metadata)
        metadata = self.activation(self.metadata_bn3(self.metadata_fc3(metadata)))
        metadata = self.dropout(metadata)
        metadata = self.activation(self.metadata_bn4(self.metadata_fc4(metadata)))

        x = torch.cat([x, metadata], dim=1)

        x = self.final_fc(x)
        # Apply sigmoid to ensure output is between 0 and 1
        x = self.final_activation(x)

        return x

    def training_step(self, batch, batch_idx):
        (images, metadata), targets = batch
        outputs = self(images, metadata)
        loss = self.loss(outputs, targets)  # targets is already one-hot encoded
        # Get the probability of the positive class
        pos_probs = outputs[:, 1].float().cpu()

        # Convert one-hot encoded targets to binary labels
        targets_binary = targets[:, 1].int().cpu()
        rocauc = self.auroc(pos_probs, targets_binary)  # Use class 1 probability

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_pAUC",
            rocauc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        (images, metadata), targets = batch
        outputs = self(images, metadata)
        loss = self.loss(outputs, targets)  # targets is already one-hot encoded
        # Get the probability of the positive class
        pos_probs = outputs[:, 1].float().cpu()

        # Convert one-hot encoded targets to binary labels
        targets_binary = targets[:, 1].int().cpu()
        rocauc = self.auroc(pos_probs, targets_binary)

        # Use class 1 probability

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_pAUC",
            rocauc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    # def test_step(self, batch, batch_idx):
    #     (images, metadata), targets = batch
    #     outputs = self(images, metadata)
    #     loss = self.loss(outputs, targets)  # targets is already one-hot encoded

    #     # Get the probability of the positive class
    #     pos_probs = outputs[:, 1].float().cpu()

    #     # Convert one-hot encoded targets to binary labels
    #     targets_binary = targets[:, 1].int().cpu()
    #     rocauc = self.auroc(pos_probs, targets_binary)

    #     self.log(
    #         "test_loss",
    #         loss,
    #         on_step=True,
    #         on_epoch=True,
    #         prog_bar=True,
    #     )
    #     self.log(
    #         "test_pAUC",
    #         rocauc,
    #         on_step=True,
    #         on_epoch=True,
    #         prog_bar=True,
    #     )

    #     return loss

    def configure_optimizers(self):
        optimizer = optim.NAdam(
            self.parameters(), lr=0.001, momentum_decay=0.5, weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=2, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            },
        }


def convert_checkpoint(ckpt_path, model_class, output_path):
    # Initialize your model
    model = model_class()

    # Load the checkpoint
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Determine the output format
    if output_path.endswith('.pth'):
        # Save as .pth
        torch.save(model.state_dict(), output_path)
        print(f"Model saved as .pth to {output_path}")
    elif output_path.endswith('.safetensors'):
        # Save as .safetensors
        state_dict = model.state_dict()
        save_file(state_dict, output_path)
        print(f"Model saved as .safetensors to {output_path}")
    else:
        print("Unsupported output format. Use .pth or .safetensors")


if __name__ == "__main__":
    ckpt_path = "/home/pupperemeritus/DL/isic-2024-challenge/checkpoints/version_98/gurunet-epoch=62-val_loss=0.33.ckpt"
    output_path = "/home/pupperemeritus/DL/isic-2024-challenge/model.safetensors"  # or "model.pth"
    convert_checkpoint(ckpt_path, GuruNet, output_path)
