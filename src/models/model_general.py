import torch
import torch.nn as nn
from torchvision import models as torchmodels
import lightning as L

from models.sota import (
    resnet3d,
    convnext2d,
    convnext3d,
    mobilenetv2_3d,
    vit2d,
    vit3d,
    swin2d,
    swin3d,
)

ARCHI_DICT_2D = {
    "ResNet": torchmodels.resnet50,
    "ConvNeXt": convnext2d.convnext_small,
    "MobileNetV2": torchmodels.mobilenet_v2,
    "ViT": vit2d.vit_b_16,
    "Swin": swin2d.SwinTransformer,
}

ARCHI_DICT_3D = {
    "ResNet": resnet3d.resnet50,
    "ConvNeXt": convnext3d.convnext_small,
    "MobileNetV2": mobilenetv2_3d.MobileNetV2,
    "ViT": vit3d.vit_b_16,
    "Swin": swin3d.SwinTransformer,
}


class Net(L.LightningModule):
    """
    A wrapper model.
    """

    def __init__(self, model_hparams, lr=0.001):
        """
        Initialize the model.
        Parameters:
            model_hparams: a class with model hyperparameters.
                data_dim: "2D" or "3D".
                architecture: "ResNet", "ConvNeXt", "MobileNetV2", "ViT", "Swin".
                granularity: "coarse" or "fine".
            lr: learning rate.
        """
        super().__init__()

        self.lr = lr
        self.archtecture = model_hparams.architecture

        if model_hparams.data_dim == "3D":
            self.archi_dict = ARCHI_DICT_3D
        elif model_hparams.data_dim == "2D":
            self.archi_dict = ARCHI_DICT_2D
        else:
            raise ValueError("Data dimension must be '2D' or '3D'")

        if model_hparams.granularity == "coarse":
            num_classes = 25
        elif model_hparams.granularity == "fine":
            num_classes = 149
        else:
            raise ValueError("Granularity must be 'coarse' or 'fine'")

        self.net = []
        self.net.append(self.archi_dict[self.archtecture]())

        self.net.append(nn.ReLU())
        self.net.append(nn.Linear(1000, num_classes))
        self.net = nn.Sequential(*self.net)

    def forward(self, X):
        prediction = self.net(X)

        return prediction

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Split batch
        X, y = batch

        # Forward pass
        y_hat = self(X)

        loss = nn.CrossEntropyLoss()(y_hat, y)
        accuracy = torch.sum(torch.argmax(y_hat, 1) == y) / X.shape[0]
        self.log("train/loss", loss)
        self.log("train/accuracy", accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        # Split batch
        X, y = batch

        # Forward pass
        y_hat = self(X)

        loss = nn.CrossEntropyLoss()(y_hat, y)
        accuracy = torch.sum(torch.argmax(y_hat, 1) == y) / X.shape[0]
        self.log("val/loss", loss)
        self.log("val/accuracy", accuracy)

    def test_step(self, batch, batch_idx):
        # Split batch
        X, y = batch

        # Forward pass
        y_hat = self(X)

        loss = nn.CrossEntropyLoss()(y_hat, y)
        accuracy = torch.sum(torch.argmax(y_hat, 1) == y) / X.shape[0]
        self.log("test/loss", loss)
        self.log("test/accuracy", accuracy)

        return accuracy
