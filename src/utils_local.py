import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from monai import transforms as monai_t

import tifffile as tiff
import os
import pandas as pd
import numpy as np

from typing import Union

class Mozzarella3D(Dataset):
    """
    Custom dataset class for volumetric MozzaVID data.
    Arguments:
        X: pd.DataFrame
            pandas DataFrame with the file paths.
        y: pd.Series
            pandas Series with the labels.
        data_aug: bool
            Whether to apply data augmentation transformations to the data a (e.g for training).
        rotate: bool
            Whether to apply rotation to the data (used in rotation ablation study).
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, data_aug: bool = False, rotate: bool = False):

        self.images_files = np.array(X["file"])
        self.labels = np.array(y)

        if data_aug:
            if rotate:
                self.transform = monai_t.Compose(
                    [
                        monai_t.NormalizeIntensity(
                            subtrahend=0.6, divisor=0.2, dtype=np.float32
                        ),
                        monai_t.RandRotate90(spatial_axes=(1, 2), prob=0.5),
                        monai_t.RandFlip(spatial_axis=1, prob=0.5),
                        monai_t.RandFlip(spatial_axis=2, prob=0.5),
                        monai_t.RandRotate(
                            range_x=np.pi / 6,
                            prob=0.5,
                            padding_mode="zeros",
                            dtype=np.float32,
                        ),
                        monai_t.ToTensor(dtype=torch.float16),
                    ]
                )
            else:
                self.transform = monai_t.Compose(
                    [
                        monai_t.NormalizeIntensity(
                            subtrahend=0.6, divisor=0.2, dtype=np.float16
                        ),
                        monai_t.RandFlip(spatial_axis=1, prob=0.5),
                        monai_t.RandFlip(spatial_axis=2, prob=0.5),
                        monai_t.ToTensor(dtype=torch.float16),
                    ]
                )
        else:
            self.transform = monai_t.Compose(
                [
                    monai_t.NormalizeIntensity(
                        subtrahend=0.6, divisor=0.2, dtype=np.float16
                    ),
                    monai_t.ToTensor(dtype=torch.float16),
                ]
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img_path = self.images_files[idx]

        # Load the 3D .tiff image (volumetric data)
        image = tiff.imread(img_path)
        # Add channel dimension
        image = image[None,]

        # Apply transformations
        image = self.transform(image)

        # Convert label to torch tensor
        label = torch.tensor(label)

        return image, label


class Mozzarella2D(Dataset):
    """
    Custom dataset class for 2D MozzaVID data.
        Arguments:
        X: pd.DataFrame
            pandas DataFrame with the file paths.
        y: pd.Series
            pandas Series with the labels.
        data_aug: bool
            Whether to apply data augmentation transformations to the data a (e.g for training).
        rotate: bool
            Whether to apply rotation to the data (used in rotation ablation study).
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, data_aug: bool = False, rotate: bool = False):
        
        self.images_files = np.array(X["file"])
        self.labels = np.array(y)
        self.rotate = rotate

        if data_aug:
            if rotate:
                self.transform = monai_t.Compose(
                    [
                        monai_t.NormalizeIntensity(
                            subtrahend=0.6, divisor=0.2, dtype=np.float32
                        ),
                        monai_t.RandRotate90(spatial_axes=(0, 1), prob=0.5),
                        monai_t.RandFlip(spatial_axis=0, prob=0.5),
                        monai_t.RandFlip(spatial_axis=1, prob=0.5),
                        monai_t.RandRotate(
                            range_x=np.pi / 6,
                            prob=0.5,
                            padding_mode="zeros",
                            dtype=np.float32,
                        ),
                        monai_t.ToTensor(dtype=torch.float16),
                    ]
                )
            else:
                self.transform = monai_t.Compose(
                    [
                        monai_t.NormalizeIntensity(
                            subtrahend=0.6, divisor=0.2, dtype=np.float16
                        ),
                        monai_t.RandFlip(spatial_axis=0, prob=0.5),
                        monai_t.RandFlip(spatial_axis=1, prob=0.5),
                        monai_t.ToTensor(dtype=torch.float16),
                    ]
                )
        else:
            self.transform = monai_t.Compose(
                [
                    monai_t.NormalizeIntensity(
                        subtrahend=0.6, divisor=0.2, dtype=np.float16
                    ),
                    monai_t.ToTensor(dtype=torch.float16),
                ]
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img_path = self.images_files[idx]

        with tiff.TiffFile(img_path) as tif:
            image = tif.asarray(key=[48, 96, 144])  # Load three evenly spaced slices

        image = self.transform(image)
        label = torch.tensor(label)

        return image, label


def get_splits(dataset_path: str, granularity: str = "coarse"):
    """
    Loads train, validation and test sets from 'dataset.csv'
    Arguments:
        dataset_path: str
            Path to the dataset directory.
        granularity: str
            Granularity of the data either "coarse" or "fine".
    Returns:
        X_train: pd.DataFrame
            DataFrame with the train set file paths.
        X_val: pd.DataFrame
            DataFrame with the validation set file paths.
        y_train: pd.Series
            Series with the train set labels.
        y_val: pd.Series
            Series with the validation set labels.
    """

    if granularity == "coarse":
        label = "label_cheese"
    elif granularity == "fine":
        label = "label_sample"
    else:
        raise ValueError("Granularity must be 'coarse' or 'fine'")

    # Load the dataset
    train_csv_path = os.path.join(dataset_path, "train.csv")
    val_csv_path = os.path.join(dataset_path, "val.csv")
    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)

    X_train = df_train[["file"]]
    y_train = df_train[label]

    X_val = df_val[["file"]]
    y_val = df_val[label]

    # Append the dataset path to the file paths
    X_train.loc[:, "file"] = dataset_path + X_train["file"]
    X_val.loc[:, "file"] = dataset_path + X_val["file"]

    return (
        X_train,
        X_val,
        y_train,
        y_val,
    )


def get_data_loaders(
    dataset_func: Union[Mozzarella2D, Mozzarella3D],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    batch_size: int = 2,
    num_workers: int = 2,
    rotate: bool = False,
):
    """
    Creates pytorch DataLoaders for the train and validation sets.
    Arguments:
        dataset_func: Union[Mozzarella2D, Mozzarella3D]
            Dataset class for the data.
        X_train: pd.DataFrame
            DataFrame with the train set file paths.
        X_val: pd.DataFrame
            DataFrame with the validation set file paths.
        y_train: pd.Series
            Series with the train set labels.
        y_val: pd.Series
            Series with the validation set labels.
        batch_size: int
            Batch size for the DataLoader.
        num_workers: int
            Number of workers for the DataLoader.
        rotate: bool
            Whether to apply rotation to the data (used in rotation ablation study).
    Returns:
        train_loader: torch.utils.data.DataLoader
            DataLoader for the training set.
        val_loader: torch.utils.data.DataLoader
            DataLoader for the validation set.
    """

    train_dataset = dataset_func(X_train, y_train, data_aug=True, rotate=rotate)
    val_dataset = dataset_func(X_val, y_val, data_aug=False, rotate=rotate)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader

