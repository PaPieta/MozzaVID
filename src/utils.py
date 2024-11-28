import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from monai import transforms as monai_t

import tifffile as tiff
import os
import pandas as pd
import numpy as np


class Mozzarella3D(Dataset):
    """
    Custom dataset class for volumetric MozzaVID data.
    """

    def __init__(self, X, y, transform=False, rotate=False):
        """
        Initialize the dataset.
        Parameters:
            X: pandas DataFrame with the file paths.
            y: pandas Series with the labels.
            transform: bool, whether to apply transformations to the data.
            rotate: bool, whether to apply rotation to the data.
        """
        self.images_files = np.array(X["file"])
        self.labels = np.array(y)

        if transform:
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
    """

    def __init__(self, X, y, transform=False, rotate=False):
        """
        Initialize the dataset.
        Parameters:
            X: pandas DataFrame with the file paths.
            y: pandas Series with the labels.
            transform: bool, whether to apply transformations to the data.
            rotate: bool, whether to apply rotation to the data.
        """
        self.images_files = np.array(X["file"])
        self.labels = np.array(y)
        self.rotate = rotate

        if transform:
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
                self.transform = v2.Compose(
                    [
                        v2.RandomVerticalFlip(p=0.5),
                        v2.RandomHorizontalFlip(p=0.5),
                        v2.Normalize(mean=[0.6], std=[0.2]),
                    ]
                )
        else:
            if rotate:
                self.transform = monai_t.Compose(
                    [
                        monai_t.NormalizeIntensity(
                            subtrahend=0.6, divisor=0.2, dtype=np.float32
                        ),
                        monai_t.ToTensor(dtype=torch.float16),
                    ]
                )
            else:
                self.transform = v2.Compose([v2.Normalize(mean=[0.6], std=[0.2])])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img_path = self.images_files[idx]

        with tiff.TiffFile(img_path) as tif:
            image = tif.asarray(key=[48, 96, 144])  # Load three evenly spaced slices

        if not self.rotate:
            image = tv_tensors.Image(torch.tensor(image))

        image = self.transform(image)
        label = torch.tensor(label)

        return image, label


def get_splits(dataset_path, granularity="coarse"):
    """
    Loads train, validation and test sets from 'dataset.csv'
    Parameters:
        dataset_path (str): Path to the dataset directory
        granularity (str): Granularity of the labels. Must be 'coarse' or 'fine'
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
    dataset_func,
    X_train,
    X_val,
    y_train,
    y_val,
    batch_size=2,
    num_workers=16,
    rotate=False,
):
    """
    Creates pytorch DataLoaders for the train, validation and test sets.
    Parameters:
        dataset_func: Custom dataset class.
        X_train: pandas DataFrame with the train set file paths.
        X_val: pandas DataFrame with the validation set file paths.
        y_train: pandas Series with the train set labels.
        y_val: pandas Series with the validation set labels.
        batch_size: int, batch size.
        num_workers: int, number of workers for the DataLoader.
        rotate: bool, whether to apply rotation to the data.
    """

    train_dataset = dataset_func(X_train, y_train, transform=True, rotate=rotate)
    val_dataset = dataset_func(X_val, y_val, transform=False, rotate=rotate)
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
