import torch
from monai import transforms as monai_t
import numpy as np

import webdataset as wds

SMALL_URL = "https://huggingface.co/datasets/dtudk/MozzaVID_Small/resolve/main/"
BASE_URL = "https://huggingface.co/datasets/dtudk/MozzaVID_Base/resolve/main/"
LARGE_URL = "https://huggingface.co/datasets/dtudk/MozzaVID_Large/resolve/main/"


def get_transform(data_dim: str, data_aug: bool = False, rotate: bool = False) -> monai_t.Compose:
    """
    Generate data transformation pipeline.
    Arguments:
        data_dim: str
            Dimensionality of the data either "2D" or "3D.
        data_aug: bool
            Whether to apply data augmentation transformations to the data (e.g for training).
        rotate: bool
            Whether to apply rotation to the data (used in rotation ablation study).
    Returns:    
        transform: monai_t.Compose
            Data transformation pipeline.
    """

    if data_dim == "3D":
        spatial_axes = (1, 2)
    elif data_dim == "2D":
        spatial_axes = (0, 1)
    else:
        raise ValueError("data_dim must be '2D' or '3D'")
    
    if data_aug:
        if rotate:
            transform = monai_t.Compose(
                [
                    monai_t.NormalizeIntensity(
                        subtrahend=0.6, divisor=0.2, dtype=np.float32
                    ),
                    monai_t.RandRotate90(spatial_axes=spatial_axes, prob=0.5),
                    monai_t.RandFlip(spatial_axis=spatial_axes[0], prob=0.5),
                    monai_t.RandFlip(spatial_axis=spatial_axes[1], prob=0.5),
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
            transform = monai_t.Compose(
                [
                    monai_t.NormalizeIntensity(
                        subtrahend=0.6, divisor=0.2, dtype=np.float16
                    ),
                    monai_t.RandFlip(spatial_axis=spatial_axes[0], prob=0.5),
                    monai_t.RandFlip(spatial_axis=spatial_axes[1], prob=0.5),
                    monai_t.ToTensor(dtype=torch.float16),
                ]
            )
    else:
        transform = monai_t.Compose(
            [
                monai_t.NormalizeIntensity(
                    subtrahend=0.6, divisor=0.2, dtype=np.float16
                ),
                monai_t.ToTensor(dtype=torch.float16),
            ]
        )
    
    return transform


class Decoder:
    """
    Custom decoder class for volumetric MozzaVID data. Necessary for parametrization before WebDataset setup.
    Arguments:
        transform: monai_t.Compose
            Data transformation pipeline.
        granularity: str
            Granularity of the data either "coarse" or "fine".
        data_dim: str
            Dimensionality of the data either "2D" or "3D.
    """

    def __init__(self, transform: monai_t.Compose, granularity: str = "coarse", data_dim: str = "2D", split: str = "train"):
    
        self.granularity = granularity
        self.data_dim = data_dim
        self.transform = transform
        self.split = split

    def check_split(self, sample: dict):
        """
        Checks if the sample within a shard belongs to the currently used split
        Arguments:
            sample: dict
                Sample from the webdataset.
        Returns:
            bool
        """
        if self.granularity == "coarse":
            return  sample["json"]['split_coarse'] == self.split
        elif self.granularity == "fine":
            return  sample["json"]['split_fine'] == self.split
        else:
            raise ValueError("Granularity must be 'coarse' or 'fine'")
        
    def decode_label(self, json: dict):
        """
        Decode the label from the json file, using the chosen granularity.
        Arguments:
            json: dict
                JSON file dict with the labels.
        Returns:
            label: torch.Tensor
                Label tensor.
        """
        if self.granularity == "coarse":
            label = torch.tensor(int(json['label_cheese']))
        elif self.granularity == "fine":
            label = torch.tensor(int(json['label_sample']))
        else:
            raise ValueError("Granularity must be 'coarse' or 'fine'")
                         
        return label
    
    def decode(self, sample: dict):
        """
        Decode a sample from the webdataset, depending on the data_dim.
        Arguments:
            sample: dict
                Sample from the webdataset.
        Returns:
            sample: dict
                Sample with the image and label decoded.
        """

        if self.data_dim == "3D":
            sample["image"] = sample["npy"][None,]
        elif self.data_dim == "2D":
            sample["image"] = sample["npy"][[48, 96, 144]]
        else:
            raise ValueError("data_dim must be '2D' or '3D'")
        
        sample["label"] = self.decode_label(sample["json"])
        
        return sample
    
    def apply_transform(self, sample: dict):
        """
        Apply the transform to the sample.
        Arguments:
            sample: dict
                Sample from the webdataset.
        Returns:
            sample: dict
                Sample with the image transformed.
        """
        sample["image"] = self.transform(sample["image"])

        return sample


def get_data_loaders(dataset_split: str, data_dim: str, granularity: str, rotate: bool = False, batch_size: int = 2, num_workers: int = 2):
    """
    Creates pytorch DataLoaders for the train and validation sets.
    Arguments:
        dataset_split: str
            Dataset split. Must be 'Small', 'Base' or 'Large'.
        data_dim: str
            Dimensionality of the data either "2D" or "3D.
        granularity: str
            Granularity of the data either "coarse" or "fine".
        rotate: bool
            Whether to apply rotation to the data (used in rotation ablation study).
        batch_size: int
            Batch size for the dataloaders.
        num_workers: int
            Number of workers for the dataloaders.
    Returns:
        train_loader: wds.WebLoader (wrapper for torch.utils.data.DataLoader)
            DataLoader for the training set.
        val_loader: wds.WebLoader (wrapper for torch.utils.data.DataLoader)
            DataLoader for the validation set.
    """

    if dataset_split == "Small":
        data_path = SMALL_URL
        shards = "{0000..0031}"
        shuffle_buffer = 10
    elif dataset_split == "Base":
        data_path = BASE_URL
        shards = "{0000..0128}"
        shuffle_buffer = 50
    elif dataset_split == "Large":
        data_path = LARGE_URL
        shards = "{0000..1033}"
        shuffle_buffer = 50
    else:
        raise ValueError("dataset_split must be 'Small', 'Base' or 'Large'")
    
    webdataset_path = data_path + f"shard_{shards}.tar"    
    
    transform_train = get_transform(data_dim, data_aug=True, rotate=rotate)
    transform_val_test = get_transform(data_dim, data_aug=False, rotate=rotate)

    decoder_train = Decoder(granularity=granularity, data_dim=data_dim, transform=transform_train, split="train")
    decoder_val = Decoder(granularity=granularity, data_dim=data_dim, transform=transform_val_test, split="val")
    decoder_test = Decoder(granularity=granularity, data_dim=data_dim, transform=transform_val_test, split="test")
   
    dataset_train = (   
        wds.WebDataset(webdataset_path, shardshuffle=True)
        .decode()
        .shuffle(shuffle_buffer)
        .select(decoder_train.check_split)
        .map(decoder_train.decode)
        .map(decoder_train.apply_transform)
        .to_tuple("image", "label")
        .batched(batch_size)
    )

    dataset_val = (
        wds.WebDataset(webdataset_path, shardshuffle=False)
        .decode()
        .select(decoder_val.check_split)
        .map(decoder_val.decode)
        .map(decoder_val.apply_transform)
        .to_tuple("image", "label")
        .batched(batch_size)
    )

    dataset_test = (
        wds.WebDataset(webdataset_path, shardshuffle=False)
        .decode()
        .select(decoder_test.check_split)
        .map(decoder_test.decode)
        .map(decoder_test.apply_transform)
        .to_tuple("image", "label")
        .batched(batch_size)
    )

    train_loader = wds.WebLoader(dataset_train, batch_size=None, num_workers=num_workers)
    val_loader = wds.WebLoader(dataset_val, batch_size=None, num_workers=num_workers)
    test_loader = wds.WebLoader(dataset_test, batch_size=None, num_workers=num_workers)

    return train_loader, val_loader, test_loader