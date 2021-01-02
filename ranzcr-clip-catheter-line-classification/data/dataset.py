import os
import re
import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class RANZCRDataset(Dataset):
    def __init__(self, dataframe, config, transforms=None):
        super().__init__()
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.
        Args:
            shape: (tuple: height, width) shape of image
            dataframe: (pd.DataFrame) dataframe containing image paths and labels
            transforms: (albumentations.transforms) transformation to apply on image
            config: (Params class object) configuration data
        """
        self.df = dataframe
        self.labels = dataframe[config.target_cols].values
        self.transforms = transforms
        self.shape = (config.height, config.width)

    def __len__(self) -> int:
        """
        return size of dataset
        """
        return len(self.df)
        
    def __getitem__(self, index: int):
        """
        Fetch index image and labels from dataset. Perform transforms on image.
        Args:
            index: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            image: (Tensor) transformed image
            label: corresponding label of image
        """
        image_path = self.df['image_path'][index]
        
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = cv2.resize(image, self.shape)
        image /= 255.0

        if self.transforms:
            image = self.transforms(image=image)['image']

        label = torch.tensor(self.labels[index]).float()
            
        return image, label

def fetch_dataloader(dataframe, config, data='train'):
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        dataframe: (dataframe) Dataframe containing data
        config: (Params class object) configuration parameters for dataloader
        data: (string) type of data (train or valid)
    Returns:
        data_loader: (torch.utils.data.DataLoader) contains
                        the DataLoader object
    """
    # get parameters
    batch_size = config.batch_size
    num_workers = config.num_workers

    if data == 'train':
        shuffle = True
    elif data == 'valid':
        shuffle = False

    # load dataset
    dataset = RANZCRDataset(dataframe, config, transforms = get_transforms(config, data))

    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, 
        shuffle=shuffle, pin_memory=True)
    
    return data_loader


def get_transforms(config, data):
    if data == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(config.height, config.width, scale=(0.85, 1.0)),
            A.RandomContrast(limit=0.2, always_apply=False, p=0.5),
            ToTensorV2(p=1.0),
        ])
    elif data == 'valid':
        return A.Compose([
            ToTensorV2(p=1.0),
        ])