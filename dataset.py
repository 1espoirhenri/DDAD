import os
from glob import glob
from pathlib import Path
import shutil
import numpy as np
import csv
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR10



class Dataset_maker(torch.utils.data.Dataset):
    def __init__(self, root, category, config, is_train=True):
        self.config = config

        # Transformations for images
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),  
                transforms.ToTensor(),  # Scales data into [0, 1]
                transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
            ]
        )

        # Transformations for masks
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),
                transforms.ToTensor(),  # Scales data into [0, 1]
            ]
        )

        # Load training or testing data paths
        if is_train:
            if category:
                self.image_files = glob(
                    os.path.join(root, category, "train", "*", "*.jpg")
                )
            else:
                self.image_files = glob(os.path.join(root, "train", "*", "*.jpg"))
        else:
            if category:
                self.image_files = glob(
                    os.path.join(root, category, "test", "*", "*.jpg")
                )
            else:
                self.image_files = glob(os.path.join(root, "test", "*", "*.jpg"))

        self.is_train = is_train

    def __getitem__(self, index):
        # Load image
        image_file = self.image_files[index]
        image = Image.open(image_file).convert("RGB")  # Ensure 3-channel RGB
        image = self.image_transform(image)

        # Handle grayscale images (1 channel)
        if image.shape[0] == 1:
            image = image.expand(3, self.config.data.image_size, self.config.data.image_size)

        if self.is_train:
            # Return image and label for training
            label = 'good'
            return image, label
        else:
            # Test case: check for 'good' or 'defective' labels
            if self.config.data.mask:
                if os.path.dirname(image_file).endswith("good"):
                    # Good sample, no mask
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    label = 'good'
                else:
                    # Defective sample, load mask
                    if self.config.data.name == 'MVTec':
                        target = Image.open(
                            image_file.replace("/test/", "/ground_truth/").replace(
                                ".png", "_mask.jpg"
                            )
                        ).convert("L")  # Ensure single channel
                    else:
                        target_file = image_file.replace("/test/", "/ground_truth/").replace(".jpg", ".png")
                        target = Image.open(target_file).convert("L")  # Ensure single channel

                    target = self.mask_transform(target)
                    label = 'defective'
            else:
                # If no mask is used
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                label = 'good' if os.path.dirname(image_file).endswith("good") else 'defective'

            return image, target, label

    def __len__(self):
        return len(self.image_files)
