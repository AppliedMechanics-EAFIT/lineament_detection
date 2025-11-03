import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from skimage.io import imread

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import GaussianNoise

import os

def image_transformation(noise=True):
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        GaussianNoise(mean=0.0, sigma=10.0/255.0) if noise else transforms.Lambda(lambda x: x),
        transforms.Normalize(mean=[0.5], std=[0.25])
    ])
    return image_transform

def label_transformation(noise=True):
    label_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        GaussianNoise(mean=0.0, sigma=10.0/255.0) if noise else transforms.Lambda(lambda x: x),
    ])
    return label_transform

class faultMappingDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_transform=None, label_transform=None, rotations=None, every_n=1, input_channels=3):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.input_channels = input_channels

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.tif')])

        self.rotations = rotations if rotations is not None else [0]
        
        base_indices = list(range(0, len(self.image_files), every_n))
        self.index = [(i, angle) for i in base_indices for angle in self.rotations]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        img_idx, angle = self.index[idx]

        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[img_idx])
        image = imread(img_path)
        if self.input_channels == 1:
            image = Image.fromarray(image[..., :3].astype(np.uint8)).convert("L")
        else:
            image = Image.fromarray(image[..., :3].astype(np.uint8)).convert("RGB")

        # Load label
        label_path = os.path.join(self.label_dir, self.label_files[img_idx])
        label = imread(label_path)
        label = Image.fromarray((label * 255).astype(np.uint8))

        # Rotation (for data augmentation)
        if angle != 0:
            if self.input_channels == 1:
                image = TF.rotate(image, angle=angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0)
            else:
                image = TF.rotate(image, angle=angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=(0, 0, 0))
            label = TF.rotate(label, angle=angle, interpolation=InterpolationMode.NEAREST, expand=False, fill=0)

        # Transforms
        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            label = self.label_transform(label)

        return [image, label]