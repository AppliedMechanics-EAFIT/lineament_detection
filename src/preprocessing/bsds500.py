import numpy as np
import scipy
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import os
from torchvision import transforms
from torchvision.transforms.v2 import GaussianNoise

def image_transformation(noise=True):
    image_transform = transforms.Compose([
        transforms.Resize((320, 480)),
        # transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        GaussianNoise(mean=0.0, sigma=10.0/255.0) if noise else transforms.Lambda(lambda x: x),
        transforms.Normalize(mean=[0.5], std=[0.25])
    ])
    return image_transform
    
def label_transformation(noise=True):
    label_transform = transforms.Compose([
        transforms.Resize((320, 480)),
        # transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        GaussianNoise(mean=0.0, sigma=10.0/255.0) if noise else transforms.Lambda(lambda x: x)
    ])
    return label_transform

class BSDS500Dataset(Dataset):
    def __init__(self, image_dir, label_dir, image_transform=None, label_transform=None, rotations=None, input_channels=3):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.input_channels = input_channels

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.mat')])

        self.rotations = rotations if rotations is not None else [0]
        self.index = [(i, angle) for i in range(len(self.image_files)) for angle in self.rotations]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        img_idx, angle = self.index[idx]

        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[img_idx])
        if self.input_channels == 1:
            image = Image.open(img_path).convert("L")
        else:
            image = Image.open(img_path).convert("RGB")

        # Load label
        label_path = os.path.join(self.label_dir, self.label_files[img_idx])
        label = scipy.io.loadmat(label_path)['groundTruth'][0][0][0][0][1]
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
