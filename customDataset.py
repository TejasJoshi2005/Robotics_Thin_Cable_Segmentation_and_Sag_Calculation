import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
import os
import numpy as np

class CableRobotDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        super(CableRobotDataset, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):

        image_file_name = self.images[index]
        image_path = os.path.join(self.image_dir, image_file_name)

        mask_file_name = f"mask_{image_file_name.replace('.jpg', '.png')}"
        mask_path = os.path.join(self.mask_dir, mask_file_name)

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype = np.float32)
       
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

