import torch
import numpy as np
import imgaug
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from pathlib import Path
import matplotlib.pyplot as plt


class LungDataset(torch.utils.data.Dataset):
    def __init__(self, root, aug):
        self.all_files = self.extract_files(root)
        self.aug = aug

    @staticmethod
    def extract_files(root):
        files = []
        for subject in root.glob("*"):
            slice_path = subject/"data"
            for slice in slice_path.glob("*.npy"):
                files.append(slice)
        return files
    
    @staticmethod
    def images_to_labels(path):
        parts = list(path.parts)
        parts[parts.index("data")] = "masks"
        return Path(*parts)

    def augment(self, slice, mask):
        random_seed = torch.randint(0, 1000000, (1,)).item() 
        imgaug.seed(random_seed)
        mask = SegmentationMapsOnImage(mask, mask.shape)
        slice_aug, mask_aug = self.aug(image=slice, segmentation_maps=mask)
        mask_aug = mask_aug.get_arr()
        return slice_aug, mask_aug

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        file_path = self.all_files[idx]
        mask_path = self.images_to_labels(file_path)
        slice = np.load(file_path).astype(np.float32) 
        mask = np.load(mask_path).astype(np.int32)

        if self.aug:
            slice, mask = self.augment(slice, mask)
        
        return np.expand_dims(slice, 0), np.expand_dims(mask, 0)