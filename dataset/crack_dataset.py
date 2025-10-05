import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class CrackDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, img_suffix='.png', lbl_suffix='.png'):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_suffix = img_suffix
        self.lbl_suffix = lbl_suffix
        self.images = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        label_path = os.path.join(self.label_dir, self.images[index].replace(self.img_suffix, self.lbl_suffix))
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)

        _, label = cv2.threshold(label, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # binaryzation
        label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_NEAREST)

        if self.transform is not None:
            augmentation = self.transform(image=image, mask=label)
            image = augmentation["image"]
            label = augmentation["mask"]
        return image, label
