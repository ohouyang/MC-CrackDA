import os
import cv2
import torch
import random
import numpy as np
from copy import deepcopy
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from dataset.crack_dataset import CrackDataset
from dataset.uda_dataset import UDADataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import OrderedDict


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
max_pixel_value = 255.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = os.path.join('D:/MC-CrackDA/')


def get_loader(img_dir, mask_dir, batch_size, num_workers=2, transform=None, pin_memory=True):
    dataset = CrackDataset(image_dir=img_dir, label_dir=mask_dir, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=pin_memory, shuffle=True)
    return dataloader


def get_uda_loader(source_img_dir, source_mask_dir, target_img_dir, target_mask_dir, batch_size,
                   data_strategy='target', transform=None, num_workers=4, pin_memory=True):
    source_dataset = CrackDataset(image_dir=source_img_dir, label_dir=source_mask_dir)
    target_dataset = CrackDataset(image_dir=target_img_dir, label_dir=target_mask_dir)

    uda_dataset = UDADataset(source=source_dataset, target=target_dataset,
                             transform=transform, data_strategy=data_strategy)
    uda_dataloader = DataLoader(dataset=uda_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=pin_memory)
    return uda_dataloader


def get_transform(cfg):
    train_transform = A.Compose(
        [
            A.Resize(height=cfg.img_height, width=cfg.img_width),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
            A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT, fill_mask=0., fill=255),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean=mean,
                std=std,
                max_pixel_value=max_pixel_value
            ),
            ToTensorV2()
        ], additional_targets={'img_brightness': 'mask'})
    val_transform = A.Compose(
        [
            A.Resize(height=cfg.img_height, width=cfg.img_width),
            A.Normalize(
                mean=mean,
                std=std,
                max_pixel_value=max_pixel_value
            ),
            ToTensorV2()
        ])

    return train_transform, val_transform


def set_random():
    seed = random.randint(1, 100)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_default_parser():
    parser = ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
    parser.add_argument('--batch_size_val', type=int, default=2, help='input val batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data process workers')
    parser.add_argument('--max_epoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--source_train_img_dir', type=str, required=True, help='source train image directory')
    parser.add_argument('--source_train_mask_dir', type=str, required=True, help='source train mask directory')
    parser.add_argument('--source_val_img_dir', type=str, required=False, help='source val image directory')
    parser.add_argument('--source_val_mask_dir', type=str, required=False, help='source val mask directory')
    parser.add_argument('--target_train_img_dir', type=str, required=False, help='target train image directory')
    parser.add_argument('--target_train_mask_dir', type=str, required=False, help='target train mask directory')
    parser.add_argument('--target_val_img_dir', type=str, required=False, help='target val image directory')
    parser.add_argument('--target_val_mask_dir', type=str, required=False, help='target val mask directory')
    parser.add_argument('--save_dir', type=str, required=True, help='save directory')
    parser.add_argument('--save_interval', type=int, default=10, help='save model interval')
    parser.add_argument('--load_model', type=str, default=None, help='pretrained model')
    parser.add_argument('--img_height', type=int, default=512, help='crop size height')
    parser.add_argument('--img_width', type=int, default=512, help='crop size width')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning rate of the model')

    return parser


def load_weights(model, weights_path):
    checkpoint = torch.load(weights_path)
    state_dict = checkpoint['state_dict']
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(OrderedDict(pretrained_dict))


def update_ema_model(model, ema_model, ema_alpha):
    model_state = model.state_dict()
    ema_state = ema_model.state_dict()

    for key in model_state.keys():
        if model_state[key].dtype.is_floating_point:
            ema_state[key].mul_(ema_alpha).add_(model_state[key] * (1.0 - ema_alpha))
        else:
            ema_state[key] = model_state[key]


def denormalize_image(image_tensor):
    image = deepcopy(image_tensor)  # .cpu().numpy()

    _mean = np.array(mean).reshape((3, 1, 1))
    _std = np.array(std).reshape((3, 1, 1))
    denormalized = image * _std + _mean
    denormalized = denormalized * max_pixel_value
    denormalized = np.clip(denormalized, 0, max_pixel_value).astype(np.uint8)
    denormalized = np.transpose(denormalized, (1, 2, 0))
    return denormalized
