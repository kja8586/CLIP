"""Flickr30k dataset and data loading utilities."""

import os

import pandas as pd
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import random_split, Dataset, DataLoader
from transformers import DistilBertTokenizer

from config import CFG


def get_transforms():
    """Return image transforms for training/evaluation."""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class FlickrDataset(Dataset):
    """Flickr30k dataset for image-text pairs."""

    def __init__(self, data_path=None, caption_path=None, model_name=None, transforms=None):
        self.data_path = data_path or CFG.data_path
        self.transforms = transforms

        tokenizer_name = model_name or CFG.textTokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)

        self.df = pd.read_csv(caption_path or CFG.caption_path, delimiter="|")
        self.df = self.df.dropna(subset=[" comment"]).reset_index(drop=True)

        unique_images = self.df["image_name"].unique()
        self.img_name_to_id = {name: idx for idx, name in enumerate(unique_images)}
        self.num_unique_images = len(unique_images)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.data_path, row["image_name"])
        img = Image.open(image_path).convert("RGB")
        cap = row[" comment"]
        image_id = self.img_name_to_id[row["image_name"]]

        if self.transforms:
            img = self.transforms(img)

        txt = self.tokenizer(
            cap,
            padding="max_length",
            truncation=True,
            max_length=CFG.max_length,
            return_tensors="pt",
        )
        txt = {k: v.squeeze(0) for k, v in txt.items()}

        return img, txt, image_id


def get_dataloaders(batch_size=None, val_ratio=None, num_workers=None):
    """Create train and validation DataLoaders."""
    bs = batch_size or CFG.batch_size
    vr = val_ratio or CFG.val_ratio
    nw = num_workers or CFG.num_workers

    transforms = get_transforms()
    dataset = FlickrDataset(transforms=transforms)

    val_size = int(len(dataset) * vr)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=nw)

    return train_loader, val_loader, train_dataset, val_dataset
