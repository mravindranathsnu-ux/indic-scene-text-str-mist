"""
dataset.py
Dataset for scene-text-recognition (cropped word/line images + labels),
script/language-agnostic — works with any TSV in image_path<TAB>label
format regardless of script.

Expected annotation file (TSV), one sample per line:
    image_path<TAB>label
e.g.
    images/img_0001.jpg	తెలుగు
"""
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


def build_train_augmentation(fill=(127, 127, 127)):
    """
    Augmentation applied to the raw PIL image BEFORE CLIP preprocessing
    (letterbox resize + normalize), training split only. CLIP's visual
    backbone was pretrained on clean natural photos, not text crops from
    real-world cameras/phones, so these help it generalize to the kind of
    lighting/angle/blur variation real scene-text images have — this
    directly targets the "overfits a tiny batch but generalizes poorly on
    the full test set" symptom, since without augmentation the model can
    just memorize exact pixel patterns from a limited training set.
    """
    return T.Compose([
        T.RandomApply([T.RandomRotation(degrees=5, fill=fill)], p=0.5),
        T.RandomApply([T.RandomPerspective(distortion_scale=0.15, p=1.0, fill=fill)], p=0.3),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.25),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15),
    ])


class SceneTextDataset(Dataset):
    def __init__(self, annotation_file, image_root, vocab, preprocess, max_len=32, sep="\t", augment=False):
        self.image_root = image_root
        self.vocab = vocab
        self.preprocess = preprocess
        self.max_len = max_len
        self.augment = build_train_augmentation() if augment else None
        self.samples = []

        with open(annotation_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split(sep)
                if len(parts) < 2:
                    continue
                img_path, label = parts[0], sep.join(parts[1:])
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        full_path = os.path.join(self.image_root, img_path)
        image = Image.open(full_path).convert("RGB")
        if self.augment is not None:
            image = self.augment(image)
        image = self.preprocess(image)

        token_ids = self.vocab.encode(label)[: self.max_len]
        return {
            "image": image,
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "label": label,
        }


def collate_fn(batch, pad_id):
    images = torch.stack([b["image"] for b in batch], dim=0)
    lengths = [len(b["token_ids"]) for b in batch]
    max_len = max(lengths)

    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        padded[i, : len(b["token_ids"])] = b["token_ids"]

    labels = [b["label"] for b in batch]
    return {"images": images, "token_ids": padded, "labels": labels}


# Backward-compatible alias — existing code/imports using TeluguSceneTextDataset keep working.
TeluguSceneTextDataset = SceneTextDataset
