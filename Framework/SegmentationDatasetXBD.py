import os
import json
import random

import torch
import cv2
import numpy as np
import rasterio

from rasterio.transform import rowcol
from torch.utils.data import Dataset
from shapely import wkt

from SegmentationHyperparameters import DAMAGE_CLASSES, PATCH_SIZE

class SegmentationDataset(Dataset):

    def __init__(self, image_dir, label_dir, transform=None, patch_size=PATCH_SIZE, stride=190, bg_ratio=0.3):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.damage_classes = DAMAGE_CLASSES

        self.image_paths = sorted([
            os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if f.endswith(".png") and "post_disaster" in f
        ])

        self.label_paths = sorted([
            os.path.join(self.label_dir, f)
            for f in os.listdir(self.label_dir)
            if f.endswith(".json") and "post_disaster" in f
        ])

        self.samples = []  # now stores (img_path, label_path, cx, cy) per patch

        # Sanity check to ensure images and labels are aligned
        for img_path, label_path in zip(self.image_paths, self.label_paths):
            img_stem = os.path.basename(img_path).replace(".png", "")
            lbl_stem = os.path.basename(label_path).replace(".json", "")
            assert img_stem == lbl_stem, f"Mismatch: {img_stem} vs {lbl_stem}"

        all_building_patches = []
        all_bg_patches = []    

        for img_path, label_path in zip(self.image_paths, self.label_paths):

            with rasterio.open(img_path) as src:
                geo_transform = src.transform
                H, W = src.height, src.width

            with open(label_path, "r") as f:
                data = json.load(f)

            half = patch_size // 2

            # --- Generate building-centered patches ---
            for feature in data["features"]["xy"]:
                damage = feature["properties"].get("subtype")
                if damage not in self.damage_classes:
                    continue

                polygon = wkt.loads(feature["wkt"])
                xs, ys = [], []

                for x, y in polygon.exterior.coords:
                    row, col = rowcol(geo_transform, x, y)
                    xs.append(col)
                    ys.append(row)

                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                cx = int((xmin + xmax) / 2)
                cy = int((ymin + ymax) / 2)

                # Skip buildings too close to the border to form a full patch
                if cx - half < 0 or cy - half < 0 or cx + half > W or cy + half > H:
                    continue

                all_building_patches.append((img_path, label_path, cx, cy))


            # --- Generate background patches via sliding window ---
            for y in range(half, H - half, stride):
                for x in range(half, W - half, stride):
                    all_bg_patches.append((img_path, label_path, x, y))

        n_building = len(all_building_patches)
        n_bg = int(n_building * (bg_ratio / (1 - bg_ratio)))
        n_bg = min(n_bg, len(all_bg_patches))
        sampled_bg = random.sample(all_bg_patches, n_bg)

        self.samples = all_building_patches + sampled_bg

        print(f"Total patches:       {len(self.samples)}")
        print(f"  Building-centered: {len(all_building_patches)}")
        print(f"  Background:        {len(sampled_bg)}")


    def __len__(self):
        return len(self.samples)
    

    def _crop(self, image, mask, cx, cy):
        half = self.patch_size // 2
        h, w = image.shape[:2]
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, x1 + self.patch_size)
        y2 = min(h, y1 + self.patch_size)
        x1 = x2 - self.patch_size
        y1 = y2 - self.patch_size
        return image[y1:y2, x1:x2], mask[y1:y2, x1:x2]


    def __getitem__(self, idx):

        img_path, label_path, cx, cy = self.samples[idx]  # cx, cy pre-decided in __init__

        # Read image and get geotransform
        with rasterio.open(img_path) as src:
            image_np = src.read()  # (C, H, W)
            geo_transform = src.transform
            H, W = src.height, src.width

        image_np = np.transpose(image_np, (1, 2, 0))

        # Create empty mask
        mask = np.zeros((H, W), dtype=np.uint8)

        with open(label_path, "r") as f:
            data = json.load(f)

        # Loop through each building and draw its polygon on the mask
        for building in data["features"]["xy"]:

            damage = building["properties"].get("subtype")
            if damage not in self.damage_classes:
                continue

            class_id = self.damage_classes[damage]
            polygon = wkt.loads(building["wkt"])
            coords = []

            for x, y in polygon.exterior.coords:
                row, col = rowcol(geo_transform, x, y)
                coords.append([col, row])

            coords = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [coords], class_id)

        # Crop at the pre-decided center — no randomness here
        image_np, mask = self._crop(image_np, mask, cx, cy)

        # Apply augmentations if provided
        if self.transform:
            transformed = self.transform(image=image_np, mask=mask)
            image_np = transformed["image"]
            mask = transformed["mask"]

        # Convert to tensors
        if not isinstance(image_np, torch.Tensor):
            image_np = np.transpose(image_np, (2, 0, 1))
            image_np = torch.tensor(image_np, dtype=torch.float32) / 255.0

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.long)

        return image_np, mask.long()
        
        