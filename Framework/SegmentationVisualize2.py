import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import rasterio
import json
import cv2

from SegmentationDatasetXBD import SegmentationDataset
from SegmentationHyperparameters import (
    IMAGES_DIRECTORY_TRAIN_PATH,
    LABELS_DIRECTORY_TRAIN_PATH,
    PATCH_SIZE,
    DAMAGE_CLASSES
)
from rasterio.transform import rowcol
from shapely import wkt


# -----------------------------
# VISUALIZATION FUNCTION
# -----------------------------
def visualize_with_context(full_image, full_mask, patch_image, patch_mask, patch_coords, title="Sample"):
    """
    full_image   : (H, W, 3) uint8 - the original full-resolution image
    full_mask    : (H, W)    uint8 - the full segmentation mask
    patch_image  : (C, H, W) float32 tensor or (H, W, 3) array - the cropped patch
    patch_mask   : (H, W)    tensor or array - the cropped mask
    patch_coords : (x1, y1, x2, y2) pixel coords of the patch on the full image
    """

    # --- Denormalize patch image ---
    if torch.is_tensor(patch_image):
        patch_image = patch_image.cpu().numpy()
        patch_image = np.transpose(patch_image, (1, 2, 0))
    patch_image = (patch_image * 255).clip(0, 255).astype(np.uint8)

    if torch.is_tensor(patch_mask):
        patch_mask = patch_mask.cpu().numpy()

    # --- Draw patch rectangle on a copy of the full image ---
    full_vis = full_image.copy()
    x1, y1, x2, y2 = patch_coords
    cv2.rectangle(full_vis, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 0: full image context
    axes[0, 0].imshow(full_vis)
    axes[0, 0].set_title("Full Image (patch in red)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(full_mask, cmap="tab10", vmin=0, vmax=9)
    axes[0, 1].add_patch(plt.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        edgecolor="red", facecolor="none", linewidth=2
    ))
    axes[0, 1].set_title("Full Mask (patch in red)")
    axes[0, 1].axis("off")

    # Zoomed-in region from the full image (without normalization artifacts)
    axes[0, 2].imshow(full_image[y1:y2, x1:x2])
    axes[0, 2].set_title("Zoomed Patch (raw)")
    axes[0, 2].axis("off")

    # Row 1: patch detail
    axes[1, 0].imshow(patch_image)
    axes[1, 0].set_title("Patch Image (normalized)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(patch_mask, cmap="tab10", vmin=0, vmax=9)
    axes[1, 1].set_title("Patch Mask")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(patch_image)
    axes[1, 2].imshow(patch_mask, alpha=0.5, cmap="tab10", vmin=0, vmax=9)
    axes[1, 2].set_title("Patch Overlay")
    axes[1, 2].axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# -----------------------------
# HELPER: build full mask for one sample
# -----------------------------
def build_full_mask(label_path, transform, H, W):
    with open(label_path, "r") as f:
        data = json.load(f)

    mask = np.zeros((H, W), dtype=np.uint8)

    for building in data["features"]["xy"]:
        damage = building["properties"].get("subtype")
        if damage not in DAMAGE_CLASSES:
            continue

        class_id = DAMAGE_CLASSES[damage]
        polygon = wkt.loads(building["wkt"])
        coords = []

        for x, y in polygon.exterior.coords:
            row, col = rowcol(transform, x, y)
            coords.append([col, row])

        coords = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [coords], class_id)

    return mask


# -----------------------------
# HELPER: compute patch coords from center
# -----------------------------
def get_patch_coords(cx, cy, patch_size, H, W):
    half = patch_size // 2

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(W, x1 + patch_size)
    y2 = min(H, y1 + patch_size)

    # Clamp to ensure exact patch size (mirrors _crop logic)
    x1 = x2 - patch_size
    y1 = y2 - patch_size

    return x1, y1, x2, y2


# -----------------------------
# MAIN INSPECTION LOOP
# -----------------------------
def main():

    dataset = SegmentationDataset(
        IMAGES_DIRECTORY_TRAIN_PATH,
        LABELS_DIRECTORY_TRAIN_PATH,
        transform=None
    )

    print("Dataset size:", len(dataset))

    for i in range(10):
        idx = random.randint(0, len(dataset) - 1)
        img_path, label_path, cx, cy = dataset.samples[idx]
        print(f"\nInspecting sample {i} | idx={idx} | center=({cx},{cy}) | img_path={img_path}")

        # --- Load the full original image ---
        with rasterio.open(img_path) as src:
            full_np = src.read()           # (C, H, W)
            geo_transform = src.transform
            H, W = src.height, src.width

        full_np = np.transpose(full_np, (1, 2, 0)).astype(np.uint8)  # (H, W, C)

        # --- Build the full mask ---
        full_mask = build_full_mask(label_path, geo_transform, H, W)

        # --- Decide patch center (mirrors __getitem__ logic) ---
        x1, y1, x2, y2 = get_patch_coords(cx, cy, PATCH_SIZE, H, W)

        # --- Get patch from dataset (applies /255 normalization) ---
        patch_image, patch_mask = dataset[idx]

        print(f"\nSample {i} | idx={idx}")
        print(f"Patch coords: ({x1}, {y1}) -> ({x2}, {y2})")
        print(f"Unique mask values in patch: {np.unique(patch_mask.cpu().numpy() if torch.is_tensor(patch_mask) else patch_mask)}")

        visualize_with_context(
            full_image=full_np,
            full_mask=full_mask,
            patch_image=patch_image,
            patch_mask=patch_mask,
            patch_coords=(x1, y1, x2, y2),
            title=f"Sample {idx} | center=({cx},{cy})"
        )


if __name__ == "__main__":
    main()