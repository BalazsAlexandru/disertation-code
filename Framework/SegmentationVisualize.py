import matplotlib.pyplot as plt
from ClassificationHyperparameters import IMAGES_DIRECTORY_TRAIN_PATH, LABELS_DIRECTORY_TRAIN_PATH, segmentationTrainingAugmentation

from SegmentationDatasetXBD import SegmentationDataset
import numpy as np
import torch

dataset = SegmentationDataset(
    IMAGES_DIRECTORY_TRAIN_PATH,
    LABELS_DIRECTORY_TRAIN_PATH,
    transform=None
)


# Load sample
image, mask = dataset[3]


# Plot
plt.figure(figsize=(12, 6))


# Original image
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")

# Mask
plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="jet")
plt.title("Mask")

# Overlay
plt.subplot(1, 3, 3)
plt.imshow(image)
plt.imshow(mask, alpha=0.5, cmap="jet")
plt.title("Overlay")

plt.show()