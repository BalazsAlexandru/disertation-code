import numpy as np
import torch
from SegmentationDatasetXBD import SegmentationDataset
from SegmentationHyperparameters import LABELS_DIRECTORY_TEST_PATH, IMAGES_DIRECTORY_TEST_PATH, MODEL_NAME, segmentationTrainingEvaluationAugmentation
from SegmentationModels import get_model
from tqdm import tqdm

from SegmentationUtils import compute_metrics
from SegmentationWrapper import SegmentationWrapper

# Set the code to run on GPU CUDA, else on CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 5
BATCH_SIZE = 1

print("Using device:", device)

# ---- Get dataset for evaluation ----
test_dataset = SegmentationDataset(
    IMAGES_DIRECTORY_TEST_PATH,
    LABELS_DIRECTORY_TEST_PATH,
    transform=segmentationTrainingEvaluationAugmentation  # only resize + normalize, no flips
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size = BATCH_SIZE,
    shuffle = False,
    num_workers = 0
)

print("Test samples:", len(test_dataset))

# ---- Load Model ----
model = SegmentationWrapper(get_model(MODEL_NAME, num_classes=5)).to(device)

# Load the saved model checkpoint
checkpoint = torch.load(f"{MODEL_NAME}.pth", map_location = device)

# Load the model state dictionary from the checkpoint
model.load_state_dict(checkpoint["model_state_dict"])

model.eval() # Set model to evaluation mode (disables dropout, batchnorm, etc.)

# ----Storage ----
all_preds = []
all_targets = []

#--- Evaluate on test set ----
with torch.no_grad():

    for images, masks in tqdm(test_loader, desc="Evaluating"):

        images = images.to(device)

        preds = torch.argmax(model(images), dim=1) # Get the index of the max log-probability which corresponds to the predicted class

        all_preds.append(preds.cpu().numpy().flatten())
        all_targets.append(masks.numpy().flatten())

all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

#---- Compute metrics ----
compute_metrics(all_targets, all_preds, num_classes=NUM_CLASSES)