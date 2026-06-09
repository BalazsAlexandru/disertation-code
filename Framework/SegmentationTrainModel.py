import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from FocalLoss import FocalLoss
from SegmentationHyperparameters import EPOCHS, IMAGES_DIRECTORY_TRAIN_PATH, LABELS_DIRECTORY_TRAIN_PATH, MODEL_NAME, segmentationTrainingAugmentation, segmentationTrainingEvaluationAugmentation, LR, BATCH_SIZE
from SegmentationModels import get_model
from SegmentationDatasetXBD import SegmentationDataset
from SegmentationUtils import compute_mean_iou, evaluate_segmentation
from SegmentationWrapper import SegmentationWrapper


# ---- Set the code to run on GPU CUDA, else on CPU ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Probably should be replaced with the transformers segmentation code
dataset = SegmentationDataset(
    IMAGES_DIRECTORY_TRAIN_PATH,
    LABELS_DIRECTORY_TRAIN_PATH,
    transform=segmentationTrainingAugmentation
)

val_dataset = SegmentationDataset(
    IMAGES_DIRECTORY_TRAIN_PATH,
    LABELS_DIRECTORY_TRAIN_PATH,
    transform=segmentationTrainingEvaluationAugmentation  # only resize + normalize, no flips
)

weights = torch.tensor([
    0.2,  # background
    1.0,  # no damage
    2.0,  # minor
    3.0,  # major
    4.0   # destroyed
]).to(device)

# ---- Split dataset and create dataloaders ----
indices = torch.randperm(len(dataset)).tolist()
split = int(0.8 * len(indices))

train_dataset = torch.utils.data.Subset(dataset, indices[:split])
val_dataset = torch.utils.data.Subset(val_dataset, indices[split:])

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

#--- Model, loss function, and optimizer setup ----
model = SegmentationWrapper(get_model(MODEL_NAME, num_classes=5)).to(device)

criterion = FocalLoss(alpha=weights, gamma=2.0, ignore_index=255)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# We use torch.amp for mixed precision training, which can speed up training on compatible GPUs while maintaining model accuracy. The GradScaler helps to prevent underflow in float16 precision by dynamically scaling the loss.
scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

best_val_loss = float("inf")

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)

# =================================================
# TRAIN
# =================================================

for epoch in range(EPOCHS):
    
    model.train()
    train_loss = 0

    train_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{EPOCHS}")

    for images, masks in train_bar:

        images = images.to(device)
        masks = masks.long().to(device)

        optimizer.zero_grad()

        # We use autocast to automatically choose the precision (float16 or float32) for the operations, which can speed up training on compatible GPUs while maintaining model accuracy.
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, masks)

        # We scale the loss to prevent underflow in float16 precision, then backpropagate and update the model parameters.
        scaler.scale(loss).backward()

        # scaler.step() will unscale the gradients, check for inf/NaN values, and update the model parameters if everything is fine. If there are inf/NaN values, it will skip the update to prevent corrupting the model weights.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)

        # scaler.update() will update the scale factor for the next iteration based on whether the gradients were finite or not. If the gradients were finite, it may increase the scale factor to allow for larger steps in future iterations. If the gradients were not finite, it will decrease the scale factor to try to prevent overflow in future iterations.
        scaler.update()

        # We accumulate the loss for reporting purposes, and update the progress bar with the current batch loss.
        train_loss += loss.item()

        # Update progress bar with current batch loss
        train_bar.set_postfix(loss=f"{loss.item():.4f}")

    train_loss /= len(train_loader)
    val_loss, mean_iou = evaluate_segmentation(model, val_loader, device, epoch, criterion)


    print(
        f"\nEpoch {epoch+1}/{EPOCHS}"
        f"\nTrain Loss: {train_loss:.4f}"
        f"\nVal Loss: {val_loss:.4f}"
        f"\nMean IoU: {mean_iou:.4f}"
    )

    # ---- Save the model state dict for later use ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "model_name": MODEL_NAME,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss
        }, f"{MODEL_NAME}.pth")
        print(f"Saved best model " f"(val loss = {val_loss:.4f})")

    scheduler.step()


print("\nTraining completed.")