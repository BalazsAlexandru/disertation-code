import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

from SegmentationHyperparameters import DAMAGE_CLASSES, EPOCHS, MODEL_NAME

def compute_mean_iou(preds, targets, num_classes=5):
    ious = []
    preds = preds.cpu()
    targets = targets.cpu()

    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = targets == cls
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union == 0:
            continue
        iou = intersection / union
        ious.append(iou)

    return np.mean(ious) if ious else 0.0


def compute_single_class_iou(preds, targets, cls_id):
    preds = preds.cpu()
    targets = targets.cpu()

    pred_cls = preds == cls_id
    target_cls = targets == cls_id

    intersection = (pred_cls & target_cls).sum().item()
    union = (pred_cls | target_cls).sum().item()

    if union == 0:
        return float("nan")  # class not present in this batch

    return intersection / union


def evaluate_segmentation(model, loader, device, epoch, criterion):
    model.eval()
    val_loss = 0
    mean_iou = 0
    val_bar = tqdm(loader, desc=f"Val Epoch {epoch+1}/{EPOCHS}")

    # Track per-class IoU across all batches
    class_names = ["background"] + list(DAMAGE_CLASSES.keys())
    class_ids = [0] + list(DAMAGE_CLASSES.values())
    per_class_ious = {name: [] for name in class_names}

    with torch.no_grad():

        for images, masks in val_bar:

            images = images.to(device)
            masks = masks.long().to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            mean_iou += compute_mean_iou(preds, masks, num_classes=5)

            for name, cls_id in zip(class_names, class_ids):
                iou = compute_single_class_iou(preds, masks, cls_id)
                if not np.isnan(iou):
                    per_class_ious[name].append(iou)

    val_loss /= len(loader)
    mean_iou /= len(loader)

    print("\nPer-class IoU:")
    for name in class_names:
        values = per_class_ious[name]
        avg_iou = np.mean(values) if values else float("nan")
        print(f"  {name:15s}: {avg_iou:.4f}")

    return val_loss, mean_iou


def compute_iou(preds, targets, num_classes):
    ious = []
    for cls in range(num_classes):

        pred_mask = preds == cls
        target_mask = targets == cls

        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()

        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)

    return ious


def compute_dice(preds, targets, num_classes):

    dices = []

    for cls in range(num_classes):

        pred_mask = preds == cls
        target_mask = targets == cls

        intersection = np.logical_and(pred_mask, target_mask).sum()

        total = pred_mask.sum() + target_mask.sum()

        if total == 0:
            dices.append(np.nan)
        else:
            dices.append((2 * intersection) / total)

    return dices


def compute_metrics(all_targets, all_preds, num_classes=5):
    output_file = f"{MODEL_NAME}_evaluation_results.txt"

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    precision, recall, f1, support = precision_recall_fscore_support(all_targets, all_preds, labels=list(range(num_classes)), zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted")
    ious = compute_iou(all_preds, all_targets, num_classes)
    mean_iou = np.nanmean(ious)
    dice_scores = compute_dice(all_preds, all_targets, num_classes)
    mean_dice = np.nanmean(dice_scores)

    print(f"Mean IoU:      {mean_iou:.4f}")
    print(f"Mean Dice:     {mean_dice:.4f}")
    print(f"Macro F1:      {macro_f1:.4f}")
    print(f"Weighted F1:   {weighted_f1:.4f}")

    print("\nPer-class metrics:\n")

    for i in range(num_classes):
        print(f"Class {i}")
        print(f" Precision: {precision[i]:.4f}")
        print(f" Recall:    {recall[i]:.4f}")
        print(f" F1:        {f1[i]:.4f}")
        print(f" IoU:       {ious[i]:.4f}")
        print(f" Dice:      {dice_scores[i]:.4f}")
        print("")

    print("Confusion Matrix:\n", cm)

    with open(output_file, "w") as f:

        f.write("=== SEGMENTATION RESULTS ===\n\n")

        f.write(f"Mean IoU: {mean_iou:.4f}\n")
        f.write(f"Mean Dice: {mean_dice:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")

        f.write("Per-class metrics:\n\n")
        for i in range(num_classes):
            f.write(f"Class {i}\n")
            f.write(f"Precision: {precision[i]:.4f}\n")
            f.write(f"Recall: {recall[i]:.4f}\n")
            f.write(f"F1: {f1[i]:.4f}\n")
            f.write(f"IoU: {ious[i]:.4f}\n")
            f.write(f"Dice: {dice_scores[i]:.4f}\n\n")

        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    print("\nSaved results to", output_file) 



