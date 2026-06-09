import json
import os

import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
from shapely import wkt

from ClassificationHyperparameters import DAMAGE_CLASSES, MODEL_NAME

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        # Compute loss
        loss = criterion(outputs, labels)
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Accumulate loss for reporting
        total_loss += loss.item()
        print(f"Batch Loss: {loss.item():.4f}");

    return total_loss / len(loader)


def evaluate(model, loader, device):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad(): # We don't need gradients for evaluation, so we wrap in no_grad to save memory and computations

        for images, labels in loader:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            preds = outputs.argmax(dim=1) # Get the index of the max log-probability which corresponds to the predicted class

            all_preds.extend(preds.cpu().numpy()) # Move predictions back to CPU and convert to numpy for metric calculations
            all_labels.extend(labels.cpu().numpy())
        
    accuracy = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    return accuracy, f1_weighted, f1_macro


def extract_polygons_and_building_info(LABEL_DIR, IMAGE_DIR):
    image_paths = []
    bounding_boxes = []
    labels = []

    for label_file in os.listdir(LABEL_DIR):

        if not label_file.endswith(".json") or "post_disaster" not in label_file:
            continue

        label_path = os.path.join(
            LABEL_DIR,
            label_file
        )

        with open(label_path, "r") as f:
            data = json.load(f)

        # Match image name
        image_name = label_file.replace(".json", ".png")

        image_path = os.path.join(
            IMAGE_DIR,
            image_name
        )

        if not os.path.exists(image_path):
            continue

        features = data["features"]["xy"]

        for building in features:

            props = building["properties"]

            damage = props.get("subtype")

            if damage not in DAMAGE_CLASSES:
                # print(f"Unknown damage type: {damage}, skipping building.")
                continue

            # Read polygon
            polygon = wkt.loads(building["wkt"])

            # Convert polygon → bounding box
            xmin, ymin, xmax, ymax = polygon.bounds

            image_paths.append(image_path)

            bounding_boxes.append(
                (xmin, ymin, xmax, ymax)
            )

            labels.append(
                DAMAGE_CLASSES[damage]
            )

    return image_paths, bounding_boxes, labels


def compute_metrics(all_preds, all_labels):
    output_file = f"{MODEL_NAME}_evaluation_results.txt"

    confusionMatrix = confusion_matrix(all_labels, all_preds)
    macroF1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    accuracy = accuracy_score(all_labels, all_preds)
    macro_precision = precision_score(all_labels, all_preds, average="macro")
    macro_recall = recall_score(all_labels, all_preds, average="macro")

    print("Evaluation Results for " + MODEL_NAME)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {macroF1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")
    print(f"Confusion Matrix:\n{confusionMatrix}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print("------------------------------------------------------------------------")
    print(classification_report(all_labels, all_preds, target_names=["no-damage", "minor-damage", "major-damage", "destroyed"]))

    with open(output_file, "w") as f:
        f.write(f"Evaluation Results for {MODEL_NAME}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1-score: {macroF1:.4f}\n")
        f.write(f"Weighted F1-score: {weighted_f1:.4f}\n")
        f.write(f"Confusion Matrix:\n{confusionMatrix}\n")
        f.write(f"Macro Precision: {macro_precision:.4f}\n")
        f.write(f"Macro Recall: {macro_recall:.4f}\n")
        f.write("------------------------------------------------------------------------\n")
        f.write(classification_report(all_labels, all_preds, target_names=["no-damage", "minor-damage", "major-damage", "destroyed"]))

    print("\nSaved results to", output_file) 
