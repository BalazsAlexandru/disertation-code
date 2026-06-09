import torch
from torch.utils.data import DataLoader, random_split
from ClassificationDatasetXBD import polygonDatasetXBD
from ClassificationHyperparameters import *
from ClassificationModels import get_model
from ClassificationUtils import extract_polygons_and_building_info, train_one_epoch, evaluate

#Set the code to run on GPU CUDA, else on CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ---- DATA LOAD AND FETCH THE BUILDINGS POLYGONS ----
image_paths = []
bounding_boxes = []
labels = []

image_paths, bounding_boxes, labels = extract_polygons_and_building_info(LABELS_DIRECTORY_TRAIN_PATH, IMAGES_DIRECTORY_TRAIN_PATH)

print("Total samples:", len(image_paths))
print("Bounding boxes:", len(bounding_boxes))
print("Labels:", len(labels))

dataset = polygonDatasetXBD(
    image_paths,
    bounding_boxes,
    labels,
    transform=trainingAugmentation
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)

# ---- MODEL ----
model = get_model(MODEL_NAME, num_classes=4).to(device)

# ---- TRAINING SETUP ----
criterion = torch.nn.CrossEntropyLoss() 
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

print(model.state_dict())
best_f1_macro = 0.0

#---- TRAIN LOOP ----
for epoch in range(EPOCHS):
    loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    accuracy, f1_weighted, f1_macro = evaluate(model, val_loader, device)

    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={accuracy:.4f}, F1_Weighted={f1_weighted:.4f}, F1_Macro={f1_macro:.4f}")

    if f1_macro > best_f1_macro:
        best_f1_macro = f1_macro
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_name": MODEL_NAME,
            "f1_macro": f1_macro
        }, f"{MODEL_NAME}.pth")
        print(f"New best model saved with F1 Macro: {best_f1_macro:.4f}")

