import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ClassificationHyperparameters import *
from ClassificationDatasetXBD import polygonDatasetXBD
from ClassificationModels import get_model
from ClassificationUtils import compute_metrics, extract_polygons_and_building_info

# Set the code to run on GPU CUDA, else on CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ---- GET TEST DATA ----
image_paths, bounding_boxes, labels = extract_polygons_and_building_info(LABELS_DIRECTORY_TEST_PATH, IMAGES_DIRECTORY_TEST_PATH)

dataset = polygonDatasetXBD(
    image_paths,
    bounding_boxes,
    labels,
    transform=evaluationAugmentation
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False
)

#---- RELOAD MODEL ----
model = get_model(MODEL_NAME, num_classes=4).to(device)

model.load_state_dict(torch.load(f"{MODEL_NAME}.pth", map_location=device))

#---- EVALUATE ----
model.eval() # Set model to evaluation mode (disables dropout, batchnorm, etc.)

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        preds = outputs.argmax(dim=1) # Get the index of the max log-probability which corresponds to the predicted class

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


#---- COMPUTE METRICS ----
compute_metrics(all_preds, all_labels)
    