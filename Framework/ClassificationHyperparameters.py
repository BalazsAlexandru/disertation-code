from torchvision import transforms
import albumentations as albumentations 

#---- Hyperparameters and configurations for training and evaluation ----

MODEL_NAME = "resnet18"
# Possible values: "resnet18", "resnet50", "resnet101", "efficientnet_b0"
BATCH_SIZE = 32
# Suggested batch sizes based on model:
# ResNet18 - 32
# ResNet50 - 16
# ResNet101	- 8
# EfficientNet_B0 - 16 or 8
EPOCHS = 2
LR = 1e-3
# ResNet18 - 1e-3
# EfficientNet_B0 - 1e-4
IMAGE_SIZE = (224, 224)
# 224x224 - resnet18
# 240x240 - efficientnet_b0
DAMAGE_CLASSES = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3
}

NORMALIZE_VALUES = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # ImageNet stats for normalization since we're using pretrained models, to be changed if we train from scratch
# [0.5, 0.5, 0.5], [0.5, 0.5, 0.5] Normalization values for EFFICIENTNET_B0 when trained from scratch, to be changed if we use pretrained weights

IMAGES_DIRECTORY_TEST_PATH = r"C:\Users\40754\Desktop\Disertation\Dataset\Test Dataset\images"
LABELS_DIRECTORY_TEST_PATH = r"C:\Users\40754\Desktop\Disertation\Dataset\Test Dataset\labels"

IMAGES_DIRECTORY_TRAIN_PATH = r"C:\Users\40754\Desktop\Disertation\Dataset\Train Dataset\train\images"
LABELS_DIRECTORY_TRAIN_PATH = r"C:\Users\40754\Desktop\Disertation\Dataset\Train Dataset\train\labels"




#---- Data augmentation for training and evaluation ----

trainingAugmentation = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),

    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),

    transforms.ToTensor(), # Convert PIL image to tensor and scale to [0, 1]

    transforms.Normalize(
        mean=NORMALIZE_VALUES[0],
        std=NORMALIZE_VALUES[1]
    ) # Normalize using ImageNet stats (since we're using pretrained models) to be changed afterwards to train it from scratch? need to check
])

evaluationAugmentation = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),

    transforms.ToTensor(), # Convert PIL image to tensor and scale to [0, 1]

    transforms.Normalize(
        mean=NORMALIZE_VALUES[0],
        std=NORMALIZE_VALUES[1]
    ) # Normalize using ImageNet stats (since we're using pretrained models) to be changed afterwards to train it from scratch? need to check
])

# Data augmentation for segmentation model, to be changed if we use pretrained weights or train from scratch
segmentationTrainingAugmentation = albumentations.Compose([

    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.RandomRotate90(p=0.5),
    albumentations.Resize(512, 512),

    albumentations.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    albumentations.ToTensorV2()
])