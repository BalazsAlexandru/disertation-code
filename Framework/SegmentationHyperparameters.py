import albumentations as albumentations 

#---- Hyperparameters and configurations for training and evaluation ----

MODEL_NAME = "deeplabv3_resnet50"
# Possible values: unet, deeplabv3_resnet50, deeplabv3_resnet101
BATCH_SIZE = 4
# Suggested batch sizes based on model:
# U-Net - 4
EPOCHS = 2
LR = 1e-4
# U-Net - 1e-4

PATCH_SIZE = 256

DAMAGE_CLASSES = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4
}
NORMALIZE_VALUES = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # ImageNet stats for normalization since we're using pretrained models, to be changed if we train from scratch
# [0.5, 0.5, 0.5], [0.5, 0.5, 0.5] Normalization values for EFFICIENTNET_B0 when trained from scratch, to be changed if we use pretrained weights

IMAGES_DIRECTORY_TEST_PATH = r"C:\Users\40754\Desktop\Disertation\Dataset\Test Dataset\images"
LABELS_DIRECTORY_TEST_PATH = r"C:\Users\40754\Desktop\Disertation\Dataset\Test Dataset\labels"

# IMAGES_DIRECTORY_TRAIN_PATH = r"C:\Users\40754\Desktop\Disertation\Dataset\Train Dataset\train\images"
# LABELS_DIRECTORY_TRAIN_PATH = r"C:\Users\40754\Desktop\Disertation\Dataset\Train Dataset\train\labels"
IMAGES_DIRECTORY_TRAIN_PATH = r"C:\Users\40754\Desktop\Disertation\Dataset\TrainSmall\images"
LABELS_DIRECTORY_TRAIN_PATH = r"C:\Users\40754\Desktop\Disertation\Dataset\TrainSmall\labels"



# ---- Data augmentation for segmentation model, to be changed if we use pretrained weights or train from scratch ----
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

# ---- Data augmentation for segmentation model, to be changed if we use pretrained weights or train from scratch ----
segmentationTrainingEvaluationAugmentation = albumentations.Compose([

    albumentations.Resize(512, 512),
    albumentations.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    albumentations.ToTensorV2()
])