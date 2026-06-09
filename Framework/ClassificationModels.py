import torch.nn as nn
import torchvision.models as models

def get_resnet18(num_classes=4): # Default to 4 classes (undamaged, minor, major, destroyed)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes) # Get the number of features from the original final layer and replace it with a new one for our number of classes

    return model

def get_resnet50(num_classes=4): # Default to 4 classes (undamaged, minor, major, destroyed)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def get_resnet101(num_classes=4): # Default to 4 classes (undamaged, minor, major, destroyed)
    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def get_efficientnet_b0(num_classes=4):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Replace final layer
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model

def get_model(model_name, num_classes=4):
    if model_name == "resnet18":
        return get_resnet18(num_classes)
    elif model_name == "resnet50":
        return get_resnet50(num_classes)
    elif model_name == "resnet101":
        return get_resnet101(num_classes)
    elif model_name == "efficientnet_b0":
        return get_efficientnet_b0(num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")