import segmentation_models_pytorch
import torch
import torchvision


def get_model(model_name, num_classes):

    model_name = model_name.lower()

    if model_name == "unet":

        return segmentation_models_pytorch.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes
        )

    elif model_name == "deeplabv3_resnet50":

        model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None)

        model.classifier[-1] = torch.nn.Conv2d(
            256,
            num_classes,
            kernel_size=1
        )

        return model

    elif model_name == "deeplabv3_resnet101":

        model = torchvision.models.segmentation.deeplabv3_resnet101(
            weights=None
        )

        model.classifier[-1] = torch.nn.Conv2d(
            256,
            num_classes,
            kernel_size=1
        )

        return model

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )