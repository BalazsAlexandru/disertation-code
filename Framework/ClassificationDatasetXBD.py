from PIL import Image
from torch.utils.data import Dataset

class polygonDatasetXBD(Dataset):

    def __init__(
        self,
        image_paths,
        bounding_boxes,
        labels,
        transform=None,
        padding=16
    ):
        
        self.image_paths = image_paths
        self.bounding_boxes = bounding_boxes
        self.labels = labels
        self.transform = transform
        self.padding = padding
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]
        print("Image path: ", image_path)
        bbox = self.bounding_boxes[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")

        xmin, ymin, xmax, ymax = bbox

        # Add padding around building
        xmin = max(0, int(xmin - self.padding))
        ymin = max(0, int(ymin - self.padding))
        xmax = min(image.width, int(xmax + self.padding))
        ymax = min(image.height, int(ymax + self.padding))

        # Crop
        cropped = image.crop((xmin, ymin, xmax, ymax))

        if self.transform:
            cropped = self.transform(cropped)

        return cropped, label