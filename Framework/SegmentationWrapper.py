import torch


class SegmentationWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):

        out = self.model(x)

        if isinstance(out, dict):
            return out["out"]

        return out
