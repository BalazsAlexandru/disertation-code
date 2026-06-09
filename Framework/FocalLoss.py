import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index


    def forward(self, inputs, targets):

        # Compute the cross-entropy loss
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, ignore_index=self.ignore_index, reduction="none")
        pt = torch.exp(-ce)

        # Compute the focal loss
        loss = (1 - pt) ** self.gamma * ce

        return loss.mean()