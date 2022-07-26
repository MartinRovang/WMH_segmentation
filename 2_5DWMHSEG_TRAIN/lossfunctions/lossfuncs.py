import torch.nn as nn
import einops
import torch

class FocalTverskyLoss_Batchmean(nn.Module):
    def __init__(self):
        super(FocalTverskyLoss_Batchmean, self).__init__()

    def forward(self, inputs, targets, smoothing=1, alpha=0.85, beta=0.15, gamma=4/3):
        #flatten label and prediction tensors
        inputs = einops.rearrange(inputs, 'b c h w -> b (c h w)')
        targets = einops.rearrange(targets, 'b c h w -> b (c h w)')

        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum(1)    
        FP = ((1-targets) * inputs).sum(1)
        FN = (targets * (1-inputs)).sum(1)

        Tversky = (TP + smoothing) / (TP + alpha*FN + beta*FP  + smoothing)
        FocalTversky = (1 - Tversky)**gamma
        FocalTversky = FocalTversky.mean()

        return FocalTversky



class FocalTverskyLoss(nn.Module):
    def __init__(self):
        super(FocalTverskyLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        #weights = torch.FloatTensor([0.3, 1, 1, 1, 1]).to('cuda:1')
        #self.crossentropy = nn.CrossEntropyLoss(weight = weights, label_smoothing=1.0)

    def forward(self, inputs, targets, smoothing=2, alpha=0.85, beta=0.15, gamma=4/3):
        #flatten label and prediction tensors

        if targets.shape[1] > 1:
            targets = targets[:, 1, :, :]
            targets = targets[:, None, :, :]
        inputs = torch.sigmoid(inputs)
        inputs = einops.rearrange(inputs, 'b c h w -> b (c h w)')
        targets = einops.rearrange(targets, 'b c h w -> b (c h w)')
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + smoothing) / (TP + alpha*FN + beta*FP  + smoothing)
        FocalTversky = (1 - Tversky)**gamma
        
        return FocalTversky


class FocalTverskyLoss_fazekas(nn.Module):
    def __init__(self):
        super(FocalTverskyLoss_fazekas, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, targets, smoothing=2, alpha=0.85, beta=0.15, gamma=4/3):

        inputs = self.softmax(inputs)

        #True Positives, False Positives & False Negatives

        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + smoothing) / (TP + alpha*FN + beta*FP  + smoothing)
        FocalTversky = (1 - Tversky)**gamma

        return FocalTversky
