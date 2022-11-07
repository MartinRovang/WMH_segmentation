
import numpy as np
import einops


class FB_score:
    """Calculate the FB metric"""
    def __init__(self, beta = 1):
        self.beta = beta
        self.batch_record = []
        self.mean_score = 0

    def __call__(self, y_pred, y):
        """Calculates the F2 score and returns the value"""
        
        y_pred = y_pred.contiguous().view(-1).cpu().numpy()
        y = y.contiguous().view(-1).cpu().numpy().astype(int)
        # y_pred[y_pred >= 0.5] = 1
        # y_pred[y_pred < 0.5] = 0
        # y_pred = y_pred.astype(int)
        TP = (y_pred * y).sum()    
        FP = ((1-y) * y_pred).sum()
        FN = (y * (1-y_pred)).sum()
        if (TP + FN + FP) == 0:
            F2 = np.nan
        else:
            F2 = ((1 + self.beta**2)*TP)/((1 + self.beta**2)*TP + self.beta**2*FN + FP)
        self.batch_record.append(F2)
    
    def aggregate(self):
        self.mean_score = np.nanmean(self.batch_record)
        return self.mean_score
    
    def reset(self):
        self.batch_record = []
        self.mean_score = 0



class dice_score:
    """Calculate the FB metric"""
    def __init__(self):
        self.batch_record = []
        self.mean_score = 0

    def __call__(self, y_pred, y):
        """Calculates the F2 score and returns the value"""

        y_pred = y_pred.contiguous().cpu().numpy()
        y = y.contiguous().cpu().numpy().astype(int)

        y = einops.rearrange(y, 'b c h w -> b (c h w)')
        y_pred = einops.rearrange(y_pred, 'b c h w -> b (c h w)')
        
        # limit = y.sum(axis = 1) > 4
        # y = y[limit, :]
        # y_pred = y_pred[limit, :]

        # y = y.flatten()
        # y_pred = y_pred.flatten()

        # y_pred[y_pred >= 0.5] = 1
        # y_pred[y_pred < 0.5] = 0
        # y_pred = y_pred.astype(int)
        TP = (y_pred * y).sum(1)    
        FP = ((1-y) * y_pred).sum(1)
        FN = (y * (1-y_pred)).sum(1)
        
        if len(TP) < 1:
            dice = [1]
        else:
            dice = 2*TP/(2*TP + FP + FN ) # Dice score / F1
        self.batch_record.append(np.nanmean(dice))
    
    def aggregate(self):
        self.mean_score = np.nanmean(self.batch_record)
        return self.mean_score
    
    def reset(self):
        self.batch_record = []
        self.mean_score = 0
