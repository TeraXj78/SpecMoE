import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    r2_score, mean_squared_error
)
import numpy as np  


class Metrics_Computation:     
    def __init__(self, output, target, num_classes=3):
        self.output = output
        self.target = target
        self.pred = torch.argmax(output, dim=1)
        self.num_classes = num_classes

        assert self.pred.shape[0] == len(target), "Prediction and target size mismatch"

    def balanced_accuracy(self):
        return balanced_accuracy_score(self.target.cpu().numpy(), self.pred.cpu().numpy())
    
    def accuracy(self):
        correct = torch.sum(self.pred == self.target).item()
        return correct / len(self.pred)

    def f1_score_macro(self):
        return f1_score(
            self.target.cpu().numpy(), self.pred.cpu().numpy(), average='weighted'
        )

    def precision(self):
        return precision_score(
            self.target.cpu().numpy(), self.pred.cpu().numpy(), average='weighted'
        )

    def recall(self):
        return recall_score(
            self.target.cpu().numpy(), self.pred.cpu().numpy(), average='weighted'
        )

    def auroc(self):
        # One-hot encode targets
        y_true = torch.nn.functional.one_hot(self.target, num_classes=self.num_classes).cpu().numpy()
        y_score = torch.softmax(self.output, dim=1).detach().cpu().numpy()

        try:
            return roc_auc_score(y_true, y_score, average='weighted', multi_class='ovr')
        except ValueError:
            return np.nan  # Return NaN if AUROC can't be computed (e.g., single class present)

    def auprc(self):
        # One-hot encode targets
        y_true = torch.nn.functional.one_hot(self.target, num_classes=self.num_classes).cpu().numpy()
        y_score = torch.softmax(self.output, dim=1).detach().cpu().numpy()
        try:
            return average_precision_score(y_true, y_score, average='weighted')
        except ValueError:
            return np.nan
        
    



class RegressionMetrics:
    def __init__(self, preds, truths):
        # preds, truths: torch tensors
        self.preds = preds.detach().view(-1).cpu().numpy()
        self.truths = truths.detach().view(-1).cpu().numpy()
        assert self.preds.shape[0] == self.truths.shape[0], "Pred/target size mismatch"

    def corrcoef(self):
        # handle constant vectors safely
        if np.std(self.preds) == 0 or np.std(self.truths) == 0:
            return np.nan
        return np.corrcoef(self.truths, self.preds)[0, 1]

    def r2(self):
        return r2_score(self.truths, self.preds)

    def rmse(self):
        return mean_squared_error(self.truths, self.preds) ** 0.5
    







