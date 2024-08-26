import torch
from torch import nn

class PinballLoss(nn.Module):

    def __init__(self, quantile=0.10, reduction='mean'):
        super(PinballLoss, self).__init__()
        self.quantile = quantile
        assert 0 < self.quantile
        assert self.quantile < 1
        self.reduction = reduction

    def __call__(self, output, target):
        assert output.shape == target.shape
        loss = torch.zeros_like(target, dtype=torch.float)
        error = output - target
        smaller_index = error < 0
        bigger_index = 0 < error
        loss[smaller_index] = self.quantile * (abs(error)[smaller_index])
        loss[bigger_index] = (1 - self.quantile) * (abs(error)[bigger_index])

        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


def quantile_regression_loss_fn(pred, target):
  q_lo_loss = PinballLoss(quantile=0.05)
  q_hi_loss = PinballLoss(quantile=0.95)
  mse_loss = nn.L1Loss()

  loss = q_lo_loss(pred[:,0,:,:,:].squeeze(), target.squeeze()) + \
         q_hi_loss(pred[:,2,:,:,:].squeeze(), target.squeeze()) + \
         2*mse_loss(pred[:,1,:,:,:].squeeze(), target.squeeze())

  return loss
