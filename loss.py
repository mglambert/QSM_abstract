import torch
from torch import nn
from utils import continuous_dipole_kernel


class QSM_Loss(nn.Module):

    def __init__(self):
        super(QSM_Loss, self).__init__()
        self.l2 = nn.MSELoss()
        self.K = torch.Tensor(continuous_dipole_kernel((64, 64, 64))).unsqueeze(0).to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    def __call__(self, output, target):
        assert output.shape == target.shape
        phase = torch.real(torch.fft.ifftn(torch.fft.fftn(output) * self.K))
        return self.l2(phase, target)


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


def quantile_regression_loss_fn(pred, target, target_phase):
    q_lo_loss = PinballLoss(quantile=0.05)
    q_hi_loss = PinballLoss(quantile=0.95)
    mse_loss = nn.L1Loss()
    qms_loss = QSM_Loss()
    loss = q_lo_loss(pred[:, 0, :, :, :].squeeze(), target.squeeze()) + \
           q_hi_loss(pred[:, 2, :, :, :].squeeze(), target.squeeze()) + \
           2 * mse_loss(pred[:, 1, :, :, :].squeeze(), target.squeeze()) + \
           3 * qms_loss(pred[:, 1, :, :, :].squeeze(), target_phase.squeeze())

    return loss
