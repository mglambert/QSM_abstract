import torch
import torch.nn as nn
from tqdm import tqdm


def run_epoch(model, dataloader, criterion, metric, device, optimizer=None, epoch=0, total_epoch=0, phase='train'):
    if phase == 'train':
        torch.cuda.empty_cache()
        model.train()
    elif phase == 'val' or 'test':
        torch.cuda.empty_cache()
        model.eval()

    cum_loss = 0.0
    cum_metric_value = 0.0

    with tqdm(dataloader, unit='batch', position=0, leave=True) as tepoch:
        for n_batch, (phi, gt, mask) in enumerate(tepoch, start=1):
            if phase == 'train':
                torch.cuda.empty_cache()
                model.train()
                tepoch.set_description(f"Epoch {epoch}/{total_epoch}")
                optimizer.zero_grad()

            elif phase == 'val':
                tepoch.set_description(f"\tVal. ")

            phi = (phi*mask).to(device)

            result = model(phi)
            result = result * mask.to(device)

            gt = gt.to(device)

            torch.cuda.empty_cache()

            loss = criterion(result, gt)

            metric_value = metric(result, gt).item()
            cum_loss += loss.item()
            cum_metric_value += metric_value

            if phase == 'train':
                loss.backward()
                optimizer.step()

            current_cum_loss = cum_loss / n_batch
            current_metric_value = cum_metric_value / n_batch
            tepoch.set_postfix(Loss=current_cum_loss,
                               NRMSE=current_metric_value)

    epoch_loss = float(cum_loss / n_batch)
    mse_epoch = float(cum_metric_value / n_batch)
    return epoch_loss, mse_epoch



def run_training(model, data_train, data_val, optimizer, criterion, metric, device, n_epochs, scheduler=None):

    history = {
        'train': {'loss': [], 'metric': []},
        'val': {'loss': [], 'metric': []},
    }

    for epoch in range(1, n_epochs + 1):
        epoch_loss, rmse = run_epoch(model, data_train, criterion, metric, device, optimizer=optimizer, epoch=epoch,
                                     total_epoch=n_epochs, phase='train')

        history['train']['loss'].append(epoch_loss)
        history['train']['metric'].append(rmse)

        if data_val is not None:
            val_loss, rmse = run_epoch(model, data_val, criterion, metric, device, phase='val', experiment=experiment)
            history['val']['loss'].append(val_loss)
            history['val']['metric'].append(rmse)

        if scheduler is not None:
            scheduler.step()
    return history

if __name__ == '__main__':

    from red import UNet3D
    from qsmloader import QSMLoader
    from torch.utils.data import DataLoader
    from utils import plot_3d_medical_image
    import matplotlib.pyplot as plt
    from loss import quantile_regression_loss_fn


    # torch.cuda.init()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hyper_params = {
        "learning_rate": 1e-4,
        "epochs": 2,
        "train_batch_size": 1,
        "val_batch_size": 1,
        "exponential_lr_param": 0.99999999999,
        "weight_decay": 1e-20,
    }

    ds_train = QSMLoader(list(range(1)), train=True)
    train_dl = DataLoader(ds_train, batch_size=hyper_params['train_batch_size'], shuffle=True)

    model = UNet3D(in_channels=1, out_channels=3)
    model = model.to(device)

    metric = nn.MSELoss()
    criterion = quantile_regression_loss_fn

    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params['learning_rate'],
                                 weight_decay=hyper_params['weight_decay']
                                 )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, hyper_params['exponential_lr_param'])

    history = run_training(model, train_dl, None, optimizer, criterion, metric, device, hyper_params['epochs'],
                            scheduler=scheduler)

    plt.figure()
    plt.plot(history['train']['loss'], label='train')
    plt.legend()
    plt.show()

    for phase, gt, mask in train_dl:
        plot_3d_medical_image(gt[0, 0], 'GT', )
        with torch.inference_mode():
            out = model((phase).to(device)).cpu()
        plot_3d_medical_image(out[0, 1], 'inference')
        break