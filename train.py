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
                torch.cuda.empty_cache()
                model.eval()
                tepoch.set_description(f"\tVal. ")

            phi = (phi * mask).to(device)

            result = model(phi)
            result = result * mask.to(device)

            gt = gt.to(device)

            torch.cuda.empty_cache()

            loss = criterion(result, gt)

            metric_value = metric(result[:, 1:2], gt).item()
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
            val_loss, rmse = run_epoch(model, data_val, criterion, metric, device, phase='val')
            history['val']['loss'].append(val_loss)
            history['val']['metric'].append(rmse)

        torch.save(model.state_dict(), f"saved/model_epoch_{epoch}.pth")

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

    torch.cuda.init()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hyper_params = {
        "learning_rate": 1e-4,
        "epochs": 50,
        "train_batch_size": 16,
        "val_batch_size": 10,
        "exponential_lr_param": 0.95,
        "weight_decay": 1e-6,
    }

    ds_train = QSMLoader(list(range(26_900)), train=True)
    train_dl = DataLoader(ds_train, batch_size=hyper_params['train_batch_size'], shuffle=True)
    ds_val = QSMLoader(list(range(26_900, 27_000)), train=False)
    val_dl = DataLoader(ds_val, batch_size=hyper_params['val_batch_size'], shuffle=True)
    val_dl = None

    model = UNet3D(in_channels=1, out_channels=3)
    model.load_state_dict(torch.load('model_epoch_10.pth'))
    model = model.to(device)

    metric = nn.MSELoss()
    criterion = quantile_regression_loss_fn

    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params['learning_rate'],
                                 weight_decay=hyper_params['weight_decay']
                                 )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, hyper_params['exponential_lr_param'])

    history = run_training(model, train_dl, val_dl, optimizer, criterion, metric, device, hyper_params['epochs'],
                           scheduler=scheduler)

    plt.figure()
    plt.plot(history['train']['loss'], label='train')
    if val_dl:
        plt.plot(history['val']['loss'], label='val')
    plt.legend()
    plt.show()

    for phase, gt, mask in train_dl:
        plot_3d_medical_image(gt[0, 0], 'GT', )
        with torch.inference_mode():
            out = model((phase).to(device)).cpu() * mask
        plot_3d_medical_image(out[0, 0], 'inference u')
        plot_3d_medical_image(out[0, 1], 'inference')
        plot_3d_medical_image(out[0, 2], 'inference l')
        break

    if val_dl:
        for phase, gt, mask in val_dl:
            plot_3d_medical_image(gt[0, 0], 'GT', )
            with torch.inference_mode():
                out = model((phase).to(device)).cpu() * mask
            plot_3d_medical_image(out[0, 1], 'inference')
            plot_3d_medical_image(out[0, 0], 'inference  u')
            plot_3d_medical_image(out[0, 2], 'inference l')
            break

    if False:
        from scipy import io
        from utils import continuous_dipole_kernel
        import numpy as np

        data = io.loadmat('data/Cosmos_SNR100.mat')
        chi = data['chi_cosmos'] * data['mask_use']
        mask = data['mask_use']

        phase = np.real(np.fft.ifftn(np.fft.fftn(chi) * continuous_dipole_kernel(chi.shape))) * mask

        mag = chi - chi.min()
        mag = mag / mag.max()
        mag = mag * mask

        scale = np.pi / (2 * np.max(np.abs(phase)))
        signal = mag * np.exp(1j * phase * scale)

        snr = np.random.randint(95, 105, (1,))

        signal = signal + ((1. / snr) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)))

        phase = np.angle(signal).astype(np.float32) / scale
        phase = phase * mask
        phase[80, 80, 80] = 2

        # model.load_state_dict(torch.load('model_epoch_10.pth'))
        # model = model.to(device)

        p = torch.Tensor(phase).unsqueeze(0).unsqueeze(0)

        with torch.inference_mode():
            out = model((p).to(device)).cpu() * mask
        plot_3d_medical_image(out[0, 1], 'inference', rango=(-0.1, 0.1))
        plot_3d_medical_image(out[0, 0], 'inference  u', rango=(-0.1, 0.1))
        plot_3d_medical_image(out[0, 2], 'inference l', rango=(-0.1, 0.1))

        gt = data['chi_cosmos'] * data['mask_use']
        pred = out[0, 1].numpy() * data['mask_use']
        m = data['mask_use'] == 1
        # pred[m] -= pred[m].mean()
        # pred = np.clip(pred, -0.264, 0.391)
        print(100 * np.linalg.norm(pred.ravel() - gt.ravel()) / np.linalg.norm(gt.ravel()))

        plot_3d_medical_image(gt, 'gt', rango=(-0.1, 0.1))
        plot_3d_medical_image(pred, 'inference', rango=(-0.1, 0.1))
        plot_3d_medical_image(gt - pred, 'diff', rango=(-0.1, 0.1))

        pred_u = out[0, 0].numpy() * data['mask_use']
        pred_l = out[0, 2].numpy() * data['mask_use']
        pred_lu = pred_l - pred_u
        print(pred_lu[m].min(), pred_lu[m].mean(), pred_lu[m].max())
        plot_3d_medical_image(pred_lu, 'uncertainty', rango=(0, 0.3))

        plt.figure()
        plt.hist(gt.ravel(), bins=100)
        plt.show()
        plt.figure()
        plt.hist(pred.ravel(), bins=100)
        plt.show()

        plt.figure()
        plt.hist(gt[m], bins=100, label='gt')
        plt.hist(pred[m], bins=100, label='pred')
        plt.legend()
        plt.show()

        plt.figure()
        plt.hist(pred[m], bins=100, label='pred')
        plt.hist(gt[m], bins=100, label='gt')
        plt.legend()
        plt.show()

        print(gt[m].min(), gt[m].mean(), gt[m].max())
        print(pred[m].min(), pred[m].mean(), pred[m].max())

        for ii in range(1, 15):
            model.load_state_dict(torch.load(f'model_epoch_{ii}.pth'))
            model = model.to(device)
            with torch.inference_mode():
                out = model((p).to(device)).cpu() * mask

            gt = data['chi_cosmos'] * data['mask_use']
            pred = out[0, 1].numpy() * data['mask_use']
            m = data['mask_use'] == 1
            pred[m] -= pred[m].mean()
            pred = np.clip(pred, -0.264, 0.391)
            print(ii, 100 * np.linalg.norm(pred.ravel() - gt.ravel()) / np.linalg.norm(gt.ravel()))



from scipy.ndimage import affine_transform
def rotation_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

def rotate_image(image, theta):
    # Matriz de rotación
    rot_mat = rotation_matrix(theta)

    # Calcular el centro de la imagen
    center = np.array(image.shape) / 2

    # Trasladar el origen al centro
    offset = center - np.dot(rot_mat, center)

    # Aplicar la transformación afín
    rotated_image = affine_transform(
        image, rot_mat, offset=offset, order=1  # 'order=1' aplica una interpolación bilineal
    )
    return rotated_image


def plot_3d_medical_image(image, title=None, cmap='gray', rango=None):
    """
    Plot a 3D medical image in three views: axial, coronal, and sagittal.

    Parameters:
    - image: 3D numpy array with shape (D, H, W)
    - title: string, optional title for the entire figure
    - cmap: string, colormap to use for the plots (default is 'gray')
    """
    if rango is None:
        rango = (image.min(), image.max())

    D, H, W = image.shape

    # Create a figure with three subplots
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    im1 = image[D // 2, :, :]
    im1 = rotate_image(im1, np.radians(-90))

    im2 = image[:, H // 2, :]
    im2 = rotate_image(im2, np.radians(-90))

    im3 = image[:, :, W // 2]
    im3 = rotate_image(im3, np.radians(90))

    im = np.concatenate([im1, im2, im3], axis=1)

    # Sagittal view (side)
    ax.imshow(im, cmap=cmap, aspect='equal', vmin=rango[0], vmax=rango[1])
    # ax3.set_title('Sagittal View')
    ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=32)

    plt.tight_layout()
    plt.show()
