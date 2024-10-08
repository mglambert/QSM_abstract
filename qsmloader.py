from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import time


class QSMLoader(Dataset):
    def __init__(self, indices, root="./data/", train=True):
        self.root = root
        self.indices = indices
        self.to_tensor = lambda x: torch.Tensor(x)
        self.train = train

    def __getitem__(self, idx):
        nombre_archivo = f'{self.root}{self.indices[idx]}.npz'

        phase_sr = self.to_tensor(np.load(nombre_archivo)['phase']).type(torch.float32)
        phase_sr = torch.unsqueeze(phase_sr, 0)

        chi = self.to_tensor(np.load(nombre_archivo)['chi']).type(torch.float32)
        chi = torch.unsqueeze(chi, 0)

        mask = self.to_tensor(np.load(nombre_archivo)['mask']).type(torch.float32)
        mask = torch.unsqueeze(mask, 0)

        mag = chi - chi.min()
        mag = mag / mag.max()
        mag = mag * mask

        scale = torch.pi / (2*torch.max(torch.abs(phase_sr)))
        signal = mag * torch.exp(1j * phase_sr * scale)
        # _rr = np.random.rand()

        snr = torch.randint(95, 105, (1,))

        signal = signal + ((1. / snr) * (torch.randn(signal.shape) + 1j * torch.randn(signal.shape)))

        phase = torch.angle(signal).type(torch.float32) / scale
        phase = phase * mask

        return phase, chi, mask, phase_sr

    def __len__(self):
        return len(self.indices)


if __name__ == '__main__':
    from utils import plot_3d_medical_image
    import numpy as np

    idx = list(range(1))

    ds_train = QSMLoader(idx)

    loader = DataLoader(ds_train, batch_size=1, shuffle=True)

    for phase, chi, mask, phase_sr in loader:
        plot_3d_medical_image(np.asarray(phase[0, 0]))
        plot_3d_medical_image(chi[0, 0])
        plot_3d_medical_image(mask[0, 0])
        plot_3d_medical_image(phase_sr[0, 0])
        break
