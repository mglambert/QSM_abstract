import numpy as np
import nibabel as nib
from utils import plot_3d_medical_image, continuous_dipole_kernel
import random
from glob import glob
from scipy import io

# path = "E:/LDM-data/LDM_100k/LDM_100k/rawdata/sub-{0:06d}/anat/sub-{0:06d}_T1w.nii.gz"
# path = "E:/LDM-data/LDM_100k/LDM_100k/rawdata/sub-000230/anat/sub-000230_T1w.nii.gz"
# img = nib.load(path.format(random.randint(0, 100000-1))).get_fdata()

paths = glob("C:/Users/mathias/Downloads/dataset/dataset/Nifti/*/*/anat/*.nii.gz")
mask_path = 'C:/Users/mathias/Downloads/dataset/dataset/Nifti/masks/masks/{}.mat'
path = random.choice(paths)

filename = path.split('\\')[-1].split('.')[0]
img = nib.load(path).get_fdata()
mask = io.loadmat(mask_path.format(filename))['mask']
img = img * mask


def split_image(img, mask, stride=32, size=64):
    # a, b, c = img.shape
    # a, b, c = a // 2, b // 2, c // 2
    # img = img[a - 96:a + 96, b - 96:b + 96, c - 96:c + 96]
    sub_images = []
    sub_masks = []
    for i in range(0, img.shape[0], stride):
        for j in range(0, img.shape[1], stride):
            for k in range(0, img.shape[2], stride):
                sub = img[i:i + size, j:j + size, k:k + size]
                sub_mask = mask[i:i + size, j:j + size, k:k + size]
                if np.sum((sub * sub_mask) != 0) / sub_mask.size > 0.6 and np.min(sub.shape)==64:
                    sub_images.append(sub)
                    sub_masks.append(sub_mask)
    return sub_images, sub_masks


print(path, img.shape)

sub_images, sub_masks = split_image(img, mask)
print(len(sub_images))
plot_3d_medical_image(sub_images[0] * sub_masks[0], '')

# %%
import matplotlib.pyplot as plt


def tranform_uniform(img, mask, inverted=False):
    im_aux = img[mask == 1]
    j = np.argsort(im_aux)
    r = np.linspace(-0.1, 0.1, j.shape[0]) * (1 if not inverted else -1)
    im_aux[j] = r
    img2 = img.copy()
    img2[mask == 1] = im_aux
    return img2


def tranform_normal(img, mask, inverted=False):
    im_aux = img[mask == 1]
    j = np.argsort(im_aux)
    r = np.random.normal(0, 0.0279, size=j.shape[0])
    # r = np.sign(r) * np.abs(r)**1.01
    r = np.sort(r)
    if inverted:
        r = r[::-1]
    im_aux[j] = r
    img2 = img.copy()
    img2[mask == 1] = im_aux
    return img2


def tranform_actual(img, mask, inverted=False):
    im_aux = img[mask == 1]
    j = np.argsort(im_aux)
    r = im_aux.copy()
    r -= np.mean(im_aux)
    r = np.sort(r)
    if inverted:
        r = r[::-1]
    im_aux[j] = r
    img2 = img.copy()
    img2[mask == 1] = im_aux
    return img2


img2 = tranform_normal(img, mask, inverted=True)
plot_3d_medical_image(img, 'img')
plot_3d_medical_image(img2, 'img2')

plt.figure()
plt.hist(img[mask == 1], 256)
plt.title('Histograma 1')
plt.show()

plt.figure()
plt.hist(img2[mask == 1], 256)
plt.title('Histograma 2')
plt.show()


# %%

def normalize(img, mask):
    vals = np.abs(img[mask == 1])
    thr = np.quantile(vals, 0.989)
    img = np.random.uniform(0.1, 0.2) * img / thr
    return img


img2 = tranform_actual(img, mask, inverted=True)
img2 = normalize(img2, mask)
plot_3d_medical_image(img, 'img')
plot_3d_medical_image(img2, 'img2')

plt.figure()
plt.hist(img[mask == 1], 256)
plt.title('Histograma 1')
plt.show()

plt.figure()
plt.hist(img2[mask == 1], 256)
plt.title('Histograma 2')
plt.show()

# %%

from tqdm import tqdm

tranform_uniform_inv = lambda img, mask: tranform_uniform(img, mask, inverted=True)
tranform_normal_inv = lambda img, mask: tranform_normal(img, mask, inverted=True)
tranform_actual_inv = lambda img, mask: tranform_actual(img, mask, inverted=True)

transforms = [
    tranform_uniform,
    tranform_uniform_inv,
    tranform_normal,
    tranform_normal_inv,
    tranform_actual,
    tranform_actual_inv
]

k = 50_000
save_path = "./data"
N = [64, 64, 64]
K = continuous_dipole_kernel(N)


def save(chi, mask, idx):
    phase = np.real(np.fft.ifftn(K * np.fft.fftn(chi)))
    with open(f'{save_path}/{idx}.npz', 'wb') as file:
        np.savez(file, chi=chi, mask=mask, phase=phase)


for path in tqdm(paths):
    filename = path.split('\\')[-1].split('.')[0]
    img = nib.load(path).get_fdata()
    mask = io.loadmat(mask_path.format(filename))['mask']
    img = img * mask
    for tr in transforms:
        img2 = tr(img, mask)
        img2 = normalize(img2, mask)
        sub_images, sub_masks = split_image(img2, mask)
        for i in range(len(sub_images)):
            save(sub_images[i], sub_masks[i], k)
            k += 1
print(k)

# %%

data = io.loadmat('data/Cosmos_SNR100.mat')
chi = data['chi_cosmos'] * data['mask_use']
chi_mask = data['mask_use']
vals = np.abs(chi[chi_mask == 1])

plt.figure()
plt.hist(vals, 256)
plt.show()
