import numpy as np
import matplotlib.pyplot as plt


def plot_3d_medical_image(image, title=None, cmap='gray'):
    """
    Plot a 3D medical image in three views: axial, coronal, and sagittal.

    Parameters:
    - image: 3D numpy array with shape (D, H, W)
    - title: string, optional title for the entire figure
    - cmap: string, colormap to use for the plots (default is 'gray')
    """

    D, H, W = image.shape

    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Axial view (top-down)
    ax1.imshow(image[D // 2, :, :], cmap=cmap, aspect='equal')
    ax1.set_title('Axial View')
    ax1.axis('off')

    # Coronal view (front)
    ax2.imshow(image[:, H // 2, :], cmap=cmap, aspect='equal')
    ax2.set_title('Coronal View')
    ax2.axis('off')

    # Sagittal view (side)
    ax3.imshow(image[:, :, W // 2], cmap=cmap, aspect='equal')
    ax3.set_title('Sagittal View')
    ax3.axis('off')

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()

def continuous_dipole_kernel(N, voxel_size=(1, 1, 1), B0_dir=(0, 0, 1)):
    rx = np.arange(-np.floor(N[0] / 2), np.ceil(N[0] / 2))
    ry = np.arange(-np.floor(N[1] / 2), np.ceil(N[1] / 2))
    rz = np.arange(-np.floor(N[2] / 2), np.ceil(N[2] / 2))

    kx, ky, kz = np.meshgrid(rx, ry, rz, indexing='ij')
    kx /= (np.max(np.abs(kx)) * voxel_size[0])
    ky /= (np.max(np.abs(ky)) * voxel_size[1])
    kz /= (np.max(np.abs(kz)) * voxel_size[2])

    k2 = kx ** 2 + ky ** 2 + kz ** 2
    # k2[k2 == 0] = np.finfo(np.float32).eps
    kernel = np.fft.ifftshift(
        1 / 3.0 - ((kx * B0_dir[0] + ky * B0_dir[1] + kz * B0_dir[2]) ** 2) / (k2 + np.finfo(np.float64).eps))
    kernel[0, 0, 0] = 0
    return kernel

# Example usage
if __name__ == "__main__":
    # Create a sample 3D image (you would replace this with your actual image data)
    sample_image = continuous_dipole_kernel((64,64,64))

    # Plot the image
    plot_3d_medical_image(sample_image, title="Sample 3D Medical Image")

