import numpy as np
import random


def create_3d_image(shape, fill_value=0):
    """Create a 3D image array filled with a specific value."""
    return np.full(shape, fill_value, dtype=np.float32)


def sphere(shape, center, radius, value=1):
    """Create a 3D sphere."""
    image = create_3d_image(shape)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist_from_center = np.sqrt((x - center[2]) ** 2 + (y - center[1]) ** 2 + (z - center[0]) ** 2)
    image[dist_from_center <= radius] = value
    return image


def rectangular_prism(shape, start, end, value=1):
    """Create a 3D rectangular prism."""
    image = create_3d_image(shape)
    image[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = value
    return image


def torus(shape, center, major_radius, minor_radius, value=1):
    """Create a 3D torus."""
    image = create_3d_image(shape)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist_from_center = np.sqrt((x - center[2]) ** 2 + (y - center[1]) ** 2)
    dist_from_ring = np.abs(dist_from_center - major_radius)
    image[(dist_from_ring <= minor_radius) & (np.abs(z - center[0]) <= minor_radius)] = value
    return image


def cone(shape, apex, base_center, base_radius, value=1):
    """Create a 3D cone."""
    image = create_3d_image(shape)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    height = apex[0] - base_center[0]
    dist_from_base_center = np.sqrt((x - base_center[2]) ** 2 + (y - base_center[1]) ** 2)
    radius_at_height = base_radius * (1 - (z - base_center[0]) / (height+1e-5))
    image[(dist_from_base_center <= radius_at_height) & (z >= base_center[0]) & (z <= apex[0])] = value
    return image


def create_composite_3d_image(shape, num_objects=5):
    """
    Create a composite 3D image by combining multiple 3D objects with random weights.

    :param shape: Tuple of (depth, height, width) for the 3D image
    :param num_objects: Number of objects to include in the composite image
    :return: Numpy array representing the composite 3D image
    """
    weight = random.uniform(-1, 1)
    composite_image = create_3d_image(shape, fill_value=weight)
    mask = create_3d_image(shape)

    object_functions = [sphere, sphere, sphere, rectangular_prism, torus, torus, cone]

    for _ in range(num_objects):
        # Choose a random object function
        obj_func = random.choice(object_functions)

        # Generate random parameters for the chosen object
        center = tuple(random.randint(0, dim - 1) for dim in shape)

        if obj_func == sphere:
            radius = random.randint(5, min(shape) // 4)
            obj = obj_func(shape, center, radius)
        elif obj_func == rectangular_prism:
            start = tuple(random.randint(0, dim - 1) for dim in shape)
            end = tuple(random.randint(s, dim - 1) for s, dim in zip(start, shape))
            obj = obj_func(shape, start, end)
        elif obj_func == torus:
            major_radius = random.randint(10, min(shape[1:]) // 3)
            minor_radius = random.randint(2, major_radius // 2)
            obj = obj_func(shape, center, major_radius, minor_radius)
        elif obj_func == cone:
            apex = tuple(random.randint(0, dim - 1) for dim in shape)
            base_radius = random.randint(5, min(shape[1:]) // 4)
            obj = obj_func(shape, apex, center, base_radius)

        # Generate a random weight (positive or negative)
        weight = random.uniform(-1, 1)

        # Add the weighted object to the composite image
        composite_image += weight * obj
        mask += obj != 0

    val = np.quantile(np.abs(composite_image), 0.95)
    composite_image = composite_image * 0.1 / (val + 0.001)

    mask = mask != 0
    return composite_image * mask, mask


# Example usage
if __name__ == "__main__":
    from utils import plot_3d_medical_image, continuous_dipole_kernel

    #
    # shape = (64, 64, 64)
    #
    # # Create a sphere
    # composite_image, mask = create_composite_3d_image(shape, num_objects=100)
    #
    # K = continuous_dipole_kernel(shape)
    #
    # phase = np.real(np.fft.ifftn(K * np.fft.fftn(composite_image)))
    #
    # plot_3d_medical_image(composite_image, title="3D composite_image")
    # plot_3d_medical_image(phase, title="3D composite_image Phase")
    # plot_3d_medical_image(mask, title="3D composite_image")

    from threading import Thread
    import os
    from tqdm import tqdm

    threads_number = 2

    path = "./data"
    if not os.path.exists(path):
        os.mkdir(path)

    N = [64, 64, 64]
    K = continuous_dipole_kernel(N)


    def gen_and_save(idx):
        chi, mask = create_composite_3d_image(N, num_objects=100)
        phase = np.real(np.fft.ifftn(K * np.fft.fftn(chi)))
        with open(f'{path}/{idx}.npz', 'wb') as file:
            np.savez(file, chi=chi, mask=mask, phase=phase)


    threads = []
    for idx in tqdm(range(10)):
        threads.append(Thread(target=gen_and_save, args=(idx,)))
        threads[-1].start()
        if idx % threads_number == 0:
            for worker in threads:
                worker.join()
            threads = []
    for worker in threads:
        worker.join()

