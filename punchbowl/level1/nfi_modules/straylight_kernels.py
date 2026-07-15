from typing import Callable
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor

import numba
import numpy as np
import scipy.ndimage


def kernel_smoothing_matrix(angles_rev, smooth_rad=0.1):
    """
    Build smoothing matrix for kernel.

    Parameters:
    -----------
    angles_rev: array_like
        1D array of angle values (in radians) at which the data is sampled.
        Must be uniformly spaced; only the spacing between the first two elements (`angles_rev[1]-angles_rev[0]`)
        is used to determine the grid resolution.

    smooth_rad: float, optional, default = 0.1
        Half-width of the smoothing kernel, in radians.
        Determines how many neighboring grid points contribute to the smoothed value.

    Returns:
    --------
    output: np.ndarray
        Circulant smoothing matrix
    """
    n_angles = len(angles_rev)  # [int]
    angle_diff = angles_rev[1] - angles_rev[0]  # [num]
    output_matrix = np.zeros([n_angles, n_angles])  # [array]
    n_radius_steps = np.floor(smooth_rad / angle_diff).astype(np.int32)  # [int]
    na = 2 * n_radius_steps + 1  # kernel width? #[int]
    smoothing_kernel_angles = angle_diff * (np.arange(na) - n_radius_steps) / smooth_rad  # [array]

    # Create array pf smoothing kernel
    smoothing_kernel = np.zeros(n_angles)  # [1D array]
    smoothing_kernel[0:na] = np.cos(0.5 * np.pi * smoothing_kernel_angles)

    # Build circulant smoothing matrix
    for i in range(n_angles):
        output_matrix[i] = np.roll(smoothing_kernel, i - n_radius_steps)

    return output_matrix


def generate_kernel(
    theta: float,
    radial_size: float = 660,
    aspect_ratio: float = 1,
    right_intensity: float = 1,
    bottom_intensity: float = 1,
    elon_abs: float | None = None,
    elon_offset: float = 0,
    blur: float = 0,
    image_size: int = 2048,
    oversamp: int = 3,
    cx: float | None = None,
    cy: float | None = None,
    r_profile: Callable = None,
    dtype: str = "float32",
):
    """
    Generate an image of a kernel at a specific position.

    This single kernel is meant to be the out-of-focus image of a single point on the occulter ring. A bunch of
    options are provided for adjusting how that kernel looks.

    Parameters
    ----------
    theta : float
        The angular position of the center of the kernel, CCW from the +x axis
    radial_size : float
        The size of the kernel in pixels. ("Radial" here means "radial out from image-center".)
    aspect_ratio : float
        The aspect ratio of the kernel. The size of the kernel perpendicular to the radial direction is
        aspect_ratio * radial_size. ("Radial" here means "radial out from image-center".)
    right_intensity : float
        Used to create an intensity gradient. For a kernel at theta=0, the intensity will vary linearly from 1 at the
        left edge to this value at the right edge. For theta != 0, the gradient direction rotates with the kernel.
    bottom_intensity : float
        Used to create an intensity gradient. For a kernel at theta=0, the intensity will vary linearly from 1 at the
        top edge to this value at the bottom edge. For theta != 0, the gradient direction rotates with the kernel.
    elon_abs : float | None
        Set the distance from the center of the image to the center of the kernel. This is measured in pixels, but it is
        otherwise the elongation angle of the kernel relative to the image center.
    elon_offset : float
        Offset the distance from the center of the image to the center of the kernel. This is measured in pixels, but it
        is otherwise the elongation angle of the kernel relative to the image center. Not used if elon_abs is set.
    blur : float
        If non-zero, a Gaussian blur is applied to the final kernel image. This parameter sets the standard deviation of
        the Gaussian in pixels.
    image_size : int
        The size of the (square) output image, in pixels.
    oversamp : int
        If set, the kernel profile is computed on an over-sampled grid and downsampled to the final output size. This
        allows pixels at the edge of the kernel to have values between 0 and 1, rather
        than having a sharp drop from 1 to 0 at the kernel edge (assuming a gradient-free kernel). This parameter sets
        the amount of oversampling in each dimension.
    cx, cy : int
        The x and y coordinate around which the kernel is rotated (by the theta parameter). This coordinate then becomes
        the center of the disk of kernels once kernels are computed for all theta values.
    r_profile : Callable
        Allows an arbitrary radial intensity profile to be provided. The callable should receive a 2D numpy array
        indicating the radial component (i.e. radial-out from image-center) of the pixel's location relative to the
        kernel center, expressed as a fraction of the kernel radius, and should return an array of the same shape
        providing a scale factor for each pixel.
    dtype : str
        The dtype to use for the output array

    Returns
    -------
    kernel : np.ndarray
        The computed kernel

    """
    # Radial position of the inner edge of the donut (i.e. the dynamic stray light pattern)
    r_to_inner_edge = 181.27180103 + 5
    # Radial position of the center of this kernel
    r_to_center = 660 / 2
    # Center of the kernel---the default values are the center of the occulted region, which isn't necessarily the
    # center of the donut of stray light
    if cx is None:
        cx = (1014.50355056 - 1) * image_size / 2048
    if cy is None:
        cy = (1037.37339562 - 1) * image_size / 2048

    r_to_inner_edge *= image_size / 2048
    r_to_center *= image_size / 2048

    if elon_abs is None:
        r_to_center = r_to_inner_edge + r_to_center + elon_offset
    else:
        r_to_center = elon_abs

    # The kernel can be oversampled for anti-aliasing
    if oversamp == 1:
        coords = np.arange(0, image_size, dtype=float)
    elif oversamp % 2 == 1:
        coords = np.linspace(-1 / oversamp, image_size - 1 / oversamp, image_size * oversamp, endpoint=False)
    else:
        raise ValueError("Oversamp must be odd")

    # x and y of each pixel relative to the defined center of the kernel
    xs = coords - (cx + r_to_center * np.cos(theta))
    ys = coords - (cy + r_to_center * np.sin(theta))
    xs, ys = np.meshgrid(xs, ys, sparse=True, copy=False)

    azimuthal_diameter = radial_size * aspect_ratio

    b = azimuthal_diameter / 2
    a = radial_size / 2

    # Theta defines where in the donut we're generating a kernel for, and it also defines how the kernel itself is
    # rotated (since we're rotating the kernel around the image center, not translating it in a circular path). These
    # arrays will hold coordinates in a rotated frame, where x increases along the line from image center to kernel
    # center, and y is perpendicular to that.
    angled_x = xs * np.cos(theta) + ys * np.sin(theta)
    angled_y = xs * np.sin(-theta) + ys * np.cos(-theta)

    # Equation for an ellipse
    kernel = angled_x**2 / a**2 + angled_y**2 / b**2 < 1
    # Identify a bounding box around the non-zero values, to cut down on the work we do later
    rows = np.any(kernel, axis=1)
    cols = np.any(kernel, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    s = np.s_[rmin : rmax + 1, cmin : cmax + 1]
    kernel = kernel.astype(float)

    # Impose linear gradients
    if right_intensity != 1:
        slope_x = (1 - right_intensity) / azimuthal_diameter
        kernel[s] *= 1 - slope_x * (angled_x[s] + azimuthal_diameter)

    if bottom_intensity != 1:
        slope_y = (1 - bottom_intensity) / radial_size
        kernel[s] *= 1 - slope_y * (angled_y[s] + radial_size)

    # Impose an arbitrary gradient
    if r_profile is not None:
        kernel[s] *= r_profile(angled_x[s] / (radial_size / 2))

    # Block sum to target resolution
    if oversamp > 1:
        kernel = kernel.reshape((image_size, oversamp, image_size, oversamp)).sum(axis=(1, 3))

    if blur > 0:
        kernel = scipy.ndimage.gaussian_filter(kernel, blur)

    return kernel.astype(dtype)


def _generate_kernel_caller(theta, args, kwargs):
    return generate_kernel(theta, *args, **kwargs)


def generate_kernels(kernel_angles, *args, n_threads=5, **kwargs):
    """
    Generate a full set of kernels in parallel.

    All arguments are passed through to `generate_kernel`.

    Parameters
    ----------
    kernel_angles : list | np.ndarray
        The theta values at which to compute kernels
    args : tuple
        Arguments to pass to generate_kernel
    n_threads : int | None
        The number of threads to use for parallel processing
    kwargs : dict
        Keyword arguments to pass to generate_kernel

    Returns
    -------
    kernels : np.ndarray
        The generated kernels
    """
    with ThreadPoolExecutor(n_threads) as p:
        kernels = p.map(_generate_kernel_caller, kernel_angles, repeat(args), repeat(kwargs))
    return np.stack(list(kernels))


@numba.njit(parallel=True)
def make_model(
    kernels: np.ndarray, intensity: np.ndarray, rmin: int = 0, rmax: int = -1, cmin: int = 0, cmax: int = -1
):
    """
    Make a forward model, given a set of kernels and an intensity for each one.

    This is multiplying each kernel by its intensity and summing, but it's done in numba and in parallel for speed.
    Also for speed, if the kernels don't reach to the edge of the image, a bounding box can be set to sum within,
    and areas outside the box are not summed.

    Parameters
    ----------
    kernels : np.ndarray
        The kernels
    intensity : np.ndarray
        The intensity for each kernel. Should match the size of the first dimension of `kernels`.
    rmin : int
        The first row to sum. Rows before this will be zero in the output image.
    rmax : int
        The last row to sum. Rows after this will be zero in the output image.
    cmin : int
        The first column to sum. Columns before this will be zero in the output image.
    cmax : int
        The last column to sum. Columns after this will be zero in the output image.

    Returns
    -------
    image : np.ndarray
        The forwared-modeled image.
    """
    #
    if rmax < 0:
        rmax = kernels.shape[1]
    if cmax < 0:
        cmax = kernels.shape[2]
    output = np.empty(kernels.shape[1:])
    for i in range(rmin):
        output[i] = 0
    for i in range(cmin):
        output[:, i] = 0
    for i in range(rmax, kernels.shape[1]):
        output[i] = 0
    for i in range(cmax, kernels.shape[2]):
        output[:, i] = 0
    for i in numba.prange(rmin, rmax):
        for j in range(cmin, cmax):
            output[i, j] = np.sum(kernels[:, i, j] * intensity)
    return output
