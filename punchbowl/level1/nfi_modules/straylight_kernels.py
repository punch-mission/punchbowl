import numpy as np


def kernel_smoothing_matrix(angles_rev, smooth_rad = 0.1):
	nangles = len(angles_rev)
	da = angles_rev[1]-angles_rev[0]
	omat = np.zeros([nangles, nangles])
	nrad = np.floor(smooth_rad/da).astype(np.int32)
	na = 2*nrad+1
	sk_angles = da*(np.arange(na)-nrad)/smooth_rad
	skernel = np.zeros(nangles)
	skernel[0:na] = np.cos(0.5*np.pi*sk_angles)
	
	for i in range(0,nangles):
		omat[i] = np.roll(skernel,i-nrad)

	return omat

# Sam's code below:

def gen_kernel(theta, radial_size=660, aspect_ratio=1, right_intensity=1, bottom_intensity=1, elon_abs=None, elon_offset=0,
               blur=0, image_size=2048, oversamp=3, cx=None, cy=None, r_profile=None, dtype='float32'):
    # Generate an image of a kernel at a specific position (an angle theta around a defined center of the stray-light donut)
    
    # Radial position of the inner edge of the donut
    r_to_inner_edge = 181.27180103 + 5
    # Radial position of the center of the kernel
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
        coords = np.linspace(-1/oversamp, image_size - 1/oversamp, image_size * oversamp, endpoint=False)
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
    # rotated (since we're rotating the kernel around the image center, not translating it in a circular path)
    angled_x = xs * np.cos(theta) + ys * np.sin(theta)
    angled_y = xs * np.sin(-theta) + ys * np.cos(-theta)
 
    # Equation for an ellipse   
    kernel = angled_x**2 / a**2 + angled_y**2 / b**2 < 1
    # Identify a bounding box around the non-zero values, to cut down on the work we do later
    rows = np.any(kernel, axis=1)
    cols = np.any(kernel, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    s = np.s_[rmin:rmax+1, cmin:cmax+1]
    kernel = kernel.astype(float)
    
    # Impose linear gradients
    if right_intensity != 1:
        slope_x = (1 - right_intensity) / azimuthal_diameter
        kernel[s] *= (1 - slope_x * (angled_x[s] + azimuthal_diameter))
    
    if bottom_intensity != 1:
        slope_y = (1 - bottom_intensity) / radial_size
        kernel[s] *= (1 - slope_y * (angled_y[s] + radial_size))
    
    # Impose an arbitrary gradient
    if r_profile is not None:
        kernel[s] *= r_profile(angled_x[s] / (radial_size / 2))

    # Block sum to target resolution
    if oversamp > 1:
        kernel = kernel.reshape((image_size, oversamp, image_size, oversamp)).sum(axis=(1, 3))
    
    if blur > 0:
        kernel = scipy.ndimage.gaussian_filter(kernel, blur)
    
    # kernel /= np.max(kernel)
    
    return kernel.astype(dtype)

import numba
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

def gen_kernels(kernel_angles, aspect_ratio=1, right_intensity=1, radial_size=660, # n_kernels=400,
				bottom_intensity=1, elon_abs=None, elon_offset=0, blur=0, image_size=2048,
                oversamp=3, cx=None, cy=None, r_profile=None, n_threads=5):
    # Generate a full set of kernels in parallel, with a set number of kernels spaced evenly around the donut
    with ThreadPoolExecutor(n_threads) as p:
        kernels = p.map(gen_kernel, kernel_angles,
                        repeat(radial_size), repeat(aspect_ratio), repeat(right_intensity), repeat(bottom_intensity), repeat(elon_abs),
                        repeat(elon_offset), repeat(blur), repeat(image_size), repeat(oversamp), repeat(cx), repeat(cy), repeat(r_profile))
    return np.stack(list(kernels))

@numba.njit(parallel=True)
def make_model(kernels, intensity, rmin=0, rmax=-1, cmin=0, cmax=-1):
    # Given a set of kernels and an intensity for each one, sum up the kernels to make a forward model. It's done in
    # numba and in parallel for speed
    if rmax < 0:
        rmax = kernels.shape[1]
    if cmax < 0:
        cmax = kernels.shape[2]
    output = np.empty(kernels.shape[1:])
    for i in range(0, rmin):
        output[i] = 0
    for i in range(0, cmin):
        output[:, i] = 0
    for i in range(rmax, kernels.shape[1]):
        output[i] = 0
    for i in range(cmax, kernels.shape[2]):
        output[:, i] = 0
    for i in numba.prange(rmin, rmax):
        for j in range(cmin, cmax):
            output[i, j] = np.sum(kernels[:, i, j] * intensity)
    return output

