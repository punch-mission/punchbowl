import itertools

import lmfit
import numpy as np
import pandas as pd
import scipy.ndimage
from astropy.io import fits
from astropy.wcs import WCS
from lmfit.lineshapes import gaussian2d
from skimage.morphology import dilation, erosion

# This file will be deleted and replaced with alignment.py


def load_test_image():
    img = fits.open("data/WFI/L0_CL1_20211111063222.fits")[0].data
    hdr = fits.open("data/WFI/L0_CL1_20211111063222.wcs")[0].header
    wcs = WCS(hdr)
    return img, hdr, wcs


def load_catalog(catalog_path="data/hip_main.dat.txt"):
    column_names = (
        "Catalog",
        "HIP",
        "Proxy",
        "RAhms",
        "DEdms",
        "Vmag",
        "VarFlag",
        "r_Vmag",
        "RAdeg",
        "DEdeg",
        "AstroRef",
        "Plx",
        "pmRA",
        "pmDE",
        "e_RAdeg",
        "e_DEdeg",
        "e_Plx",
        "e_pmRA",
        "e_pmDE",
        "DE:RA",
        "Plx:RA",
        "Plx:DE",
        "pmRA:RA",
        "pmRA:DE",
        "pmRA:Plx",
        "pmDE:RA",
        "pmDE:DE",
        "pmDE:Plx",
        "pmDE:pmRA",
        "F1",
        "F2",
        "---",
        "BTmag",
        "e_BTmag",
        "VTmag",
        "e_VTmag",
        "m_BTmag",
        "B-V",
        "e_B-V",
        "r_B-V",
        "V-I",
        "e_V-I",
        "r_V-I",
        "CombMag",
        "Hpmag",
        "e_Hpmag",
        "Hpscat",
        "o_Hpmag",
        "m_Hpmag",
        "Hpmax",
        "HPmin",
        "Period",
        "HvarType",
        "moreVar",
        "morePhoto",
        "CCDM",
        "n_CCDM",
        "Nsys",
        "Ncomp",
        "MultFlag",
        "Source",
        "Qual",
        "m_HIP",
        "theta",
        "rho",
        "e_rho",
        "dHp",
        "e_dHp",
        "Survey",
        "Chart",
        "Notes",
        "HD",
        "BD",
        "CoD",
        "CPD",
        "(V-I)red",
        "SpType",
        "r_SpType",
    )
    df = pd.read_csv(
        catalog_path,
        sep="|",
        names=column_names,
        usecols=["HIP", "Vmag", "RAdeg", "DEdeg"],
        na_values=["     ", "       ", "        ", "            "],
    )
    return df


def get_potential_stars(catalog, center, radius, mag_threshold=5):
    ra_mask = (center[0] - radius < catalog.RAdeg) & (
        catalog.RAdeg < center[0] + radius
    )
    dec_mask = (center[1] - radius < catalog.DEdeg) & (
        catalog.DEdeg < center[1] + radius
    )
    mag_mask = catalog.Vmag < mag_threshold
    mask = ra_mask & dec_mask & mag_mask
    subset = catalog[mask]
    return subset


def calculate_first_star_mask(
    data,
    median_filter_size=20,
    noise_pixel_separation=5,
    num_noise_pixels=3000,
    mask_sigma=8,
):
    data_median_filtered = data - scipy.ndimage.median_filter(
        data, size=median_filter_size
    )
    coordinates = np.array(
        list(
            itertools.product(
                np.arange(data.shape[0] - noise_pixel_separation),
                np.arange(data.shape[1] - noise_pixel_separation),
            )
        )
    )
    sample_pixels = coordinates[
        np.random.choice(np.arange(len(coordinates)), num_noise_pixels)
    ]
    data_std = np.sqrt(
        np.nanvar(
            data_median_filtered[sample_pixels]
            - data_median_filtered[sample_pixels + noise_pixel_separation]
        )
        / 2
    )
    candidate_mask = data_median_filtered > (mask_sigma * data_std)
    final_mask = scipy.ndimage.label(candidate_mask)
    return data_median_filtered, final_mask[0]


def clean_star_mask(data, mask, boundary_width=10):
    boundary_mask = np.ones(data.shape, dtype=bool)
    boundary_mask[
        boundary_width:-boundary_width, boundary_width:-boundary_width
    ] = False
    mask[boundary_mask] = 0
    return data, mask


def fit_star_centers(data, mask, window_size=3):
    if window_size % 2 != 1:
        raise RuntimeError("window_size must be odd")
    half_window = window_size // 2

    centers = []

    for i in np.unique(mask):
        if i != 0:  # 0 is used for no star found label
            pixels = np.where(mask == i)
            peak_pixel_index = np.argmax(data[pixels])
            peak_pixel = (pixels[0][peak_pixel_index], pixels[1][peak_pixel_index])
            xstart, xend = peak_pixel[0] - half_window, peak_pixel[0] + half_window + 1
            ystart, yend = peak_pixel[1] - half_window, peak_pixel[1] + half_window + 1

            coords = np.array(
                list(
                    itertools.product(np.arange(xstart, xend), np.arange(ystart, yend))
                )
            )
            x = coords[:, 0]
            y = coords[:, 1]
            z = data[x, y]

            if not np.any(np.isnan(z)):
                model = lmfit.models.Gaussian2dModel()
                params = model.guess(z, x, y)
                result = model.fit(z, x=x, y=y, params=params)
                centers.append((result.values["centerx"], result.values["centery"]))

    centers = np.array(centers)
    return centers


def solve_direct(center, radius, best_guess_wcs, img):
    catalog = load_catalog()
    potential_stars = get_potential_stars(catalog, center, radius)
    img_masking, mask = calculate_first_star_mask(img, mask_sigma=3)
    mask = dilation(erosion(mask > 0))
    mask, _ = scipy.ndimage.label(mask)
    centers = fit_star_centers(img_masking, mask)
    centers_ra, centers_dec = best_guess_wcs.all_pix2world(
        centers[:, 1], centers[:, 0], 0
    )

    matching = []
    for ra, dec in zip(centers_ra, centers_dec):
        distances = np.sqrt(
            np.square(ra - potential_stars.RAdeg)
            + np.square(dec - potential_stars.DEdeg)
        )
        matching.append(np.argmin(distances))

    catalog_portion = potential_stars.iloc[matching]

    y = np.stack([centers_ra, centers_dec], axis=1)
    x = np.stack([catalog_portion.RAdeg, catalog_portion.DEdeg], axis=1)

    x0 = np.mean(x, axis=0)
    y0 = np.mean(y, axis=0)

    H = np.sum(
        [np.outer((y[i] - y0), (x[i] - x0).T) for i in range(x.shape[0])], axis=0
    )

    u, s, v = np.linalg.svd(H)

    R = np.matmul(v, u.T)
    t = y0 - np.matmul(R, x0)

    # TODO: update the wcs using the extracted rotation and translation
    return R, t  # TODO: change this to return the updated WCS instead
