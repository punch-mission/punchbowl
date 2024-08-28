import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.wcs import WCS
from numpy import ndarray

MSB = u.def_unit("MSB", 2.0090000E7 * u.W / u.m ** 2 / u.sr)


def calculate_image_pixel_area(wcs: WCS, data_shape: tuple[int, int]) -> u.sr:
    """Calculate the sky area of every pixel in an image according to its WCS."""
    xx, yy = np.meshgrid(np.arange(data_shape[0]+1), np.arange(data_shape[1]+1))
    coords = wcs.pixel_to_world(xx, yy)
    tx = coords.Tx.to(u.deg).value
    ty = coords.Ty.to(u.deg).value
    pixel_area = abs(tx[:, 1:] - tx[:, :-1])[1:, :] * abs(ty[1:] - ty[:-1])[:, 1:]
    return pixel_area * u.degree * u.degree


def msb_to_dn(data: ndarray,
              data_wcs: WCS,
              gain: float = 4.9 * u.photon / u.DN,
              wavelength: float = 530. * u.nm,
              exposure: float = 49 * u.s,
              aperture: float = 49.57 * u.mm**2,
              ) -> ndarray:
    """Convert mean solar brightness to DNs."""
    energy_per_photon = (const.h * const.c / wavelength).to(u.J) / u.photon
    photon_flux = MSB / energy_per_photon
    pixel_scale = calculate_image_pixel_area(data_wcs, data.shape).to(u.sr) / u.pixel
    photon_count = (photon_flux * exposure * aperture * pixel_scale * u.pixel).decompose()
    return (data * photon_count / gain).astype(int)


def dn_to_msb(data: ndarray,
              data_wcs: WCS,
              gain: float = 4.9 * u.photon / u.DN,
              wavelength: float = 530. * u.nm,
              exposure: float = 49 * u.s,
              aperture: float = 34 * u.mm**2,
              ) -> ndarray:
    """Convert DN to mean solar brightness."""
    energy_per_photon = (const.h * const.c / wavelength).to(u.J) / u.photon
    photon_flux = MSB / energy_per_photon
    pixel_scale = calculate_image_pixel_area(data_wcs, data.shape).to(u.sr) / u.pixel
    photon_count = (photon_flux * exposure * aperture * pixel_scale * u.pixel).decompose()
    return data * gain / photon_count
