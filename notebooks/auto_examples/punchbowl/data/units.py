import astropy.constants as const
import astropy.units as u
import numpy as np
import scipy.interpolate
from astropy.wcs import WCS
from numpy import ndarray

MSB = u.def_unit("MSB", 2.0090000E7 * u.W / u.m ** 2 / u.sr)


def calculate_image_pixel_area(wcs: WCS, data_shape: tuple[int, int], stride: int = 1) -> u.sr:
    """Calculate the sky area of every pixel in an image according to its WCS."""
    xx, yy = np.meshgrid(np.arange(0, data_shape[1] + stride, stride)-0.5,
                         np.arange(0, data_shape[0] + stride, stride)-0.5)
    coords = wcs.pixel_to_world(xx, yy)
    dx = coords[:, 1:].separation(coords[:, :-1]).to(u.deg)[:-1]
    dy = coords[1:, :].separation(coords[:-1, :]).to(u.deg)[:, :-1]
    area = dx * dy
    if stride > 1:
        area /= stride**2
        # Set these areas to be valid at the center of the region they span
        gridx, gridy = xx[0], yy[:, 0]
        gridx = (gridx[:-1] + gridx[1:]) / 2
        gridy = (gridy[:-1] + gridy[1:]) / 2
        # With fill_value=None, this will extrapolate to the edge pixels
        interpolator = scipy.interpolate.RegularGridInterpolator(
            (gridy, gridx), area, bounds_error=False, fill_value=None)
        xx, yy = np.meshgrid(np.arange(0, data_shape[1], 1),
                             np.arange(0, data_shape[0], 1))
        unit = area.unit
        area = interpolator(np.stack((yy, xx), axis=-1))
        area <<= unit
    return area

def msb_to_dn(data: ndarray,
              data_wcs: WCS,
              gain_bottom: float = 4.9 * u.photon / u.DN,
              gain_top: float = 4.9 * u.photon / u.DN,
              wavelength: float = 530. * u.nm,
              exposure: float = 49 * u.s,
              aperture: float = 49.57 * u.mm**2,
              pixel_area_stride: int = 1,
              ) -> ndarray:
    """Convert mean solar brightness to DNs."""
    energy_per_photon = (const.h * const.c / wavelength).to(u.J) / u.photon
    photon_flux = MSB / energy_per_photon
    pixel_scale = calculate_image_pixel_area(data_wcs, data.shape, pixel_area_stride).to(u.sr) / u.pixel
    photon_count = (photon_flux * exposure * aperture * pixel_scale * u.pixel).decompose()
    gain = split_ccd_array(data.shape, gain_bottom, gain_top)
    return data * photon_count / gain


def dn_to_msb(data: ndarray,
              data_wcs: WCS,
              gain_bottom: float = 4.9 * u.photon / u.DN,
              gain_top: float = 4.9 * u.photon / u.DN,
              wavelength: float = 530. * u.nm,
              exposure: float = 49 * u.s,
              aperture: float = 34 * u.mm**2,
              pixel_area_stride: int = 1,
              pixel_scale: u.Quantity = None,
              ) -> ndarray:
    """Convert DN to mean solar brightness."""
    energy_per_photon = (const.h * const.c / wavelength).to(u.J) / u.photon
    photon_flux = MSB / energy_per_photon
    if pixel_scale is None:
        pixel_scale = calculate_image_pixel_area(data_wcs, data.shape, pixel_area_stride).to(u.sr) / u.pixel
    photon_count = (photon_flux * exposure * aperture * pixel_scale * u.pixel)
    gain = split_ccd_array(data.shape, gain_bottom, gain_top)
    factor = (gain / photon_count).decompose().value
    return data * factor


def split_ccd_array(shape:tuple, value_bottom:float, value_top:float) -> ndarray:
    """Generate parameters across CCD halves."""
    array = np.zeros(shape)
    array[:,:shape[1]//2] = value_bottom
    array[:,shape[1]//2:] = value_top
    return array.T
