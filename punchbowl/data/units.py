import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.wcs import WCS
from numpy import ndarray

MSB = u.def_unit("MSB", 2.0090000E7 * u.W / u.m ** 2 / u.sr)


def calculate_image_pixel_area(wcs: WCS, data_shape: tuple[int, int], stride: int = 1) -> u.sr:
    """Calculate the sky area of every pixel in an image according to its WCS."""
    ny, nx = data_shape
    y, x = np.indices((ny, nx), dtype=float)

    # SIP/PV distortions are included
    lon_ctr, lat_ctr = wcs.all_pix2world(x, y, 0)
    lon_dx, lat_dx = wcs.all_pix2world(x + stride, y, 0)
    lon_dy, lat_dy = wcs.all_pix2world(x, y + stride, 0)

    dlon_dx = ((lon_dx - lon_ctr + 180) % 360 - 180) / stride
    dlon_dy = ((lon_dy - lon_ctr + 180) % 360 - 180) / stride

    dlat_dx = (lat_dx - lat_ctr) / stride
    dlat_dy = (lat_dy - lat_ctr) / stride

    # The width of a degree of HPLN/RA shrinks as we move toward the poles
    cos_lat = np.cos(np.radians(lat_ctr))

    # Compute the Jacobian Determinant (Area)
    # Area = | (dlon_dx * cos_lat * dlat_dy) - (dlon_dy * cos_lat * dlat_dx) |
    return np.abs((dlon_dx * cos_lat * dlat_dy) - (dlon_dy * cos_lat * dlat_dx))* u.deg**2

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
