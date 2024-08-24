import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import copy
from numpy import ndarray
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area

MSB = u.def_unit('MSB', 2.0090000E7 * u.W / u.m ** 2 / u.sr)

def msb_to_dn(data: ndarray,
              data_wcs: WCS,
              gain: float = 4.9 * u.photon / u.DN,
              wavelength: float = 530. * u.nm,
              exposure: float = 49 * u.s,
              aperture: float = 49.57 * u.mm**2
              ) -> ndarray:
    energy_per_photon = (const.h * const.c / wavelength).to(u.J) / u.photon
    photon_flux = MSB / energy_per_photon
    pixel_scale = (proj_plane_pixel_area(data_wcs) * u.deg**2).to(u.sr) / u.pixel
    photon_count = (photon_flux * exposure * aperture * pixel_scale * u.pixel).decompose()
    data_dn = (data * photon_count / gain).astype(int)
    return data_dn

def dn_to_msb(data: ndarray,
              data_wcs: WCS,
              gain: float = 4.9 * u.photon / u.DN,
              wavelength: float = 530. * u.nm,
              exposure: float = 49 * u.s,
              aperture: float = 34 * u.mm**2
              ) -> ndarray:
    energy_per_photon = (const.h * const.c / wavelength).to(u.J) / u.photon
    photon_flux = MSB / energy_per_photon
    pixel_scale = (proj_plane_pixel_area(data_wcs) * u.deg**2).to(u.sr) / u.pixel
    photon_count = (photon_flux * exposure * aperture * pixel_scale * u.pixel).decompose()
    data_msb = data * gain / photon_count
    return data_msb