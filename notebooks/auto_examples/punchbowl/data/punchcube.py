from typing import TYPE_CHECKING

import numpy as np
from astropy.nddata import StdDevUncertainty
from astropy.units import Unit
from astropy.wcs import WCS
from ndcube import NDCube

if TYPE_CHECKING:
    import punchbowl.data.meta


class PUNCHCube(NDCube):
    """PUNCHCube with a secondary celestial WCS."""

    def __init__(self, *args: list, celestial_wcs: WCS | None = None,  **kwargs: dict) -> None:
        """Initialize a PUNCHCube."""
        super().__init__(*args, **kwargs)
        self.celestial_wcs = celestial_wcs

    def replace(self, data: np.ndarray | None = None,
                meta: "punchbowl.data.meta.NormalizedMetadata | None" = None, wcs: WCS | None = None,
                celestial_wcs: WCS | None = None, unit: Unit | None = None,
                mask: np.ndarray | None = None,
                uncertainty: StdDevUncertainty | None = None) -> "PUNCHCube":
        """
        (Shallow) copy this PUNCHCube, but with certain attributes replaced.

        Useful because PUNCHCubes don't allow changing their data array, WCS, etc., so to change the WCS you have to
        make a new cube. Using this function ensures everything is copied over.

        Parameters
        ----------
        data : ndarray | None
            A replacement data array
        meta : NormalizedMetadata | None
            A replacement meta
        wcs : WCS | None
            A replacement WCS
        celestial_wcs : WCS | None
            A replacement celestial WCS
        unit : Unit | None
            A replacement unit
        mask : np.ndarray | None
            A replacement mask
        uncertainty : StdDevUncertainty | None
            A replacement uncertainty

        Returns
        -------
        cube : PUNCHCube

        """
        return PUNCHCube(data=data if data is not None else self.data,
                         meta=meta if meta is not None else self.meta,
                         wcs=wcs if wcs is not None else self.wcs,
                         celestial_wcs=celestial_wcs if celestial_wcs is not None else self.celestial_wcs,
                         unit=unit if unit is not None else self.unit,
                         mask=mask if mask is not None else self.mask,
                         uncertainty=uncertainty if uncertainty is not None else self.uncertainty)
