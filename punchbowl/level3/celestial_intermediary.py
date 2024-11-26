
import numpy as np
import reproject
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS, utils
from ndcube import NDCube


def to_celestial_frame_cutout(data_cube: NDCube, cdelt: float = 0.02) -> NDCube:
    """
    Reproject an image to a bounding-box cutout of an all-sky map.

    The bounding box of the input image in the celestial frame is found, and that cutout of the all-sky map is
    returned with the input image reprojected into it. All outputs from this function will be on the same pixel grid
    (if the same ``cdelt`` value is provided), so that different outputs from this function can easily be co-aligned by
    adding and removing pixel rows and columns (no reprojected needed).

    Input files must have two image dimensions, and can have arbitrarily-many extra leading dimensions.

    The all-sky map used is a plate carrée (CAR) projection. The CRPIX/CRVAL values of the WCSes returned by this
    function will vary in longitude so that the output frame is approximately centered on the output data,
    but CRVAL latitude will always be zero (as otherwise the CAR projection no longer has straight lat/lon lines).

    The motivation behind this function is to reduce reprojection time and disk space usage when saving a lot of
    images in celestial coordinates, but still keep the stackability that the output images would have if every image
    were reprojected into the same all-sky map.

    The uncertainty layer of the NDCube is expected to be present and is handled.

    Parameters
    ----------
    data_cube : NDCube
        The data cube to be reprojected
    cdelt : float
        The CDELT value to use for the output projection

    Returns
    -------
    reprojected_cube : NDCube
        The cutout of the all-sky map containing the reprojected input data

    """
    # Combine the data and uncertainty layers along a new axis
    data = np.stack((data_cube.data, data_cube.uncertainty.array))

    wcs = data_cube.wcs.celestial

    # We build the output WCS here
    wcs_out = WCS(naxis=2)
    wcs_out.wcs.ctype = "RA---CAR", "DEC--CAR"
    wcs_out.wcs.cdelt = cdelt, cdelt
    # This won't be the exact center of the output image, but that's ok. Just need to be close.
    wcs_out.wcs.crpix = data.shape[-1] // 2, data.shape[-2] // 2
    crval = wcs.wcs.crval
    # Center our output frame near the input data. Since we choose an even multiple of CDELT, we'll be on the
    # same pixel grid as any other output from this function. The latitude value must be zero, otherwise the CAR
    # projection starts bending the lat/lon lines.
    wcs_out.wcs.crval = ((crval[0] // cdelt) * cdelt) % 360, 0

    # Now find the exact bounds of the input image in this output frame, so we can set the output array size
    xs = np.linspace(-1, data.shape[-1], 30)
    ys = np.linspace(-1, data.shape[-2], 30)
    edgex = np.concatenate((xs, np.full(len(ys), xs[-1]), xs, np.zeros(len(ys))))
    edgey = np.concatenate((np.zeros(len(xs)), ys, np.full(len(xs), ys[-1]), ys))

    xs, ys = utils.pixel_to_pixel(wcs, wcs_out, edgex, edgey)
    xmin, xmax = np.floor(xs.min()), np.ceil(xs.max())
    ymin, ymax = np.floor(ys.min()), np.ceil(ys.max())
    out_shape = int(ymax - ymin), int(xmax - xmin)

    # Shift crpix so that the bounding box we just found starts at pixel 0
    crpix = wcs_out.wcs.crpix
    wcs_out.wcs.crpix = crpix[0] - xmin, crpix[1] - ymin

    reprojected = reproject.reproject_adaptive((data, wcs.celestial), wcs_out, out_shape,
                                               return_footprint=False, roundtrip_coords=False,
                                               boundary_mode="strict", conserve_flux=True, center_jacobian=True)
    return NDCube(data=reprojected[0], uncertainty=StdDevUncertainty(reprojected[1]), wcs=wcs_out)


def shift_image_onto(source: NDCube,
                     target: NDCube,
                     fill_value: float = np.nan) -> NDCube:
    """
    Aligns one image to the frame of a second, if both are different cutouts of the same all-sky coordinate frame.

    These two input images should be outputs from `to_celestial_frame_cutout`. Co-aligning the images only requires
    cropping and padding the first image, since the two images are cutouts of the same pixel grid. This provides the
    same accuracy (to within ~floating-point error) as reprojecting the first image into the second's frame,
    while being incredibly faster.

    Input files must have two image dimensions, and can have arbitrarily-many extra leading dimensions.

    The uncertainty layer of the NDCube is expected to be present and is handled.

    Parameters
    ----------
    source : NDCube
        The image to the aligned
    target : NDCube
        The image onto which ``input`` is to be aligned
    fill_value : float
        The value to be used for empty pixels (pixels of the output frame that aren't spanned by the input image)

    Returns
    -------
    aligned_image : NDCube
        The data of the ``input`` in the WCS frame of ``target``.

    """
    if any(source.wcs.wcs.cdelt != target.wcs.wcs.cdelt):
        msg = "WCSes must have identical CDELTs"
        raise ValueError(msg)

    source_data = np.stack((source.data, source.uncertainty.array))

    # Record any amounts by which we'll have to pad the array
    padding = np.zeros((2, 2), dtype=int)

    corner = utils.pixel_to_pixel(target.wcs, source.wcs, target.data.shape[-1] - 1, target.data.shape[-2] - 1)
    # Sanity check---since the images are on the same pixel grids, the corner's coordinates should come out as integers
    np.testing.assert_allclose(np.rint(corner), corner)
    corner = np.rint(corner).astype(int)

    dcorner = corner[0] - (source_data.shape[-1] - 1), corner[1] - (source_data.shape[-2] - 1)
    if dcorner[0] < 0:
        source_data = source_data[..., :dcorner[0]]
    else:
        padding[1][1] = dcorner[0]
    if dcorner[1] < 0:
        source_data = source_data[..., :dcorner[1], :]
    else:
        padding[0][1] = dcorner[1]

    corner2 = utils.pixel_to_pixel(target.wcs, source.wcs, 0, 0)
    # Sanity check---since the images are on the same pixel grids, the corner's coordinates should come out as integers
    np.testing.assert_allclose(np.rint(corner2), corner2)
    corner2 = np.rint(corner2).astype(int)

    if corner2[0] > 0:
        source_data = source_data[..., :, corner2[0]:]
    else:
        padding[1][0] = -corner2[0]
    if corner2[1] > 0:
        source_data = source_data[..., corner2[1]:, :]
    else:
        padding[0][0] = -corner2[1]

    if corner2[0] > 0 and corner[0] < 0:
        # This can occur when there's no overlap between the two frames.
        padding[1] = [0, target.data.shape[-1]]

    if np.any(padding != 0):
        if source_data.ndim > 2:
            padding = np.concatenate(([(0, 0)] * (source_data.ndim - 2), padding), axis=0)
        source_data = np.pad(source_data, padding, constant_values=fill_value)
    return NDCube(data=source_data[0], wcs=target.wcs, uncertainty=StdDevUncertainty(source_data[1]))
