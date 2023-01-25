from typing import Optional
from datetime import datetime
from punchbowl.data import PUNCHData
from prefect import task, get_run_logger


@task
def remove_deficient_pixels(data_object: PUNCHData, 
                            deficient_pixel_map: PUNCHData) -> PUNCHData:
    """subtracts a deficient pixel map from an input data frame.
        
    checks the dimensions of input data frame and map match and
    subtracts the background model from the data frame of interest.
    
    Parameters
    ----------
    data : PUNCHData
        A PUNCHobject data frame to be background subtracted
        
    deficient_pixel_map : PUNCHData
        A deficient_pixel map

    Returns
    -------

    bkg_subtracted_data : ['punchbowl.data.PUNCHData']
        A background subtracted data frame

    TODO
    ----
    
    # TODO: exclude data if flagged in weight array
    # TODO: update meta data with input file and version of deficient pixel map
    # TODO: output weight - update weights
    """

    logger = get_run_logger()
    logger.info("remove_deficient_pixels started")
    
    data_array=data_object.data
    output_wcs=data_object.wcs
    output_meta=data_object.meta
    #output_uncertainty=shape_PUNCHobject.weight
    output_mask=data_object.mask

    
    deficient_pixel_array=deficient_pixel_map.data

    # check dimensions match
    if data_array.shape != deficient_pixel_array.shape:
        raise Exception("f_background_subtraction expects the data_object and"
                         "f_background arrays to have the same dimensions." 
                         f"data_array dims: {data_array.shape} and deficient_pixel_map dims: {deficient_pixel_array.shape}")
            
    bkg_subtracted_data = data_array - deficient_pixel_array
    
    output_PUNCHobject=PUNCHData(bkg_subtracted_data, 
                                wcs=output_wcs, 
                                meta=output_meta, 
                                mask=output_mask)

    logger.info("remove_deficient_pixels finished")
    output_PUNCHobject.add_history(datetime.now(), "LEVEL1-remove_deficient_pixels", "deficient pixels removed")

    return output_PUNCHobject



