import numpy as np
from ndcube import NDCube

from punchbowl.data.meta import NormalizedMetadata
from punchbowl.prefect import punch_task


@punch_task
def create_low_noise_task(inputs: list[NDCube]) -> NDCube:
    """Create a low noise image from a set of inputs."""
    num_images = np.nansum([np.logical_not(np.isnan(c.data)) for c in inputs], axis=0)
    new_data = np.nanmean([c.data for c in inputs], axis=0)
    combined_uncertainty = np.sqrt(np.nansum(np.square([c.uncertainty.array for c in inputs]), axis=0))
    final_uncertainty = combined_uncertainty / np.sqrt(num_images)

    new_code = inputs[0].meta.product_code[0] + "A" + inputs[0].meta.product_code[2]
    new_meta = NormalizedMetadata.load_template(new_code, "3")
    # TODO - This needs to be done more specifically, some keywords need to be used for timing, etc
    # timing_keywords = ["DATE-OBS", "DATE-BEG", "DATE-OBS"]  # noqa: ERA001
    # for old_key, old_value in inputs[0].meta.items():
    #     if old_key not in timing_keywords:
    #         new_meta[old_key] = old_value  # noqa: ERA001

    # TODO - timing meta updates

    return NDCube(data=new_data, uncertainty=final_uncertainty, wcs=inputs[0].wcs, meta=new_meta)
