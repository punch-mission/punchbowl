import pytest
from pytest import fixture
import pathlib
import os
import numpy as np
from punchbowl.data import (
    PUNCHData,
    History,
    HistoryEntry,
    HeaderTemplate,
    HEADER_TEMPLATE_COLUMNS,
)


from punchbowl.level3.f_corona_model import query_f_corona_model_source
from punchbowl.level3.f_corona_model import construct_f_corona_background
from punchbowl.level3.f_corona_model import subtract_f_corona_background



TEST_DIRECTORY = pathlib.Path(__file__).parent.resolve()
TESTDATA_DIR = os.path.dirname(__file__)
SAMPLE_FITS_PATH = os.path.join(TESTDATA_DIR, "L0_CL1_20211111070246_PUNCHData.fits")


@fixture
def sample_data():
    return PUNCHData.from_fits(SAMPLE_FITS_PATH)

@fixture
def sample_data_list():
    number_elements=25
    data_list=[]
    for iStep in range(number_elements):
        data_list.append(SAMPLE_FITS_PATH)
    return data_list

def test_list_input_2(sample_data_list):
    #background = construct_f_corona_background.fn(sample_data)
    #assert isinstance(background, PUNCHData)
    assert isinstance(sample_data_list, list)
    
#def test_stupid_run(sample_data_list):
#    background = construct_f_corona_background.fn(sample_data_list)
#    #assert isinstance(background, PUNCHData)
#    assert isinstance(background, np.ndarray)
#


#def test_bad2(sample_data_list):
#    print("hereHEREhereHEREhereHEREhereHEREhereHEREhereHERE")
#    print(len(sample_data_list))
#    for z_step in range(len(sample_data_list)):
#        address_out=sample_data_list[z_step]
#        print(address_out)
#        if z_step==0:
#            new_data_cube_base=PUNCHData(PUNCHData.from_fits(address_out)).data
#        if z_step>0:
#            new_data_cube_base=np.dstack((new_data_cube_base, PUNCHData(PUNCHData.from_fits(address_out)).data))
#    print(np.shape(new_data_cube_base))
#    assert True

