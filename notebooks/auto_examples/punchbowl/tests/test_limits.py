from typing import Iterable

import numpy as np
import pytest
from astropy.io.fits import Header

from punchbowl.data import NormalizedMetadata
from punchbowl.limits import Limit, LimitSet


@pytest.mark.parametrize("type", [Header, NormalizedMetadata, Iterable])
def test_Limit_is_good(type):
    limit = Limit('CROTA', [0, 360], 'DATAMDN', [2, 3], '>')
    if type is Header:
        input_good = Header({'CROTA': 1, 'DATAMDN': 2.5})
        input_bad = Header({'CROTA': 300, 'DATAMDN': 2.5})
    elif type is NormalizedMetadata:
        input_good = NormalizedMetadata.load_template('CR4', '1')
        input_good['CROTA'] = 1
        input_good['DATAMDN'] = 2.5
        input_bad = NormalizedMetadata.load_template('CR4', '1')
        input_bad['CROTA'] = 300
        input_bad['DATAMDN'] = 2.5
    else:
        input_good = (1, 2.5)
        input_bad = (300, 2.5)

    assert limit.is_good(input_good)
    assert not limit.is_good(input_bad)

    if type is not Iterable:
        assert np.all(limit.is_good([input_good, input_bad]) == np.array([True, False]))


def test_Limit_comparison_types():
    limit = Limit('CROTA', [0, 360], 'DATAMDN', [2, 3], '>')
    header = Header({'CROTA': 1, 'DATAMDN': 2.5})
    assert limit.is_good(header)

    limit = Limit('CROTA', [0, 360], 'DATAMDN', [2, 3], '<')
    header = Header({'CROTA': 1, 'DATAMDN': 1})
    assert limit.is_good(header)

    limit = Limit('CROTA', [0, 360], 'DATAMDN', [2, 2], '<=')
    header = Header({'CROTA': 1, 'DATAMDN': 1})
    assert limit.is_good(header)
    header = Header({'CROTA': 1, 'DATAMDN': 2})
    assert limit.is_good(header)

    limit = Limit('CROTA', [0, 360], 'DATAMDN', [2, 2], '>=')
    header = Header({'CROTA': 1, 'DATAMDN': 3})
    assert limit.is_good(header)
    header = Header({'CROTA': 1, 'DATAMDN': 2})
    assert limit.is_good(header)

    limit = Limit('CROTA', [0, 360], 'DATAMDN', [2, 2], '=')
    header = Header({'CROTA': 1, 'DATAMDN': 2})
    assert limit.is_good(header)

    limit = Limit('CROTA', [0, 360], 'DATAMDN', [2, 2], '-')
    header = Header({'CROTA': 1, 'DATAMDN': 2})
    with pytest.raises(ValueError):
        limit.is_good(header)


def test_Limit_serialize():
    limit = Limit('CROTA', [0, 360], 'DATAMDN', [2, 2], '<=')

    limit2 = Limit.from_serialized(limit.serialize())

    assert limit.xkey == limit2.xkey
    assert limit.ykey == limit2.ykey
    assert limit.comp == limit2.comp
    assert np.all(limit.xs == limit2.xs)
    assert np.all(limit.ys == limit2.ys)


def test_LimitSet():
    limit1 = Limit('CROTA', [0, 360], 'DATAMDN', [2, 3], '<=')
    limit2 = Limit('CROTA', [0, 360], 'DATAP90', [5, 4], '>')

    limit_set = LimitSet([limit1])
    limit_set.add(limit2)

    header = Header({'CROTA': 300, 'DATAMDN': 2.5, 'DATAP90': 4.5})
    assert limit_set.is_good(header)

    header = Header({'CROTA': 300, 'DATAMDN': 3, 'DATAP90': 4.5})
    assert not limit_set.is_good(header)

    header = Header({'CROTA': 300, 'DATAMDN': 2.5, 'DATAP90': 5})
    assert limit_set.is_good(header)


def test_LimitSet_to_file(tmp_path):
    limit1 = Limit('CROTA', [0, 360], 'DATAMDN', [2, 3], '<=')
    limit2 = Limit('CROTA', [0, 360], 'DATAP90', [5, 4], '>')

    limit_set = LimitSet([limit1, limit2])

    path = tmp_path / 'limitset.npz'
    limit_set.to_file(path)
    limit_set2 = LimitSet.from_file(path)

    for i in range(2):
        assert limit_set.limits[i].xkey == limit_set2.limits[i].xkey
        assert limit_set.limits[i].ykey == limit_set2.limits[i].ykey
        assert limit_set.limits[i].comp == limit_set2.limits[i].comp
        assert np.all(limit_set.limits[i].xs == limit_set2.limits[i].xs)
        assert np.all(limit_set.limits[i].ys == limit_set2.limits[i].ys)
