import pytest
from imagecodecs import JpeglsError, jpegls_decode


def test_image_decompression_succeeds():
    path = "punchbowl/auto/flows/tests/data/packet_sample.bin"
    with open(path, 'rb') as f:
        packets = f.read()
    image = jpegls_decode(packets)

    assert image.shape == (2048, 2048)

    # these are hand inspected values of the image
    assert image[0, 0] == 0
    assert image[45, 1000] == 252

def test_image_decompression_fails_on_bad_packets():
    path = "punchbowl/auto/flows/tests/data/faulty_packet_sample.bin"
    with open(path, 'rb') as f:
        packets = f.read()

    with pytest.raises(JpeglsError):
        jpegls_decode(packets)
