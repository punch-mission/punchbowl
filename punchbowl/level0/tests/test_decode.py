from punchbowl.level0.decode import create_fake_ndcube

from ndcube import NDCube


def test_created_fake_ndcube():
    cube = create_fake_ndcube()
    assert isinstance(cube, NDCube)
