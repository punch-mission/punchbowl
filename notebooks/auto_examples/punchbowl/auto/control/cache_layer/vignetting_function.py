import os

from ndcube import NDCube

from punchbowl.auto.control.cache_layer import manager
from punchbowl.auto.control.cache_layer.loader_base_class import LoaderABC
from punchbowl.data import load_ndcube_from_fits
from punchbowl.util import DataLoader


class VignettingLoader(LoaderABC[NDCube]):
    def __init__(self, path: str):
        self.path = path

    def gen_key(self) -> str:
        return f"vignetting-{os.path.basename(self.path)}-{os.path.getmtime(self.path)}"

    def src_repr(self) -> str:
        return self.path

    def load_from_disk(self) -> NDCube:
        return load_ndcube_from_fits(self.path, include_provenance=False)

    def __repr__(self):
        return f"VignettingLoader({self.path})"


def wrap_if_appropriate(vignetting_path: str) -> str | DataLoader:
    if manager.caching_is_enabled():
        return VignettingLoader(vignetting_path)
    return vignetting_path
