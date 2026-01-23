import os

from ndcube import NDCube

from punchbowl.data import load_ndcube_from_fits
from punchbowl.pipe.control.cache_layer import manager
from punchbowl.pipe.control.cache_layer.loader_base_class import LoaderABC
from punchbowl.util import DataLoader


class StrayLightLoader(LoaderABC[NDCube]):
    def __init__(self, path: str):
        self.path = path

    def gen_key(self) -> str:
        return f"sl-{os.path.basename(self.path)}-{os.path.getmtime(self.path)}"

    def src_repr(self) -> str:
        return self.path

    def load_from_disk(self) -> NDCube:
        return load_ndcube_from_fits(self.path)

    def __repr__(self):
        return f"StrayLight({self.path})"


def wrap_if_appropriate(quartic_path: str) -> str | DataLoader:
    if manager.caching_is_enabled():
        return StrayLightLoader(quartic_path)
    return quartic_path
