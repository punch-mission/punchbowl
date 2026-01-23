import os

from regularizepsf import ArrayPSFTransform

from punchbowl.pipe.control.cache_layer import manager
from punchbowl.pipe.control.cache_layer.loader_base_class import LoaderABC
from punchbowl.util import DataLoader


class PSFLoader(LoaderABC[ArrayPSFTransform]):
    def __init__(self, path: str):
        self.path = path

    def gen_key(self) -> str:
        return f"psf-{os.path.basename(self.path)}-{os.path.getmtime(self.path)}"

    def src_repr(self) -> str:
        return self.path

    def load_from_disk(self) -> ArrayPSFTransform:
        return ArrayPSFTransform.load(self.path)

    def __repr__(self):
        return f"PSFLoader({self.path})"


def wrap_if_appropriate(psf_path: str) -> str | DataLoader:
    if manager.caching_is_enabled():
        return PSFLoader(psf_path)
    return psf_path
