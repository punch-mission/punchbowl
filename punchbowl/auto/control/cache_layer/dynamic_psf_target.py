import os

from regularizepsf import ArrayPSF

from punchbowl.auto.control.cache_layer import manager
from punchbowl.auto.control.cache_layer.loader_base_class import LoaderABC
from punchbowl.util import DataLoader


class DynamicPSFTargetLoader(LoaderABC[ArrayPSF]):
    def __init__(self, path: str):
        self.path = path

    def gen_key(self) -> str:
        return f"dynamic-psf-target-{os.path.basename(self.path)}-{os.path.getmtime(self.path)}"

    def src_repr(self) -> str:
        return self.path

    def load_from_disk(self) -> ArrayPSF:
        return ArrayPSF.load(self.path)

    def __repr__(self):
        return f"DynamicPSFTargetLoader({self.path})"


def wrap_if_appropriate(target_path: str) -> str | DataLoader:
    if manager.caching_is_enabled():
        return DynamicPSFTargetLoader(target_path)
    return target_path
