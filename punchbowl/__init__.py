"""punchbowl: data reduction and calibration pipeline for PUNCH."""
import importlib.metadata

try:
  __version__ = importlib.metadata.version("punchbowl")
except PackageNotFoundError:
  __version__ = "0.0.0"
