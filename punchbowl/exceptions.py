class PUNCHBowlError(Exception):
    """Base class for exceptions in punchbowl."""


class InvalidDataError(PUNCHBowlError):
    """Invalid data error."""


class InvalidHeaderError(PUNCHBowlError):
    """Header is not properly formatted."""


class MissingMetadataError(PUNCHBowlError):
    """Metadata missing for processing."""


class PUNCHBowlWarning(Warning):
    """Base class for warnings in punchbowl."""


class ExtraMetadataWarning(PUNCHBowlWarning):
    """Extra metadata found but ignored."""
