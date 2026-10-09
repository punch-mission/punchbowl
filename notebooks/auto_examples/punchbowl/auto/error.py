class PunchPipeWarning(Warning):
    pass


class CCSDSPacketConstructionWarning(PunchPipeWarning):
    pass


class CCSDSPacketDatabaseUpdateWarning(PunchPipeWarning):
    pass

class PunchPipeError(Exception):
    pass


class MissingCCSDSDataError(PunchPipeError):
    pass
