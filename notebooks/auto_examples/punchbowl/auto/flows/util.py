import os
from datetime import datetime

from punchbowl.auto.control.db import File


def file_name_to_full_path(file_name: str | None | list[str | None], root_dir: str) -> str | None | list[str | None]:
    if isinstance(file_name, list):
        return [file_name_to_full_path(f, root_dir) for f in file_name]
    if file_name is None:
        return None
    level = file_name.split("_")[1][1]
    code = file_name.split("_")[2][:2]
    obs = file_name.split("_")[2][-1]
    version = file_name.split("_")[-1].split(".")[0][1:]
    date = datetime.strptime(file_name.split("_")[3], "%Y%m%d%H%M%S")

    file = File(level=level, file_type=code, observatory=obs, file_version=version, software_version="", date_obs=date,
                state="")

    return os.path.join(file.directory(root_dir), file_name)


def summarize_files_missing_cal_files(files: list[File]):
    if len(files) < 1000:
        summary = ", ".join([f.filename() for f in files])
    else:
        dates = [f.date_obs for f in files]
        types = {f.file_type + f.observatory for f in files}
        summary = (f"{len(files)} files, of types {sorted(types)}, with date-obs ranging from "
                   f"{min(dates).isoformat()} to {max(dates).isoformat()}")
    return summary
