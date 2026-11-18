from __future__ import annotations

from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from dateutil.parser import parse as parse_datetime_str
from matplotlib.lines import Line2D

from punchbowl.auto.control.db import File as File_DB
from punchbowl.auto.control.util import get_database_session

# This file contains a bunch of classes to simulate the pipeline, as well as utility functions for running and
# plotting it. Each file type has a class that inherits from File, adjusting t0, the imaging cadence,
# and other parameters as needed. Each type's t0 defines a nominal start time---the exact time doesn't matter,
# it just defines what e.g. a cadence of four minutes is relative to. The workhorse functions are
# get_file_closest_to, get_files_closest_to, and get_files_between. Each subclass uses those to define get_inputs and
# get_cal_files. The first returns the direct inputs (e.g. the specific CIM that becomes a CTM), and the latter
# returns any pipeline-generated background files needed (e.g. the CFMs that are interpolated and  subtracted from
# the CIM.) The L1 files contain some extra logic to handle polarization and observatory, so we can just have an L1
# class instead of having to define CR1, CR2, PP3, etc. The hierarchy reads the pipeline configuration to set
# everything up---provide it by calling File.set_pipeline_config. That function uses some funny business to get every
# subclass to read the configuration. See monitor/pages/dependency_tree.py for an example of using this module.

@dataclass
class TreeData:
    """Class to hold the varied outputs from `build_tree`."""

    all_files: list[File]
    connections1_x: list[tuple[datetime, datetime, None]]
    connections2_x: list[tuple[datetime, datetime, None]]
    connections3_x: list[tuple[datetime, datetime, None]]
    connections1_y: list[tuple[str, str, str]]
    connections2_y: list[tuple[str, str, str]]
    connections3_y: list[tuple[str, str, str]]
    mcc_x: list[tuple]
    mcc_y: list[tuple]
    main_chain: list[File]


def build_tree(top: File | list[File]) -> TreeData:
    """
    Build a dependency tree.

    Parameters
    ----------
    top : File | list[File]
        The tope node(s) of the tree. Multiple top-nodes generate a combined dependency tree with multiple "main chains"
        slicing through it.

    Returns
    -------
    TreeData
        The generated data

    """
    if not isinstance(top, (list, tuple)):
        top = [top]
    # This recursively builds the tree
    nodes_to_visit = deque([*top])
    all_files = [*top]
    all_files_set = {*top}
    # We record connections between files by file level. Since there are very different amounts of files and connections at
    # different levels, we plot them slightly differently
    connections3_x, connections3_y = [], []
    connections2_x, connections2_y = [], []
    connections1_x, connections1_y = [], []
    while nodes_to_visit:
        node = nodes_to_visit.popleft()
        parents = node.get_cal_files() + node.get_inputs()
        node.generated_parents = parents
        new_parents = [p for p in parents if p not in all_files_set]
        all_files_set.update(set(new_parents))
        nodes_to_visit.extend(new_parents)
        all_files.extend(new_parents)
        for parent in parents:
            if parent.level == "3":
                c_x, c_y = connections3_x, connections3_y
            elif parent.level == "2":
                c_x, c_y = connections2_x, connections2_y
            else:
                c_x, c_y = connections1_x, connections1_y
            c_x.append((node.timestamp, parent.timestamp))
            c_y.append((node.kind, parent.kind))

    # Now we do it again, ignoring calibration files, to get just the "main chain"
    nodes_to_visit = [*top]
    main_chain = [*top]
    mcc_x, mcc_y = [], []
    while nodes_to_visit:
        node = nodes_to_visit.pop(0)
        parents = node.get_inputs()
        nodes_to_visit.extend(parents)
        main_chain.extend(parents)
        for parent in parents:
            mcc_x.append((node.timestamp, parent.timestamp))
            mcc_y.append((node.kind, parent.kind))

    return TreeData(all_files=all_files,
                    connections1_x=connections1_x, connections2_x=connections2_x, connections3_x=connections3_x,
                    connections1_y=connections1_y, connections2_y=connections2_y, connections3_y=connections3_y,
                    mcc_x=mcc_x, mcc_y=mcc_y, main_chain=main_chain)


def plot_tree(tree_data: TreeData, figsize: tuple | None = None, dark: bool = False) -> None:
    """
    Plot the output of `build_tree`.

    Parameters
    ----------
    tree_data : TreeData
        The output of `build_tree`.
    figsize : tuple | None
        The figure size to use
    dark : bool
        If true, adapt to plotting in dark mode. (This does not itself apply a dark mode, it just adjusts some colors.)

    """
    # Group up the files by type
    files_by_category = defaultdict(list)
    for file in tree_data.all_files:
        files_by_category[file.kind].append(file)

    y = 50
    dy = 0
    category_to_y = {}
    category_is_cal = {}
    timestamps_by_category = {}
    for category, files in files_by_category.items():
        if len(files) > 2000:
            # Down-sample the file counts for types that we have a lot of. This massively improves plotting speed.
            factor = len(files) // 2000
            files = files[::factor]
            files_by_category[category] = files
        timestamps_by_category[category] = {f.timestamp for f in files}

        # Assign a y position to this file code
        if file.level == "1" or file.kind[1] == "F":
            # Use less spacing for very similar product codes
            if dy == 1:
                y -= .2
            if file.kind == "SR1":
                y -= 0.2
            dy = 0.3
            y -= dy
        else:
            if dy == 0.3:
                y += 0.5
            dy = 1.0
            y -= dy
        category_to_y[files[0].kind] = y
        category_is_cal[files[0].kind] = files[0].is_cal

    have_existence_data = tree_data.all_files[0].exists is not None

    if figsize is None:
        figsize = (12, 9)
    plt.figure(figsize=figsize, dpi=200)

    # Plot the files themselves
    for category, files in files_by_category.items():
        is_cal = category_is_cal[category]
        xs = [f.timestamp for f in files]
        ys = np.full(len(xs), category_to_y[category])
        if have_existence_data:
            colors = ["C2" if f.exists else "C1" if f.could_exist else "C3" for f in files]
            sizes = [55 if f.exists else 15 for f in files]
        else:
            colors = "C2" if is_cal else "C0"
            sizes = 15

        plt.scatter(xs, ys, s=sizes, c=colors, marker="|" if not is_cal and have_existence_data else "o")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(list(category_to_y.values()), list(category_to_y.keys()))

    # Plot the connecting lines
    for cx, cy, alpha, lw in ([tree_data.connections3_x, tree_data.connections3_y, 0.4 if dark else 0.5, 2],
                              [tree_data.connections2_x, tree_data.connections2_y, 0.4 if dark else 0.3, 2],
                              [tree_data.connections1_x, tree_data.connections1_y, 0.5 if dark else 0.3, 1],
                              ):
        xs_by_top_kind = defaultdict(list)
        ys_by_top_kind = defaultdict(list)
        for (x1, x2), (y1, y2) in zip(cx, cy, strict=False):
            if x1 in timestamps_by_category[y1] and x2 in timestamps_by_category[y2]:
                key = (y1, x1) if category_is_cal[y1] else (y2, x2) if category_is_cal[y2] else None
                xs_by_top_kind[key].extend([x1, x2, None])
                ys_by_top_kind[key].extend([category_to_y[y1], category_to_y[y2], None])
        for xs, ys in zip(xs_by_top_kind.values(), ys_by_top_kind.values(), strict=False):
            plt.plot(xs, ys, color="0.6" if dark else "0.7", zorder=-10, alpha=alpha, lw=lw)

    mcc_y_cat = []
    for connection in tree_data.mcc_y:
        mcc_y_cat.extend([category_to_y[c] for c in connection])
        mcc_y_cat.append(None)
    mc_cat_vals = [category_to_y[n.kind] for n in tree_data.main_chain]
    mcc_x = []
    for connection in tree_data.mcc_x:
        mcc_x.extend(connection)
        mcc_x.append(None)

    if have_existence_data is None:
        mc_colors = "C0"
    else:
        mc_colors = ["C2" if f.exists else "C3" for f in tree_data.main_chain]

    # Plot the main chain files
    plt.scatter([f.timestamp for f in tree_data.main_chain], mc_cat_vals, s=100, c=mc_colors, zorder=20,
                edgecolors="w" if dark else "k")
    # And the main chain connections
    plt.plot(mcc_x, mcc_y_cat, color="0.8" if dark else "0.2", zorder=19)

    if have_existence_data:
        legend_elements = [
            Line2D([0], [0], color="C2", markersize=10, marker="o", label="File exists", linestyle=""),
            Line2D([0], [0], color="C3", markersize=5, marker="o", label="File doesn't exist", linestyle=""),
            Line2D([0], [0], color="C1", markersize=5, marker="o", label="File doesn't exist but appears ready",
                   linestyle=""),
            Line2D([0], [0], color="w", markersize=10, marker="o", markeredgecolor="k",
                   label="Target file & direct inputs", linestyle="") ]
    else:
        legend_elements = [
            Line2D([0], [0], color="C0", markersize=10, marker="o", label="Image", linestyle=""),
            Line2D([0], [0], color="C2", markersize=10, marker="o", label="Calibration file", linestyle=""),
            Line2D([0], [0], color="C3", markersize=10, marker="o", markeredgecolor="k",
                   label="Target file & direct inputs", linestyle="") ]
    plt.legend(handles=legend_elements, loc="upper right")


def mark_file_existence(tree_data: TreeData) -> None:
    """
    Mark whether files exist in a TreeData.

    This is done by querying the database. Actual files are allowed to differ in their timestamp by a few seconds.

    Parameters
    ----------
    tree_data : TreeData
        An output from `build_tree`. Contents are marked in place.

    """
    files_by_type = defaultdict(list)
    for file in tree_data.all_files + tree_data.main_chain:
        type = (file.level, file.kind[-3:-1])
        files_by_type[type].append(file)

    with get_database_session() as session:
        for type, files in files_by_type.items():
            times = [f.timestamp for f in files]
            db_files = (session.query(File_DB.date_obs, File_DB.observatory)
                        .where(File_DB.level == type[0])
                        .where(File_DB.file_type == type[1])
                        .where(File_DB.state.in_(["created", "progressed"]))
                        .where(
                File_DB.date_obs.between(min(times) - timedelta(seconds=2), max(times) + timedelta(seconds=2))).all())
            existing_files = set()
            for file in db_files:
                t = file.date_obs.replace(microsecond=0)
                existing_files.add((t, file.observatory))
            for file in files:
                file.exists = False
                t = file.timestamp.replace(microsecond=0)
                for dt in [0, -1, 1, -2, 2, -3, 3]:
                    if (t + timedelta(seconds=dt), file.kind[-1]) in existing_files:
                        file.exists = True
                        break

    for file in tree_data.all_files:
        n_parents = len(file.generated_parents)
        if n_parents == 0:
            file.could_exist = False
        else:
            n_existing = len([f for f in file.generated_parents if f.exists])
            if n_parents > 100:
                file.could_exist = n_existing > 0.9 * n_parents
            else:
                file.could_exist = n_existing == n_parents


# An alternate t0 to use for NFI
NFI_T0 = datetime(2025, 3, 1, 0, 0, 21)


class File:
    __slots__ = ("L2_plus_cadence_scale_factor", "L2_t0_shift", "exists", "kind", "pipeline_config", "timestamp")
    t0 = datetime(2025, 3, 1, 0, 2, 30)
    cadence = timedelta(minutes=4)
    new_args = ()
    is_cal = False
    subclasses = []

    @classmethod
    def set_pipeline_config(cls, pipeline_config: dict) -> None:
        """
        Set the pipeline configuration for this class and all subclasses.

        Parameters
        ----------
        pipeline_config : dict
            The config to use.

        """
        cls.pipeline_config = pipeline_config
        config = pipeline_config["flows"]["level2"]
        cls.L2_plus_cadence_scale_factor = 1
        cls.L2_t0_shift = timedelta(seconds=0)
        if ("accepted_phases" in config and "1" not in config["accepted_phases"]) or "5" not in config["accepted_phases"]:
            cls.L2_plus_cadence_scale_factor = 2
            if "5" not in config["accepted_phases"]:
                cls.L2_t0_shift = timedelta(seconds=60 * 4)
        for subclass in cls.subclasses:
            subclass.update_config()

    @classmethod
    def update_config(cls):
        """Update class variables in response to the config file being set."""

    def __init__(self, time: datetime):
        """
        Instantiate a file instance.

        Parameters
        ----------
        time : datetime
            The date_obs of the file

        """
        self.timestamp = time
        self.kind = type(self).__name__
        self.exists = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses.append(cls)

    def get_file_closest_to(self, target: datetime) -> File:
        """
        Find the closest file to target, following the normal cadence for the image type.

        Parameters
        ----------
        target : datetime
            The target time

        Returns
        -------
        File
            The closest file

        """
        return self.get_files_closest_to(target, 1)[0]

    def get_files_closest_to(self, target: datetime, n: int = 1) -> List[File]:
        """
        Find the closest files to target, following the normal cadence for the image type.

        Parameters
        ----------
        target : datetime
            The target time
        n : int
            Return the N closest files.

        Returns
        -------
        list[File]
            The closest files

        """
        # This could be a class method, but for L1s we need to read the instance's pol and obs values
        difference = target - self.t0
        n_steps = difference.total_seconds() // self.cadence.total_seconds()

        t_before = self.t0 + n_steps * self.cadence
        t = t_before + self.cadence

        if abs(t - target) > abs(t_before - target):
            t = t_before

        result_times = [t]
        n_steps = 1
        while len(result_times) < n:
            candidates = [t - self.cadence * n_steps, t + self.cadence * n_steps]
            if len(result_times) < n - 1:
                result_times.insert(0, candidates[0])
                result_times.append(candidates[1])
            else:
                if abs(candidates[0] - target) < abs(candidates[1] - target):
                    result_times.insert(0, candidates[0])
                else:
                    result_times.append(candidates[1])
            n_steps += 1
        return [self.__class__(t, *self.new_args) for t in result_times]

    def get_files_between(self, t1, t2) -> list[File]:
        """
        Get all files between two times, given the normal image cadence.

        Parameters
        ----------
        t1, t2 : datetime
            The datetime bounds

        Returns
        -------
        list[File]
            The identified files

        """
        # This could be a class method, but for L1s we need to read the instance's pol and obs values
        if t2 < t1:
            t1, t2 = t2, t1
        times = []
        t = self.t0

        for step in 1000, 100, 10, 1:
            while True:
                if t1.timestamp() - t.timestamp() > self.cadence.total_seconds() * step:
                    t += self.cadence * step
                else:
                    break

        while t <= t2:
            t += self.cadence
            if t1 <= t <= t2:
                times.append(t)
        return [self.__class__(t, *self.new_args) for t in times]

    def get_cal_files(self) -> list[File]:
        """
        Get all input calibration files for this File instance.

        Returns
        -------
        list[File]
            The calibration files

        """
        return []

    def get_inputs(self) -> list[File]:
        """
        Get all input non-calibration files for this File instance.

        Returns
        -------
        list[File]
            The input files

        """
        return []

    def __str__(self):
        return f"<{self.kind} {self.timestamp.strftime("%Y-%m-%d %H:%M:%S")}>"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.kind, self.timestamp))

    def __eq__(self, other):
        if isinstance(other, File):
            return self.kind == other.kind and self.timestamp == other.timestamp
        return NotImplemented


class PAM(File):
    level = "3"

    @classmethod
    def update_config(cls):
        cls.config = cls.pipeline_config["flows"]["level3_PAM"]
        cls.cadence = timedelta(minutes=32)
        cls.t0 = parse_datetime_str(cls.config["t0"]) - cls.cadence / 2

    def get_inputs(self):
        return PTM().get_files_between(self.timestamp - self.cadence / 2, self.timestamp + self.cadence / 2)


class PTM(File):
    level = "3"

    @classmethod
    def update_config(cls):
        cls.cadence = timedelta(minutes=4) * cls.L2_plus_cadence_scale_factor
        cls.t0 = cls.t0 + cls.L2_t0_shift

    def get_cal_files(self):
        return PSM().get_files_closest_to(self.timestamp, 2)

    def get_inputs(self):
        return PIM().get_files_closest_to(self.timestamp)


class PIM(File):
    level = "3"

    @classmethod
    def update_config(cls):
        cls.cadence = timedelta(minutes=4) * cls.L2_plus_cadence_scale_factor
        cls.t0 = cls.t0 + cls.L2_t0_shift

    def get_inputs(self):
        return L2_PTM().get_files_closest_to(self.timestamp)

    def get_cal_files(self):
        files = []
        for obs in ["1", "2", "3"]:
            files += PFM(None, obs).get_files_closest_to(self.timestamp, 2)
        return files


class CAM(File):
    level = "3"

    @classmethod
    def update_config(cls):
        cls.config = cls.pipeline_config["flows"]["level3_CAM"]
        cls.cadence = timedelta(minutes=32)
        cls.t0 = parse_datetime_str(cls.config["t0"]) - cls.cadence / 2

    def get_inputs(self):
        return (CTM().get_files_between(self.timestamp - self.cadence / 2, self.timestamp + self.cadence / 2)
                + CNN().get_files_closest_to(self.timestamp, 5))


class CTM(File):
    level = "3"
    cadence = timedelta(minutes=8)
    t0 = datetime(2025, 3, 1, 0, 0, 30)

    def get_cal_files(self):
        return CSM().get_files_closest_to(self.timestamp, 2)

    def get_inputs(self):
        return CIM().get_files_closest_to(self.timestamp) + CNN().get_files_closest_to(self.timestamp)


class CIM(File):
    level = "3"
    cadence = timedelta(minutes=8)
    t0 = datetime(2025, 3, 1, 0, 0, 30)

    def get_inputs(self):
        return L2_CTM().get_files_closest_to(self.timestamp)

    def get_cal_files(self):
        files = []
        for obs in ["1", "2", "3"]:
            files += CFM(None, obs).get_files_closest_to(self.timestamp, 2)
        return files


class CNN(File):
    level = "3"
    cadence = timedelta(minutes=8)
    t0 = NFI_T0

    def get_inputs(self):
        return L2_CNN().get_files_closest_to(self.timestamp)

    def get_cal_files(self):
        return CFM(None, "N").get_files_closest_to(self.timestamp, 2)


class XSM(File):
    level = "3"
    is_cal = True

    @classmethod
    def update_config(cls):
        cls.config = cls.pipeline_config["flows"]["construct_starfield_background"]
        cls.cadence = timedelta(days=cls.config["model_spacing_days"])
        cls.t0 = parse_datetime_str(cls.config["t0"])

    def get_inputs(self):
        prefix = "clear" if self.pol == "C" else "pol"
        n = int(2 * self.config[prefix + "_before_min_files"])
        stride = int(self.config[prefix + "_image_cadence"])
        input_kind = CIM if self.pol == "C" else PIM
        inputs = input_kind().get_files_closest_to(self.timestamp, n=n * stride)[::stride]
        return inputs


class PSM(XSM):
    pol = "P"


class CSM(XSM):
    pol = "C"


class XFM(File):
    level = "3"
    is_cal = True

    @classmethod
    def update_config(cls):
        cls.config = cls.pipeline_config["flows"]["construct_f_corona_background"]
        cls.cadence = timedelta(hours=cls.config["model_spacing_hours"])
        cls.t0 = parse_datetime_str(cls.config["t0"])

    def __init__(self, time: dateimte, obs: str):
        """
        Instantiate a file instance.

        Parameters
        ----------
        time : datetime
            The date_obs of the file
        obs : str
            The observatory identifier

        """
        super().__init__(time)
        self.obs = obs
        self.new_args = (obs,)
        self.kind = type(self).__name__[:2] + obs

    def get_inputs(self):
        prefix = "nfi" if self.obs == "N" else "clear" if self.pol == "C" else "pol"
        n = int(2 * self.config[prefix + "_min_files_per_half"])
        stride = int(self.config[prefix + "_file_stride"])
        input_kind = L2_CNN if self.obs == "N" else L2_CTM if self.pol == "C" else L2_PTM
        inputs = input_kind().get_files_closest_to(self.timestamp, n=n * stride)[::stride]
        return inputs


class PFM(XFM):
    pol = "P"


class CFM(XFM):
    pol = "C"


class L2_XTM(File):
    level = "2"

    def get_inputs(self):
        inputs = []
        for obs in ["1", "2", "3"]:
            for pol in self.input_pols:
                inputs.append(L1(None, pol, obs).get_file_closest_to(self.timestamp))
        return inputs


class L2_PTM(L2_XTM):
    input_pols = ["M", "Z", "P"]

    @classmethod
    def update_config(cls):
        cls.cadence = timedelta(minutes=4) * cls.L2_plus_cadence_scale_factor


class L2_CTM(L2_XTM):
    input_pols = ["R"]
    cadence = timedelta(minutes=8)
    t0 = datetime(2025, 3, 1, 0, 0, 30)
    pol = "C"


class L2_CNN(File):
    level = "2"
    t0 = NFI_T0

    def get_inputs(self):
        return [X(None, "R", "4").get_file_closest_to(self.timestamp)]


L1_t0s = {
    "M": datetime(2025, 3, 1, 0, 3, 32),
    "Z": datetime(2025, 3, 1, 0, 2, 30),
    "P": datetime(2025, 3, 1, 0, 1, 26),
    "R": datetime(2025, 3, 1, 0, 0, 29),
}


class L1_File(File):
    level = "1"
    __slots__ = ("new_args", "obs", "pol")

    def __init__(self, time: datetime, pol: str = "Z", obs: str = "1"):
        """
        Instantiate a file instance.

        Parameters
        ----------
        time : datetime
            The date_obs of the file
        pol : str
            The polarization state identifier
        obs : str
            The observatory identifier

        """
        super().__init__(timestamp)
        self.obs = obs
        self.pol = pol
        self.new_args = (pol, obs)
        self._set_kind(pol, obs)

    @property
    def t0(self):
        return NFI_T0 if self.obs == "4" else L1_t0s[self.pol]

    def _set_kind(self, pol, obs):
        self.kind = type(self).__name__ + pol + obs


class L1_Noncal_File(L1_File):
    cadence_clear = timedelta(minutes=8)
    cadence_pol = timedelta(minutes=4)

    @property
    def cadence(self):
        return self.cadence_clear if self.pol == "R" else self.cadence_pol


class L1(L1_Noncal_File):
    def _set_kind(self, pol, obs):
        self.kind = ("C" if pol == "R" else "P") + pol + obs

    def get_inputs(self):
        if self.pol == "R":
            ftype = X
        else:
            ftype = Y
        return ftype(None, self.pol, self.obs).get_files_closest_to(self.timestamp)

    def get_cal_files(self):
        return S(None, self.pol, self.obs).get_files_closest_to(self.timestamp, 2)


class S(L1_File):
    is_cal = True

    @classmethod
    def update_config(cls):
        cls.config = cls.pipeline_config["flows"]["construct_stray_light"]
        cls.cadence = timedelta(hours=cls.config["model_spacing_hours"])
        cls.t0 = parse_datetime_str(cls.config["t0"])

    def get_inputs(self):
        prefix = "clear" if self.pol == "R" else "pol"

        n = int(2 * self.config[prefix + "_min_files_per_half"])
        stride = int(self.config[prefix + "_file_stride"])
        input_kind = X if self.pol == "R" else Y
        inputs = input_kind(None, self.pol, self.obs).get_files_closest_to(self.timestamp, n=n * stride)[::stride]
        return inputs


class Y(L1_Noncal_File):
    def get_inputs(self):
        return X(None, self.pol, self.obs).get_files_closest_to(self.timestamp)

    def get_cal_files(self):
        return T(None, self.pol, self.obs).get_files_closest_to(self.timestamp, 2)


class T(L1_File):
    is_cal = True

    @classmethod
    def update_config(cls):
        cls.config = cls.pipeline_config["flows"]["construct_dynamic_stray_light"]
        cls.cadence = timedelta(hours=cls.config["model_spacing_hours"])
        cls.t0 = parse_datetime_str(cls.config["t0"])

    def get_inputs(self):
        n = int(2 * self.config["min_files_per_half"])
        stride = 1
        inputs = X(None, self.pol, self.obs).get_files_closest_to(self.timestamp, n=n * stride)[::stride]
        return inputs


class X(L1_Noncal_File):
    pass
