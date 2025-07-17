from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from astropy.io.fits import Header

from punchbowl.data import NormalizedMetadata


class Limit:
    """Represents a limit."""

    def __init__(self, xkey: str, xs: np.ndarray, ykey: str, ys: np.ndarray, comp: str) -> None:
        """Create a Limit."""
        self.xkey = xkey
        self.xs = xs
        self.ykey = ykey
        self.ys = ys
        self.comp = comp

    def is_good(self, point: Header | NormalizedMetadata | Iterable) -> bool | np.ndarray:
        """Check if a point satisfies a limit."""
        if isinstance(point, (Header, NormalizedMetadata)):
            x = point[self.xkey]
            y = point[self.ykey]
        elif isinstance(point, Iterable) and isinstance(point[0], (Header, NormalizedMetadata)):
            return np.array([self.is_good(p) for p in point])
        else:
            x, y = point
        limit_value = np.interp(x, self.xs, self.ys)
        if self.comp == "<":
            return y < limit_value
        if self.comp == ">":
            return y > limit_value
        if self.comp in ["=", "=="]:
            return y == limit_value
        if self.comp == ">=":
            return y >= limit_value
        if self.comp == "<=":
            return y <= limit_value
        ruff_says = "Unrecognized comparison type"
        raise ValueError(ruff_says)

    def plot(self, points: list[Header | NormalizedMetadata | Iterable] | None = None) -> None:
        """Plot the limit."""
        plt.plot(self.xs, self.ys, color="C1")
        if points:
            if isinstance(points[0], (Header, NormalizedMetadata)):
                xs = [p[self.xkey] for p in points]
                ys = [p[self.ykey] for p in points]
            else:
                xs, ys = points
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            plt.scatter(xs, ys, s=10, color="C0")
            is_bad = ~np.array([self.is_good(p) for p in zip(xs, ys, strict=False)])
            plt.scatter(xs[is_bad], ys[is_bad], s=10, alpha=.5, color="C3")
        plt.xlabel(self.xkey)
        plt.ylabel(self.ykey)

    def serialize(self) -> tuple:
        """Convert the limit to a tuple."""
        return (np.array((self.xkey, *self.xs)),
                np.array((self.ykey, *self.ys)),
                np.array((self.comp,)))

    @staticmethod
    def from_serialized(serialized: tuple) -> "Limit":
        """Convert a tuple to a limit."""
        x, y, comp = serialized
        xkey = x[0].item()
        xs = np.array([float(xx) for xx in x[1:]])
        ykey = y[0].item()
        ys = np.array([float(yy) for yy in y[1:]])
        comp = comp.item()
        return Limit(xkey, xs, ykey, ys, comp)

    def __repr__(self) -> str:
        """Repr."""
        return f"Limit[{self.xkey}, {self.ykey}, {self.comp}]"


class LimitSet:
    """Represents a set of limits."""

    def __init__(self, limits: list[Limit] | None = None) -> None:
        """Create a LimitSet."""
        self.limits = limits or []

    def add(self, limit: Limit) -> None:
        """Add a Limit to the set."""
        self.limits.append(limit)

    def is_good(self, point: Header | NormalizedMetadata | Iterable) -> bool | np.ndarray:
        """Check if a point satisfies all limits."""
        ok = self.limits[0].is_good(point)
        for limit in self.limits[1:]:
            ok = np.logical_and(ok, limit.is_good(point))
        return ok

    def plot(self, points: list[Header] | None = None, xkey: str | None = None, ykey: str | None = None) -> None:
        """Plot the limits."""
        xkey = xkey or self.limits[0].xkey
        ykey = ykey or self.limits[0].ykey
        for limit in self.limits:
            if limit.xkey == xkey and limit.ykey == ykey:
                limit.plot()
        if points:
            xs = np.asarray([p[xkey] for p in points])
            ys = np.asarray([p[ykey] for p in points])
            plt.scatter(xs, ys, s=10, color="C0")
            is_bad = ~np.array([self.is_good(p) for p in points])
            plt.scatter(xs[is_bad], ys[is_bad], s=10, alpha=.5, color="C3")
        plt.xlabel(xkey)
        plt.ylabel(ykey)

    def to_file(self, path: str) -> None:
        """Write to a file."""
        data = {}
        for i, limit in enumerate(self.limits):
            x, y, comp = limit.serialize()
            data[f"x{i}"] = x
            data[f"y{i}"] = y
            data[f"comp{i}"] = comp
        data["n_limits"] = len(self.limits)
        np.savez(path, **data)

    @staticmethod
    def from_file(path: str) -> "LimitSet":
        """Load from a file."""
        data = np.load(path)
        n_limits = data["n_limits"]
        limit_set = LimitSet()
        for i in range(n_limits):
            x = data[f"x{i}"]
            y = data[f"y{i}"]
            comp = data[f"comp{i}"]
            limit_set.add(Limit.from_serialized((x, y, comp)))
        return limit_set
