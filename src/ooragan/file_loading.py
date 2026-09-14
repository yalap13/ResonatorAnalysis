import os
import h5py
import re
import numpy as np
from glob import glob
from pathlib import Path
from typing import Any, Self, Optional, overload
from warnings import warn

from numpy.typing import NDArray

from .parameters import NullParameter, Parameter
from .util import convert_complex_to_magphase, convert_magphase_to_complex, str_to_time

KNOWN_PARAMETERS = [
    "VNA",
    "VNA Average",
    "VNA Power",
    "VNA Bandwidth",
    "Variable Attenuator",
    "VNA Frequency",
    "s21_real",
    "s21_imag",
    "s21_mag",
    "s21_phase",
    "Index",
    "Magnet",
]


def _broadcast_along_axis(array: NDArray, shape: tuple, axis: int) -> NDArray:
    """Broadcasts a 1D array along a given axis to match the target shape."""
    if array.shape == shape:
        return array
    if array.ndim != 1:
        raise ValueError("Input array muxt be 1-dimensional")
    if not (0 <= axis < len(shape)):
        raise ValueError("Axis out of bounds for target shape")
    if shape[axis] != array.shape[0]:
        raise ValueError(f"Size mismatch: shape[{axis}] != len(array)")
    reshaped = array.reshape(
        [shape[axis] if i == axis else 1 for i in range(len(shape))]
    )
    return np.broadcast_to(reshaped, shape)


def _walk_hdf(
    file_or_group: Any,
    additional_params: list[str],
) -> dict[str, dict]:
    """Walks an HDF file hierarchy and converts it into a dictionary."""
    out = {}
    for key in file_or_group.keys():
        if key in KNOWN_PARAMETERS or key in additional_params:
            match type(file_or_group[key]):
                case h5py.Dataset:
                    out[key] = {
                        "values": np.asarray(file_or_group[key]),
                        "description": None,
                        "unit": None,
                    }
                    if "Description" in file_or_group[key].attrs.keys():
                        out[key]["description"] = file_or_group[key].attrs[
                            "Description"
                        ]
                    if "Unit" in file_or_group[key].attrs.keys():
                        out[key]["unit"] = file_or_group[key].attrs["Unit"]
                case h5py.Group:
                    out[key] = _walk_hdf(file_or_group[key], additional_params)
                case _:
                    raise TypeError("Invalid type")
    return out


def _read_hdf(path: str | Path, additional_params: list[str]) -> dict:
    """Reads an HDF file from its path."""
    out = {"attributes": {}, "datasets": {}, "dimensions": []}
    try:
        file = h5py.File(path, "r")
    except OSError as e:
        print(path)
        raise e
    for atr in file.attrs.keys():
        if atr in ["Ended", "Started"]:
            out["attributes"][atr] = str_to_time(file.attrs[atr])
        else:
            out["attributes"][atr] = file.attrs[atr]
    out["datasets"] = _walk_hdf(file, additional_params)
    vna_group = file["VNA"]
    assert isinstance(vna_group, h5py.Group)
    try:
        if "s21_real" in list(vna_group.keys()):
            data = vna_group["s21_real"]
            assert isinstance(data, h5py.Dataset)
            dims = [dim.keys()[0] for dim in data.dims]
        else:
            data = vna_group["s21_mag"]
            assert isinstance(data, h5py.Dataset)
            dims = [dim.keys()[0] for dim in data.dims]
    except KeyError as e:
        print(path)
        raise e
    out["dimensions"] = dims
    file.close()
    return out


class File:
    """
    Defines a loaded HDF file.

    .. note::
        The ``File`` objects are created automatically when creating a
        :class:`Dataset` from a path.

    Parameters
    ----------
    path : str
        Path to the HDF file.
    cryostat_attenuation : float
        Total attenuation present in the cryostat. Must be a negative number.
    additional_params : list of str, optional
        list of additional parameter names to extract from the files.

        .. note::

            If left to ``None`` only those parameters will be extracted:

            - VNA
            - VNA Average
            - VNA Power
            - VNA Bandwidth
            - VNA Frequency
            - Variable Attenuator
            - s21_real
            - s21_imag
            - s21_mag
            - s21_phase
            - Index
            - Magnet
    """

    def __init__(
        self,
        path: str | Path,
        cryostat_attenuation: float,
        additional_params: Optional[list[str]] = None,
    ) -> None:
        self.path = path
        self._additional_params = (
            additional_params if additional_params is not None else []
        )
        if cryostat_attenuation > 0:
            raise ValueError("Attenuation must be negative")
        self.cryostat_attenuation = cryostat_attenuation
        self._file_dict = _read_hdf(path, self._additional_params)
        self.infos = self._file_dict["attributes"]

        # Declare all possible parameters and s21_* parameters.
        self.vna_average = NullParameter()
        self.vna_bandwidth = NullParameter()
        self.vna_frequency = NullParameter()
        self.vna_power = NullParameter()
        self.variable_attenuator = NullParameter()
        self.magnet = NullParameter()
        self.index = NullParameter()
        self.voltage_bias = NullParameter()
        self.s21_mag = NullParameter()
        self.s21_phase = NullParameter()
        self.s21_real = NullParameter()
        self.s21_imag = NullParameter()

        self._populate_params()

        # Get the shape from one of the data arrays
        self.shape = tuple(self.s21_real.range.shape)
        self._dimensions = self._file_dict["dimensions"]
        self._reshape_params()

    def _populate_params(self) -> None:
        """
        Replaces the NullParameters for Parameters when they exist in the file
        """
        for key, value in self._file_dict["datasets"].items():
            if len(value.keys()) == 1 and list(value.keys()) == [key]:
                attribute = key.lower().replace(" ", "_")
                parameter = Parameter(
                    value[key]["values"],
                    key,
                    value[key]["description"],
                    value[key]["unit"],
                )
                self.with_param(attribute, parameter)
            elif list(value.keys()) == ["values", "description", "unit"]:
                attribute = key.lower().replace(" ", "_")
                parameter = Parameter(
                    value["values"], key, value["description"], value["unit"]
                )
                self.with_param(attribute, parameter)
            elif key == "VNA":
                self.with_param(
                    "vna_frequency",
                    Parameter(
                        value["VNA Frequency"]["values"],
                        "VNA Frequency",
                        value["VNA Frequency"]["description"],
                        value["VNA Frequency"]["unit"],
                    ),
                )
                if "s21_real" in value.keys():
                    self.with_param(
                        "s21_real",
                        Parameter(
                            value["s21_real"]["values"],
                            "s21_real",
                        ),
                    )
                    self.with_param(
                        "s21_imag",
                        Parameter(
                            value["s21_imag"]["values"],
                            "s21_imag",
                        ),
                    )
                    mag, phase = convert_complex_to_magphase(
                        value["s21_real"]["values"], value["s21_imag"]["values"]
                    )
                    self.with_param(
                        "s21_mag",
                        Parameter(mag, "s21_mag", unit="dB"),
                    )
                    self.with_param(
                        "s21_phase", Parameter(phase, "s21_phase", unit="deg")
                    )
                elif "s21_mag" in value.keys():
                    self.with_param(
                        "s21_mag",
                        Parameter(
                            value["s21_mag"]["values"],
                            "s21_mag",
                            unit=value["s21_mag"]["unit"],
                        ),
                    )
                    self.with_param(
                        "s21_phase",
                        Parameter(
                            value["s21_phase"]["values"],
                            "s21_phase",
                            unit=value["s21_phase"]["unit"],
                        ),
                    )
                    real, imag = convert_magphase_to_complex(
                        value["s21_mag"]["values"], value["s21_phase"]["values"]
                    )
                    self.with_param("s21_real", Parameter(real, "s21_real"))
                    self.with_param("s21_imag", Parameter(imag, "s21_imag"))
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

    def _reshape_params(self) -> None:
        """Reshapes all parameters to match data shape."""
        for p in self.list_params():
            if p != "VNA Frequency" and not p.startswith("s21_"):
                param = self.__dict__[p.lower().replace(" ", "_")]
                if param.range.shape == self.shape[:-1]:
                    continue
                else:
                    param.range = _broadcast_along_axis(
                        param.range, self.shape[:-1], self._dimensions.index(p)
                    )

    def with_param(self, attribute: str, parameter: Parameter) -> Self:
        """
        Adds a :class:`Parameter` to a file.

        Parameters
        ----------
        attribute : str
            Attribute name. The Parameter will be called using the
            ``File.attribute`` syntax.
        parameter : :class:`Parameter`
            The parameter to add.
        """
        self.__dict__.update({attribute: parameter})
        return self

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        if not name.startswith("__") and isinstance(value, NullParameter):
            print(
                f"UserWarning: The attribute {name} is not defined for this file! Will return a NullParameter."
            )
        return value

    def list_params(self) -> list[str]:
        """
        Lists available parameter names.
        """
        out = []
        for value in self.__dict__.values():
            if isinstance(value, Parameter) and not isinstance(value, NullParameter):
                out.append(value.name)
        return out

    def __str__(self) -> str:
        out = ""
        if isinstance(self.path, str):
            path_parts = self.path.split("/")
        else:
            path_parts = self.path.parts
        fdl = [part for part in path_parts if "FDL" in part]
        if len(fdl) != 0:
            sample = path_parts[path_parts.index(fdl[0]) - 1]
            out += f" {sample}, {fdl[0]}, "
        out += f"mean frequency = {np.mean(self.vna_frequency.range)}"
        return out

    def __repr__(self) -> str:
        return f"ooragan.File({self.path}, {self.list_params()}, {self.infos})"


def _load_files_from_path(
    path: str | Path,
    cryostat_attenuation: float,
    additional_params: list[str],
) -> list[File]:
    """Loads multiple files from a directory, walking through it."""
    files = []
    for paths, _, _ in os.walk(path):
        glob_out = glob(os.path.join(paths, "*.hdf5"))
        glob_out.sort()
        for file in glob_out:
            file_obj = File(file, cryostat_attenuation, additional_params)
            files.append(file_obj)
    if not files:
        raise RuntimeError("No HDF5 files were found in this directory")
    return files


class Dataset:
    """
    Data container. Contains data and information on measurements saved in HDF5
    files.

    .. attention::
        Supports only data from HDF5 files as of now. For txt files, use the old
        Dataset class located in ooragan.old.Dataset.

    Parameters
    ----------
    path : str, optional
        Path of the folder for multiple data files or for a single data file.
    cryostat_attenuation : float, optional
        Total attenuation present in the cryostat. Must be a negative number.
    files : list of Files, optional
        List of :class:`File` to build a dataset from.
    additional_params : list of str, optional, optional
        list of additional parameter names to extract from the files.

        .. note::

            If left to ``None`` only those parameters will be extracted:

            - VNA
            - VNA Average
            - VNA Power
            - VNA Bandwidth
            - VNA Frequency
            - Variable Attenuator
            - s21_real
            - s21_imag
            - s21_mag
            - s21_phase
            - Index
            - Magnet

    Attributes
    ----------
    files : dict
        Dictionary of the contained files where the keys are the index of the file starting with 0.
    """

    def __init__(
        self,
        path: Optional[str | Path] = None,
        cryostat_attenuation: Optional[float] = None,
        files: Optional[list[File]] = None,
        additional_params: Optional[list[str]] = None,
    ) -> None:
        self.files: dict[str, File] = {}
        if path is not None:
            if cryostat_attenuation is None:
                raise ValueError("Cryostat attenuation must be specified with a path")
            if cryostat_attenuation > 0:
                raise ValueError("Attenuation must be negative")
            self.cryostat_attenuation = cryostat_attenuation
            if Path(path).suffix == "":
                additional_params = (
                    additional_params if additional_params is not None else []
                )
                files_list = _load_files_from_path(
                    path, cryostat_attenuation, additional_params
                )
            else:
                files_list = [File(path, cryostat_attenuation, additional_params)]

            for i, file in enumerate(files_list):
                self.files.update({str(i): file})
        else:
            if files is None:
                raise ValueError(
                    "A list of files must be specified if no path is given"
                )
            att = files[0].cryostat_attenuation
            for file in files:
                if file.cryostat_attenuation != att:
                    raise ValueError(
                        "All attenuation must be equal to create a Dataset from files"
                    )
            self.cryostat_attenuation = att
            for i, file in enumerate(files):
                self.files.update({str(i): file})

    def __getattribute__(self, name: str) -> Any:
        if not name.startswith("__") and re.fullmatch(r"f\d+", name):
            warn("Use the indexing syntax instead", DeprecationWarning, stacklevel=2)
            try:
                return self.files[name.removeprefix("f")]
            except KeyError:
                raise IndexError(f"No file with index {name.removeprefix('f')}")
        return super().__getattribute__(name)

    def __str__(self) -> str:
        out = "Files :"
        for idx, file in self.files.items():
            out += "\n" + idx + ". " + file.__str__()
        return out

    def __repr__(self) -> str:
        return f"ooragan.Dataset({self.files}, {self.cryostat_attenuation})"

    @overload
    def __getitem__(self, index: int) -> File: ...

    @overload
    def __getitem__(self, index: slice) -> list[File]: ...

    def __getitem__(self, index: int | slice) -> File | list[File]:
        """
        Returns the File(s) with the given index (or indices).

        Parameters
        ----------
        index : int or slice
            The index or indices (as a slice) of Files to get from this Dataset.
            If the start or stop of the slice are left empty, will return all Files
            with indices inside the given bounds. This means for returning all Files,
            use slice ``[:]``.

        Returns
        -------
        File or list of File
            If a single index is specified, a single File is returned. A list otherwise.
        """
        total_files = len(self.files.keys())
        if isinstance(index, int):
            if not -total_files <= index <= total_files - 1:
                raise IndexError(
                    "index {} out of bounds for number of files {}".format(
                        index, total_files
                    )
                )
            return self.files[str(index)]
        if isinstance(index, slice):
            if index.start and not -total_files <= index.start <= total_files - 1:
                raise IndexError(
                    "index [{}, {}] out of bounds for number of files {}".format(
                        index.start, index.stop, total_files
                    )
                )
            if index.stop and not -total_files <= index.stop <= total_files - 1:
                raise IndexError(
                    "index [{}, {}] out of bounds for number of files {}".format(
                        index.start, index.stop, total_files
                    )
                )
            idx_list = list(range(*index.indices(total_files)))
            if index.start is None or index.stop is None:
                idx_list = set(idx_list).intersection(map(int, list(self.files.keys())))
            return [self.files[str(i)] for i in idx_list]
        raise TypeError("indices must be int or slice, not {}".format(type(index)))
