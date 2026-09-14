import numpy as np
import pandas as pd
import re
import os

from datetime import datetime
from pathlib import Path
from numpy.typing import NDArray, ArrayLike


def str_to_time(time_string: str) -> float:
    """
    Converts the time format from the MeaVis' HDF5 files into a datetime timestamp.

    Parameters
    ----------
    time_string : str
        Timestamp string from MeaVis.
    """
    fields = dict(re.findall(r"(tm_\w+)=(-?\d+)", time_string))
    dt = datetime(
        year=int(fields["tm_year"]),
        month=int(fields["tm_mon"]),
        day=int(fields["tm_mday"]),
        hour=int(fields["tm_hour"]),
        minute=int(fields["tm_min"]),
        second=int(fields["tm_sec"]),
    )
    return dt.timestamp()


def convert_magphase_to_complex(
    mag: NDArray, phase: NDArray, deg: bool = True, dBm: bool = True
) -> tuple[NDArray, NDArray]:
    r"""
    Converts magnitude and phase data into real and imaginary.

    Parameters
    ----------
    mag : NDArray
        Magnitude array.
    phase : NDArray
        Phase array
    deg : bool, optional
        Set to ``True`` if the phase is in degrees. Defaults to ``True``.
    dBm : bool, optional
        Set to ``True`` if the magnitude is in dBm. Defaults to ``True``.

    Notes
    -----
    This conversion is defined as

    .. math::

        S_{21}^\text{complex} = 10^{\frac{|S_{21}|}{20}}e^{i\phi}

    where the magnitude :math:`|S_{21}|` is in dB and the phase :math:`\phi` is in degrees.
    """
    if deg:
        phase = np.deg2rad(phase)
    if dBm:
        s21_complex = 10 ** (mag / 20) * np.exp(1j * phase)
    else:
        s21_complex = mag * np.exp(1j * phase)

    return s21_complex.real, s21_complex.imag


def convert_complex_to_magphase(
    real: NDArray, imag: NDArray, deg: bool = True
) -> tuple[NDArray, NDArray]:
    r"""
    Converts real and imaginary data into magnitude (dBm) and phase.

    Parameters
    ----------
    real : NDArray
        Real data array.
    imag : NDArray
        Imaginary data array.
    deg : bool, optional
        If ``True`` the phase is returned in degrees. Defaults to ``True``.

    Notes
    -----
    This conversion is defined as

    .. math::

        |S_{21}|=20\cdot\log_{10}\sqrt{\mathrm{Re}(S_{21})^2+\mathrm{Im}(S_{21})^2}

        \phi=\arctan\left(\frac{\mathrm{Im}(S_{21})}{\mathrm{Re}(S_{21})}\right)
    """

    phase = np.angle(real + 1j * imag, deg=deg)
    mag = 20 * np.log10(np.sqrt(real**2 + imag**2))

    return mag, phase


def load_graph_data(path: str) -> dict[str, NDArray]:
    """
    Loads the data saved in csv files by the Grapher objects and returns it
    as a dictionnary with label as key and NDArrays of the data.

    Parameters
    ----------
    path : str
        Complete file file path of the file containing the data.
    """
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    loaded_data = {}
    for label in df.columns.get_level_values(0).unique():
        loaded_data[label] = np.array(df[label]).T
    return loaded_data


def level_phase(phase: ArrayLike, deg: bool = False) -> ArrayLike:
    """
    Levels the phase by substracting the slope.
    """
    phase = np.asarray(phase)
    unwrapped_phase = np.unwrap(phase, 180) if deg else np.unwrap(phase)
    pointA = unwrapped_phase[0]
    pointB = unwrapped_phase[-1]
    slope = np.linspace(pointA, pointB, len(unwrapped_phase))
    return unwrapped_phase - slope


def data_path() -> Path:
    """Helper function to return the path for the data folder."""
    if os.name == "posix":
        if os.path.exists(Path.home() / "data"):
            return (Path.home() / "data").resolve()
        elif os.path.exists(Path.home() / "Data"):
            return (Path.home() / "Data").resolve()
        else:
            raise NotImplementedError(
                "Cannot find data path automatically. Input path manually."
            )
    elif os.name == "nt":
        return Path("D:/")
    else:
        raise NotImplementedError(
            "Cannot find data path automatically. Input path manually."
        )
