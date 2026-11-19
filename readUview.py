# -*- coding: utf-8 -*-
"""
readUview - reader for Elmitec uView image files (.dat).

Copyright 2019 Dr. Helder Marchetto (modernised 2026)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Usage
-----
::

    from readUview import readUviewClass

    reader = readUviewClass()
    img = reader.getImage(r"K:\\Data\\20190507a001.dat")   # numpy array (height, width)
    print(reader)                                          # file name and size

    for p in reader.paramList:                             # all LEEM parameters
        print(p)                                           # {'number', 'name', 'units', 'value'}

    reader.parameters["Start Voltage"]                     # value by name
    reader.getParameter("Start Voltage")                   # same, with a default

    import matplotlib.pyplot as plt
    plt.imshow(img, cmap="gray")
    plt.show()

Short form::

    from readUview import read_uview
    img, params = read_uview("file.dat")

The file layout follows the Elmitec "FileFormats 2017" specification:
file header -> (optional recipe) -> image header -> (optional markup)
-> LEEM parameter block -> image data (16-bit, little endian).
"""

from __future__ import annotations

import bisect
import datetime as _dt
import logging
import os
import struct
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

__all__ = ["readUviewClass", "read_uview", "UviewFileError", "UNITS"]
__version__ = "2.0.0"

log = logging.getLogger(__name__)

PathLike = Union[str, "os.PathLike[str]"]
Param = Dict[str, Any]

FILE_HEADER_SIZE = 104
RECIPE_SIZE = 128
MARKUP_BLOCK = 128
LEEM_DATA_LEGACY_SIZE = 240
TEXT_ENCODING = "cp1252"        # uView runs on Windows; byte 181 (0xB5) = 'µ'

# Units table, page 10 of the "FileFormats 2017" specification
UNITS: Dict[str, str] = {
    "0": "none", "1": "V", "2": "mA", "3": "A", "4": "C",
    "5": "K", "6": "mV", "7": "pA", "8": "nA", "9": "uA",
}

# Devices that exist in the specification but whose layout is not implemented.
# Parsing stops when one of them is found (their length is unknown).
_UNSUPPORTED_DEVICES: Dict[int, str] = {
    102: "Varian controller #1",
    103: "Varian controller #2",
    111: "Phi/Theta",
    112: "Spin",
    113: "FoV rotation (from LEEM presets)",
    114: "Mirror state",
    115: "MCP screen voltage in kV",
    116: "MCP channelplate voltage in kV",
    **{n: "additional gauge" for n in range(120, 131)},
}


class UviewFileError(Exception):
    """The file is not a valid uView file, or it is truncated."""


def _u16(buf: bytes, pos: int) -> int:
    return int.from_bytes(buf[pos:pos + 2], byteorder="little")


def _float(buf: bytes, pos: int) -> float:
    if pos + 4 > len(buf):
        raise UviewFileError(f"Unexpected end of LEEM data at byte {pos}")
    return struct.unpack_from("<f", buf, pos)[0]


def _text(raw: bytes) -> str:
    return raw.decode(TEXT_ENCODING, errors="replace")


class readUviewClass:
    """Reader for uView .dat files.

    ``readUviewClass()`` creates an empty reader; ``getImage(fn)`` loads a file.
    ``readUviewClass(fn)`` loads the file immediately (the image is then in
    ``self.image``).
    """

    def __init__(self, fn: Optional[PathLike] = None) -> None:
        self.fn: Optional[str] = None
        self.fc: bytes = b""
        self.image: Optional[np.ndarray] = None
        self.paramList: List[Param] = []
        if fn is not None:
            self.getImage(fn)

    def __repr__(self) -> str:
        if self.fn is None:
            return "<readUviewClass: no file loaded>"
        return f"Filename = {self.fn}\nimgSize = ({self.imageWidth},{self.imageHeight})"

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def getImage(self, fn: PathLike) -> np.ndarray:
        """Read *fn* and return the image as a numpy array of shape (height, width)."""
        self.fn = os.fspath(fn)
        try:
            with open(self.fn, mode="rb") as file:
                self.fc = file.read()
        except OSError as exc:
            raise UviewFileError(f"Cannot read {self.fn}: {exc}") from exc

        self.fileHeader()
        self.imageHeader()
        self.leemParameters()

        start = self.dataOffset
        n_pixels = self.imageWidth * self.imageHeight
        n_bytes = n_pixels * 2
        if n_pixels == 0:
            raise UviewFileError(f"{self.fn}: image size is 0 x 0")
        if start + n_bytes > len(self.fc):
            raise UviewFileError(
                f"{self.fn}: file is truncated - expected {n_bytes} bytes of image "
                f"data at offset {start}, but only {len(self.fc) - start} are present"
            )
        if start + n_bytes < len(self.fc):
            log.debug("%s: %d extra bytes after the image data ignored",
                      self.fn, len(self.fc) - start - n_bytes)

        self.image = (np.frombuffer(self.fc, dtype="<u2", count=n_pixels, offset=start)
                      .reshape(self.imageHeight, self.imageWidth)
                      .astype(np.uint16))      # writable copy in native byte order
        return self.image

    @property
    def dataOffset(self) -> int:
        """Byte position where the image data start."""
        return (self.headerSize + self.imageHeadersize
                + self.attachedMarkupSize + self.LEEMDataVersion)

    # ------------------------------------------------------------------ #
    # File header
    # ------------------------------------------------------------------ #
    def fileHeader(self, verbose: bool = False) -> None:
        """Read the file header (first 104 bytes, plus 128 bytes recipe if present)."""
        fc = self.fc
        if len(fc) < FILE_HEADER_SIZE:
            raise UviewFileError(f"{self.fn}: file too short for a uView header")

        end_id = fc.find(0, 0, 20)
        self.UK_id = _text(fc[0:end_id if end_id >= 0 else 20])
        if not self.UK_id.startswith("UKSOFT"):
            log.warning("%s: unexpected file id %r - may not be a uView file",
                        self.fn, self.UK_id)

        self.UK_size = _u16(fc, 20)
        self.UK_version = _u16(fc, 22)
        self.UK_bitsPerPixel = _u16(fc, 24)

        if self.UK_version >= 8:
            self.UK_cameraBitsPerPixel = _u16(fc, 26)
            self.MCPDiameterInPixels = _u16(fc, 28)
            self.hBinning = fc[30]
            self.vBinning = fc[31]
        else:
            self.UK_cameraBitsPerPixel = 0
            self.MCPDiameterInPixels = 0
            self.hBinning = 0
            self.vBinning = 0

        if self.UK_version >= 2:
            self.imageWidth = _u16(fc, 40)
            self.imageHeight = _u16(fc, 42)
            self.nrImages = _u16(fc, 44)
        else:
            self.imageWidth = self.imageHeight = self.nrImages = 0

        self.attachedRecipeSize = _u16(fc, 46) if self.UK_version >= 7 else 0
        self.hasRecipe = self.attachedRecipeSize > 0
        self.fixedRecipeSize = RECIPE_SIZE
        self.headerSize = FILE_HEADER_SIZE + (RECIPE_SIZE if self.hasRecipe else 0)

        if self.UK_bitsPerPixel not in (0, 16):
            log.warning("%s: %d bits per pixel - only 16-bit images are supported",
                        self.fn, self.UK_bitsPerPixel)
        if verbose:
            self._print_fields(
                "headerSize UK_id UK_size UK_version UK_bitsPerPixel "
                "UK_cameraBitsPerPixel MCPDiameterInPixels hBinning vBinning "
                "imageWidth imageHeight nrImages attachedRecipeSize hasRecipe")

    # ------------------------------------------------------------------ #
    # Image header
    # ------------------------------------------------------------------ #
    def imageHeader(self, verbose: bool = False) -> None:
        """Read the image header that follows the file header."""
        fc = self.fc
        fp = self.headerSize
        if len(fc) < fp + 28:
            raise UviewFileError(f"{self.fn}: file too short for the image header")

        self.imageHeadersize = _u16(fc, fp)
        self.version = _u16(fc, fp + 2)
        self.colorScaleLow = _u16(fc, fp + 4)
        self.colorScaleHigh = _u16(fc, fp + 6)
        self.imageTime = int.from_bytes(fc[fp + 8:fp + 16], byteorder="little")
        self.maskXShift = _u16(fc, fp + 16)
        self.maskYShift = _u16(fc, fp + 18)
        self.rotateMask = _u16(fc, fp + 20)
        self.attachedMarkupSize = _u16(fc, fp + 22)
        self.hasAttachedMarkup = self.attachedMarkupSize != 0
        if self.hasAttachedMarkup:
            # markup is stored in blocks of 128 bytes
            self.attachedMarkupSize = MARKUP_BLOCK * (self.attachedMarkupSize // MARKUP_BLOCK + 1)
        self.spin = _u16(fc, fp + 24)
        self.LEEMDataVersion = _u16(fc, fp + 26)
        fp += 28

        if self.version > 5:
            if len(fc) < fp + LEEM_DATA_LEGACY_SIZE + 16:
                raise UviewFileError(f"{self.fn}: file too short for the image header")
            self.hasLEEMData = True
            self.LEEMDataStartPos = fp
            self.LEEMData = struct.unpack("240c", fc[fp:fp + LEEM_DATA_LEGACY_SIZE])
            fp += LEEM_DATA_LEGACY_SIZE
            self.appliedProcessing = fc[fp]
            self.grayAdjustZone = fc[fp + 1]
            self.backgroundvalue = _u16(fc, fp + 2)
            self.desiredRendering = fc[fp + 4]
            self.desired_rotation_fraction = fc[fp + 5]
            self.rendering_argShort = _u16(fc, fp + 6)
            self.rendering_argFloat = struct.unpack_from("<f", fc, fp + 8)[0]
            self.desired_rotation = _u16(fc, fp + 12)
            self.rotation_offset = _u16(fc, fp + 14)
            # 4 spare bytes follow
        else:
            self.hasLEEMData = False
            self.LEEMDataStartPos = None
            self.LEEMData = struct.unpack("240c", bytes(LEEM_DATA_LEGACY_SIZE))
            self.appliedProcessing = b""
            self.grayAdjustZone = b""
            self.backgroundvalue = 0
            self.desiredRendering = b""
            self.desired_rotation_fraction = 0
            self.rendering_argShort = 0
            self.rendering_argFloat = 0.0
            self.desired_rotation = 0
            self.rotation_offset = 0
        self.rotaion_offset = self.rotation_offset     # old (misspelt) name

        if verbose:
            self._print_fields(
                "imageHeadersize version colorScaleLow colorScaleHigh imageTime "
                "imageDateTime maskXShift maskYShift rotateMask attachedMarkupSize "
                "hasAttachedMarkup spin LEEMDataVersion")

    @property
    def imageDateTime(self) -> Optional[_dt.datetime]:
        """Acquisition time, if imageTime is a Windows FILETIME (100 ns since 1601)."""
        try:
            return (_dt.datetime(1601, 1, 1, tzinfo=_dt.timezone.utc)
                    + _dt.timedelta(microseconds=self.imageTime // 10))
        except (AttributeError, OverflowError):
            return None

    # ------------------------------------------------------------------ #
    # LEEM parameters
    # ------------------------------------------------------------------ #
    def leemParameters(self, verbose: bool = False) -> None:
        """Read the LEEM parameter block into ``self.paramList``."""
        start = self.headerSize + self.imageHeadersize + self.attachedMarkupSize
        self.leemData = self.fc[start:start + self.LEEMDataVersion + 1]
        self.zeroList = [i for i, b in enumerate(self.leemData) if b == 0 and i > 0]
        self.paramList = []
        if not self.zeroList:
            return

        end = self.zeroList[-1]
        pos = 0
        while pos <= end:
            if self.leemData[pos] == 0xFF:         # end-of-block marker
                break
            try:
                device, new_pos = self.extractLeemParam(pos)
            except (UviewFileError, IndexError, struct.error) as exc:
                log.warning("%s: LEEM data corrupt at byte %d (%s); "
                            "remaining parameters skipped", self.fn, pos, exc)
                break
            if not device or new_pos <= pos:       # unknown/unsupported device
                break
            self.paramList.append(device)
            pos = new_pos

        if verbose:
            for p in self.paramList:
                print(p)

    @property
    def parameters(self) -> Dict[str, Any]:
        """All LEEM parameters as a {name: value} dictionary."""
        return {p["name"]: p["value"] for p in self.paramList}

    def getParameter(self, name: str, default: Any = None) -> Any:
        """Return the value of the parameter *name* (case-insensitive), or *default*."""
        key = name.strip().lower()
        for p in self.paramList:
            if p["name"].strip().lower() == key:
                return p["value"]
        return default

    def getUnits(self, val: str) -> str:
        """Return the unit for the code *val* (page 10 of 'FileFormats 2017')."""
        return UNITS.get(val, "none")

    def _next_zero(self, pos: int) -> int:
        """Position of the first NUL byte at or after *pos*."""
        i = bisect.bisect_left(self.zeroList, pos)
        if i >= len(self.zeroList):
            raise UviewFileError(f"Missing string terminator after byte {pos}")
        return self.zeroList[i]

    def extractLeemParam(self, startPos: int) -> Tuple[Union[Param, int], int]:
        """Decode the parameter at *startPos*.

        Returns (device_dict, position_of_next_parameter), or (0, startPos)
        for devices that are not supported.
        """
        d = self.leemData
        devNr = d[startPos] & 0x7F                 # highest bit is a flag

        if devNr < 100:
            # Normal device: name + one-character unit code, NUL, float value
            endName = self._next_zero(startPos + 1)
            return {
                "number": devNr,
                "name": _text(d[startPos + 1:endName - 1]),
                "units": self.getUnits(_text(d[endName - 1:endName])),
                "value": _float(d, endName + 1),
            }, endName + 5

        if devNr == 100:
            # Mitutoyo micrometer: two floats (X, Y)
            valX = _float(d, startPos + 1)
            valY = _float(d, startPos + 5)
            return {"number": devNr, "name": "Mitutoyo", "units": "mm",
                    "value": f"{valX},{valY}"}, startPos + 9

        if devNr in (101, 110):
            # Field of view (101 = old format, 110 = current): text, NUL, float
            if devNr == 101:
                log.info("%s: old FoV entry (device 101) - not tested", self.fn)
            endName = self._next_zero(startPos + 1)
            return {"number": devNr, "name": "FOV=" + _text(d[startPos + 1:endName]),
                    "units": "", "value": _float(d, endName + 1)}, endName + 5

        if devNr == 104:
            # Camera exposure: float (s), averaging type, number of images
            value = _float(d, startPos + 1)
            avg_type, n_avg = d[startPos + 5], d[startPos + 6]
            if avg_type == 255:
                avg = "(sliding average)"
            elif avg_type == 0:
                avg = "(average off)"
            else:
                avg = f"(average {n_avg} images)"
            return {"number": devNr, "name": f"Camera exposure {avg}",
                    "units": "seconds", "value": value}, startPos + 7

        if devNr == 105:
            # Title: NUL-terminated text
            endName = self._next_zero(startPos + 1)
            return {"number": devNr, "name": "Title", "units": "none",
                    "value": _text(d[startPos + 1:endName])}, endName + 1

        if 106 <= devNr <= 109:
            # Generic device: name NUL units NUL float
            endName = self._next_zero(startPos + 1)
            endUnits = self._next_zero(endName + 1)
            return {"number": devNr, "name": _text(d[startPos + 1:endName]),
                    "units": _text(d[endName + 1:endUnits]),
                    "value": _float(d, endUnits + 1)}, endUnits + 5

        name = _UNSUPPORTED_DEVICES.get(devNr, "unknown device")
        log.warning("%s: device %d (%s) is not supported; remaining LEEM "
                    "parameters skipped", self.fn, devNr, name)
        return 0, startPos

    # ------------------------------------------------------------------ #
    def _print_fields(self, names: str) -> None:
        for name in names.split():
            print(f"{name} = {getattr(self, name)}")


def read_uview(fn: PathLike) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read a uView file; return (image, {parameter name: value})."""
    reader = readUviewClass()
    img = reader.getImage(fn)
    return img, reader.parameters


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if len(sys.argv) < 2:
        sys.exit("usage: python readUview.py FILE.dat [FILE2.dat ...]")
    for path in sys.argv[1:]:
        r = readUviewClass(path)
        print(r)
        for p in r.paramList:
            print(f"  {p['number']:>3}  {p['name']:<35} {p['value']!s:>14} {p['units']}")
