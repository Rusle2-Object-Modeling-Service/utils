"""Convert between common raster types."""
import typing
from io import StringIO

import libraster as lr
import numpy as np
from osgeo import gdal


def gdal_get_info(gdal_raster: gdal.Dataset) -> typing.Dict:
    """
    Take in a GDAL raster in memory and return header information.

    Pertains to the raster size and spatial reference system.

    :param gdal_raster: GDAL raster in memory such as one from gdal.Open or
        from libraster_to_gdal
    :return: Dictionary of raster's information as it pertains to SRS. Contains
        number of columns, number of rows, lower left x and y coordinate, cell
        size (resolution), NODATA value, and the projection string
    """
    # Dimensions are pulled from GDAL raster size
    cols = gdal_raster.RasterXSize
    rows = gdal_raster.RasterYSize

    # Returns array with reference coordinates, skew, and resolution
    # NOTE: Check to make sure xres and yres are the same
    geo_transform = gdal_raster.GetGeoTransform()

    # Get NODATA value from GDAL raster
    nodata = gdal_raster.GetRasterBand(1).GetNoDataValue()

    # Get projection as string from GDAL raster
    projection = gdal_raster.GetProjection()

    # Construct return object
    return {
        "cols": cols,
        "rows": rows,
        "xll": geo_transform[0],
        "yll": geo_transform[3] - (geo_transform[1] * rows),
        "size": geo_transform[1],
        "nodata": nodata,
        "projection": projection,
    }


def gdal_to_libraster(
    gdal_raster: gdal.Dataset,
    flip_rows: bool = False,
    fmt: str = "int",
    precision: int = 5,
    info: typing.Dict | None = None,
) -> typing.Union[lr.intRaster, lr.doubleRaster]:
    """
    Convert GDAl raster in memory to libraster format.

    :param gdal_raster: GDAL raster in memory such as one from gdal.Open or
        from libraster_to_gdal
    :param flip_rows: Flag to flip the raster data by rows. Not needed in most
        cases. Defaults to False
    :param fmt: Type of libraster raster to be created. Defaults to int.
        Available options are "int" and "double"
    :param precision: Level of precision to be used after the decimal point for
        double fmt
    :param info: Dictionary containing header and spatial reference system
        information for the GDAL raster such as that returned by gdal_get_info.
        If no value is provided, info will be gathered from gdal_get_info
    :return: Either a libraster intRaster or doubleRaster
    """
    if not info:
        info = gdal_get_info(gdal_raster)
    numpy_raster = gdal_to_numpy(gdal_raster, flip_rows)
    return numpy_to_libraster(numpy_raster, info, fmt, precision)


def gdal_to_numpy(
    gdal_raster: gdal.Dataset, flip_rows: bool = False
) -> np.ndarray:
    """
    Convert GDAL raster in memory to numpy ndarray.

    This causes a loss of rasterheader information. It is recommended to
    preserve the header informationwith a call to gdal_get_info if using this
    function by itself.

    :param gdal_raster: GDAL raster in memory such as one from gdal.Open or
        from libraster_to_gdal
    :param flip_rows: Flag to flip the raster data by rows. Not needed in most
        cases
    :return: Numpy ndarray with raster row/column data
    """
    # Return GDAL raster as numpy ndarray
    numpy_raster = gdal_raster.ReadAsArray()

    # GDAL rasters stored in memory buffers have their rows flipped. Undo this
    # for correct orientation
    if flip_rows:
        numpy_raster = np.flipud(numpy_raster)

    return numpy_raster


def gdal_to_postgis():
    """Needs to be implemented and accept gdal_raster."""
    return


def libraster_get_info(
    raster: typing.Union[lr.intRaster, lr.doubleRaster, lr.boolRaster]
) -> typing.Dict:
    """
    Get header information from libraster raster.

    :param raster: libraster raster of type intRaster, doubleRaster, or
        boolRaster
    :return: Dictionary of raster's information as it pertains to SRS. Contains
        number of columns, number of rows, lower left x and y coordinate, cell
        size (resolution), NODATA value, and the projection string
    """
    return {
        "cols": raster.getCols(),
        "rows": raster.getRows(),
        "xll": raster.getXLowerLeftCorner(),
        "yll": raster.getYLowerLeftCorner(),
        "size": raster.getCellSize(),
        "nodata": raster.getNODATA(),
        "projection": raster.getProjectionString(),
    }


def libraster_to_numpy(
    raster: typing.Union[lr.intRaster, lr.doubleRaster, lr.boolRaster]
) -> np.ndarray:
    """
    Convert libraster raster to numpy ndarray.

    This causes a loss of raster header information. It is recommended to
    preserve the header information with a call to libraster_get_info if using
    this function by itself.

    :param raster: libraster raster of type intRaster, doubleRaster, or
        boolRaster
    :return: Numpy ndarray with raster row/column data
    """
    # Write libraster to string and store in in-memory file
    raster_io = StringIO(raster.writeString())

    # Generate numpy ndarray, skip raster headers
    return np.genfromtxt(raster_io, skip_header=6)


def libraster_to_gdal(
    raster: typing.Union[lr.intRaster, lr.doubleRaster, lr.boolRaster],
    fmt: str = "int",
) -> gdal.Dataset:
    """
    Convert libraster raster to GDAL raster in memory.

    :param raster: libraster raster of type intRaster, doubleRaster, or
        boolRaster
    :param fmt: Type of GDAL raster to be created. Defaults to int.
    :return: GDAL raster in memory
    """
    numpy_raster = libraster_to_numpy(raster)
    return numpy_to_gdal(numpy_raster, libraster_get_info(raster), fmt)


def numpy_to_libraster(
    numpy_raster: np.ndarray,
    info: typing.Dict,
    fmt: str = "int",
    precision: int = 5,
) -> typing.Union[lr.intRaster, lr.doubleRaster, lr.boolRaster]:
    """
    Convert numpy ndarray to libraster raster.

    :param numpy_raster: Numpy ndarray with raster row/column data
    :param info: Dictionary containing header and spatial reference system
        information for the ndarray data such as that returned by gdal_get_info
    :param fmt: Type of libraster raster to be created. Defaults to int.
    :param precision: Number of decimal places to use when writing doubleRaster
        data. Defaults to 5.
    :return: libraster raster of type intRaster, doubleRaster, or boolRaster
    """
    # Open Python in-memory file for writing numpy raster
    raster_io = StringIO()

    # Convert Numpy NaN to NODATA value
    if np.isnan(numpy_raster).any():
        info["nodata"] = -9999
        numpy_raster[np.isnan(numpy_raster)] = info["nodata"]

    # Create blank Libraster raster based on format argument
    if fmt == "double":
        info["nodata"] = float(info["nodata"])
        fmt = f"%1.{precision}f"
        raster = lr.doubleRaster()
    elif fmt == "int":
        info["nodata"] = int(info["nodata"])
        fmt = "%i"
        raster = lr.intRaster()

    # Write numpy raster to in-memory file
    np.savetxt(raster_io, numpy_raster, delimiter="\t", fmt=fmt)

    # Reset cursor on in-memory file
    raster_io.seek(0)

    # Populate blank raster
    raster.readString(
        f"ncols\t{info['cols']}\n"
        f"nrows\t{info['rows']}\n"
        f"xllcorner\t{info['xll']}\n"
        f"yllcorner\t{info['yll']}\n"
        f"cellsize\t{info['size']}\n"
        f"NODATA_value\t{info['nodata']}\n"
        f"{raster_io.read()}"
    )
    raster.setProjectionString(info["projection"])

    # Close in-memory file when done
    raster_io.close()

    return raster


def numpy_to_gdal(
    numpy_raster: np.ndarray, info: typing.Dict, fmt: str = "int"
) -> gdal.Dataset:
    """
    Convert numpy ndarray to GDAL raster in memory.

    :param numpy_raster: Numpy ndarray with raster row/column data
    :param info: Dictionary containing header and spatial reference system
        information for the ndarray data such as that returned by gdal_get_info
    :param fmt: Type of GDAL raster to be created. Defaults to int.
    :return: GDAL raster in memory
    """
    # Prepare a GDAl GeoTiff object
    driver = gdal.GetDriverByName("GTiff")

    # Determine GDAL type by passed format
    if fmt == "int":
        fmt = gdal.GDT_Int32
    elif fmt == "float":
        fmt = gdal.GDT_Float64

    # Create GDAL raster in-memory
    vsi_path = "/vsimem/raster.tiff"
    gdal_raster = driver.Create(vsi_path, info["cols"], info["rows"], 1, fmt)

    # Create GDAL raster's location information
    geo_transform = (
        info["xll"],
        info["size"],
        0,
        info["yll"],
        0,
        info["size"],
    )

    # Write all metadata and values to GeoTiff
    gdal_raster.SetGeoTransform(geo_transform)
    gdal_raster.SetProjection(info["projection"])
    gdal_raster.GetRasterBand(1).WriteArray(numpy_raster)
    # gdal_raster.GetRasterBand(1).WriteArray(np.flipud(numpy_raster))
    gdal_raster.GetRasterBand(1).SetNoDataValue(info["nodata"])

    # Free up file in memory
    gdal.Unlink(vsi_path)

    return gdal_raster


def postgis_to_gdal(row: typing.Tuple) -> gdal.Dataset:
    """
    Convert PostGIS row to GDAL raster in memory.

    :param row: PostGIS row containing raster data
    :return: GDAL raster in memory
    """
    # Create path for virtual memory file
    vsi_path = "/vsimem/temp.tiff"

    # Load return result of PostGIS raster query into GDAL virtual memory file
    gdal.FileFromMemBuffer(vsi_path, bytes(row))

    # Open raster from GDAL virtual memory file
    gdal_raster = gdal.Open(vsi_path)

    # Close virtual memory file when done
    gdal.Unlink(vsi_path)

    return gdal_raster


def postgis_to_libraster(
    row: typing.Tuple, fmt: str = "int", precision: int = 5
) -> typing.Union[lr.intRaster, lr.doubleRaster]:
    """
    Convert PostGIS row to libraster raster in memory.

    :param row: PostGIS row containing raster data
    :param fmt: Type of libraster raster to be created. Defaults to int.
    :param precision: Number of decimal places to use when writing doubleRaster
        data. Defaults to 5.
    :return: libraster raster of type intRaster, doubleRaster, or boolRaster
    """
    # Convert DB return result to GDAL raster
    gdal_raster = postgis_to_gdal(row)

    # Get raster info from GDAL
    info = gdal_get_info(gdal_raster)

    # Convert GDAL raster to numpy ndarray
    numpy_raster = gdal_to_numpy(gdal_raster, True)

    # Get Libraster object from numpy ndarray
    raster = numpy_to_libraster(numpy_raster, info, fmt, precision)

    return raster
