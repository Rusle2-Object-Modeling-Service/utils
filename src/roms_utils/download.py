"""For downloading common files for Rusler simulations."""

from io import StringIO
import os
import typing
from tempfile import NamedTemporaryFile

import geopandas as gpd
import libraster as lr
import py3dep
import pygeohydro as pygh
import requests
from osgeo import gdal
from pyproj import Transformer
from shapely.geometry import box

from roms_utils.convert import gdal_to_libraster


def reproject_bbox(
    bbox: tuple[float, float, float, float], src_srs: str, dst_srs: str
) -> list[float]:
    """
    Reproject a bounding box from one coordinate system to another.

    Strings forsrc_srs and dst_srs are expected to be in the format of
    'EPSG:XXXX' where XXXX is the EPSG code. Example: 'EPSG:4326', 'EPSG:3857',
    and 'EPSG:26915'.

    :param bbox: bounding box in the source coordinate system
    :param src_srs: source coordinate system
    :param dst_srs: destination coordinate system
    :return: bounding box in the destination coordinate system
    """
    transformer = Transformer.from_crs(src_srs, dst_srs)
    _bbox = list(bbox)
    _bbox[0], _bbox[1] = transformer.transform(  # pylint: disable=E0633
        bbox[0], bbox[1]
    )
    _bbox[2], _bbox[3] = transformer.transform(  # pylint: disable=E0633
        bbox[2], bbox[3]
    )
    return _bbox


# Soil Data Access (SSURGO)
class SDA:
    """
    Class for querying and downloading data from the SSURGO database.

    Uses the Soil Data Access (SDA) API.
    """

    def __init__(
        self,
        bbox: typing.Tuple[float, float, float, float],
        src: str = "EPSG:4326",
        utm: str = "EPSG:26915",
    ):
        """
        Initialize SDA class.

        :param bbox: Bounding box to query in the source coordinate system
            defined by src. Example: (xmin, ymin, xmax, ymax)
        :param src: Source coordinate system to query in. Default is EPSG:4326.
            Example: 'EPSG:4326' and 'EPSG:3857'.
        :param utm: UTM coordinate system to query in. Default is EPSG:26915.
            Example: 'EPSG:26915'.
        """
        self.bbox = bbox
        self.src_srs = src
        self.utm_srs = utm
        self.url = "https://SDMDataAccess.sc.egov.usda.gov"

    def soils(self) -> gpd.GeoDataFrame:
        """
        Query the SSURGO database for soil data.

        :return: GeoPandas GeoDataFrame of soils
        """
        # URL of spatial service
        url = f"{self.url}/Spatial/SDMWGS84Geographic.wfs?"

        # Make BBOX into supported string format
        bbox = f"{self.bbox[0]},{self.bbox[1]} {self.bbox[2]},{self.bbox[3]}"

        # Request filter expected by SDA
        sda_filter = f"""
            <Filter>
                <BBOX>
                    <PropertyName>Geometry</PropertyName>
                    <Box srsName='{self.src_srs}'>
                        <coordinates>{bbox}</coordinates>
                    </Box>
                </BBOX>
            </Filter>
        """

        # Parameters to be passed to request
        params = {
            "SERVICE": "WFS",
            "VERSION": "1.0.0",
            "REQUEST": "GetFeature",
            "TYPENAME": "MapunitPolyExtended",
            "FILTER": sda_filter,
            "SRSNAME": self.src_srs,
            "OUTPUTFORMAT": "GML2",
        }

        # For soils object to be populated after request
        soils = None

        # Make request
        response = requests.get(url, params=params, timeout=30)
        gml = response.text
        soils = gpd.read_file(StringIO(gml))
        soils = soils.set_crs(self.src_srs)

        if isinstance(soils, gpd.GeoDataFrame):
            # Clip soils geometry by bounding box
            bbox = box(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3])
            soils = soils.clip(bbox)

            # Reproject soils to UTM coordinates
            soils = soils.to_crs(self.utm_srs)

        return soils


# United States Geological Survey
class USGS:
    """
    Class for downloading data from the USGS database.

    Uses 3DEP and NHD services.
    """

    def __init__(
        self,
        bbox: typing.Tuple[float, float, float, float],
        src: str = "EPSG:4326",
        utm: str = "EPSG:26915",
        resolution: int = 10,
    ):
        """
        Initialize USGS class.

        :param bbox: Bounding box to query in the source coordinate system
        :param src: Source coordinate system to query in. Default is EPSG:4326.
            See py3dep for list of supported coordinate systems.
        :param utm: UTM coordinate system to query in. Default is EPSG:26915.
        :param resolution: Resolution of data to download. Default is 10. To
            seea list of available resolutions,
            use py3dep.check_3dep_availability
        """
        self.bbox = bbox
        self.src_srs = src
        self.utm_srs = utm
        self.resolution = resolution

    def elevations(
        self, bbox: tuple[float, float, float, float] | None = None
    ) -> lr.doubleRaster:
        """
        Query the USGS database for elevation data.

        :param bbox: Bounding box to query in the source coordinate system
            defined by self.src_srs. Defaults to None in which this method will
            use the bounding box defined by self.bbox.
            Example: (xmin, ymin, xmax, ymax)
        :return: libraster.doubleRaster raster of elevation data
        """
        # For writing RasterIO raster to file
        temp_rio = NamedTemporaryFile(suffix=".tif", delete=False)
        temp_name_rio = temp_rio.name
        temp_rio.close()

        # Check if user provided custom BBOX to be used instead. If not, use
        # the bbox that the class was initialized with
        if isinstance(bbox, tuple):
            bbox = self.bbox

        dem = py3dep.get_map(
            "DEM",
            geometry=bbox,
            resolution=self.resolution,
            geo_crs=self.src_srs,
            crs=self.src_srs,
        )
        dem.rio.to_raster(temp_name_rio)

        # Open file in GDAL
        data_source = gdal.Open(temp_name_rio)

        # For writing GDAL Warp results to file
        temp_gdal = NamedTemporaryFile(suffix=".asc", delete=False)
        temp_name_gdal = temp_gdal.name
        temp_gdal.close()

        # Warp to UTM and store as AAIGrid raster
        opt = gdal.WarpOptions(
            dstSRS=self.utm_srs,
            xRes=self.resolution,
            yRes=self.resolution,
            dstNodata=-9999,
        )
        gdal.Warp(temp_name_gdal, data_source, options=opt)
        data_source = None

        # Format output raster to be compatible with LibRaster
        data_source = gdal.Open(temp_name_gdal)
        raster = gdal_to_libraster(data_source, fmt="double")
        data_source = None

        # Cleanup temp directory when we're done
        os.remove(temp_name_rio)
        os.remove(temp_name_gdal)
        os.remove(temp_name_gdal.replace(".asc", ".asc.aux.xml"))
        os.remove(temp_name_gdal.replace(".asc", ".prj"))

        return raster

    def elevations_by_watersheds(
        self, huc: str = "huc12"
    ) -> typing.Tuple[typing.Dict[str, lr.doubleRaster], gpd.GeoDataFrame]:
        """
        Download elevations by watershed boundary.

        :param huc: HUC to query. Default is huc12. For list of supported HUCs,
            see pygeohydro WBD class documentation
        :return: Tuple of dictionaries of raster data and GeoDataFrame of
            watersheds that intersect self.bbox
        """
        wts = self.watersheds(huc)
        elevations = {}

        for i, row in wts.iterrows():  # pylint: disable=W0612
            elevations[row["name"]] = self.elevations(row["geometry"])

        return elevations, wts

    def watersheds(self, huc: str = "huc12") -> gpd.GeoDataFrame:
        """
        Query the USGS NHD database for watershed data.

        :param huc: HUC to query. Default is huc12. For list of supported HUCs,
            see pygeohydro WBD class documentation
        :return: gpd.GeoDataFrame of watershed data
        """
        wbd = pygh.WBD(huc)
        wts = wbd.bygeom(geom=self.bbox, geo_crs=self.src_srs)
        return wts


if __name__ == "__main__":
    pass
