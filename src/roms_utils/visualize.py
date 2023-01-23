"""For visualization of common Rusler layers."""

import json
import os
from tempfile import NamedTemporaryFile

import geopandas as gpd
import libraster as lr
import pandas as pd
from osgeo import gdal
from osgeo_utils import gdal_polygonize
from shapely import wkt

from roms_utils.convert import (
    gdal_get_info,
    gdal_to_libraster,
    libraster_get_info,
    libraster_to_gdal,
)

# from .convert import *


def create_output_structure(output: str) -> dict[str, str]:
    """
    Create ouput structure when writing Project.layers to disk.

    :param output: root output directory
    :return: dictionary of output directories by each layer type
    """
    # Setup output structure
    if not os.path.exists(output):
        os.makedirs(output)
    if not os.path.exists(os.path.join(output, "metadata")):
        os.mkdir(os.path.join(output, "metadata"))
    if not os.path.exists(os.path.join(output, "images")):
        os.mkdir(os.path.join(output, "images"))
    if not os.path.exists(os.path.join(output, "vectors")):
        os.mkdir(os.path.join(output, "vectors"))
    if not os.path.exists(os.path.join(output, "rasters")):
        os.mkdir(os.path.join(output, "rasters"))

    return {
        "met_path": os.path.join(output, "metadata"),
        "img_path": os.path.join(output, "images"),
        "ras_path": os.path.join(output, "rasters"),
        "vec_path": os.path.join(output, "vectors"),
    }


class Project:
    """
    Class for defining project structure of layers.

    Needed in visualization of downloaded data in libraster_utils.download.
    This class is used to create a project structure and to generate additional
    layers needed for visualization and simulation.
    """

    def __init__(self, utm: str, resolution: int = 10):
        """
        Initialize Project class.

        :param utm: UTM zone of project given as string. Example: "EPSG:26915"
        :param resolution: resolution of raster in meters from UTM. Default is
            10 meters.
        """
        self.layers: dict[str, dict] = {
            "images": {},
            "rasters": {},
            "vectors": {},
            "metadata": {},
        }
        self.resolution = resolution
        self.srs = utm

    def export(self, output_dir: str):
        """
        Export project to disk.

        :param output_dir: Root output directory to write the project to
        """
        # Setup output directory
        paths = create_output_structure(output_dir)

        # Export layers
        for met in self.layers["metadata"]:
            with open(
                os.path.join(paths["met_path"], f"{met}.json"),
                "w",
                encoding="UTF-8",
            ) as outfile:
                json.dump(
                    self.layers["metadata"][met],
                    outfile,
                    sort_keys=True,
                    indent=4,
                )
        for img in self.layers["images"]:
            gdal.Translate(
                os.path.join(paths["img_path"], f"{img}.png"),
                self.layers["images"][img],
            )
        for ras in self.layers["rasters"]:
            self.layers["rasters"][ras].exportGRIDASCII(
                os.path.join(paths["ras_path"], f"{ras}.asc")
            )
        for vec in self.layers["vectors"]:
            self.layers["vectors"][vec].to_file(
                os.path.join(paths["vec_path"], f"{vec}.geojson"),
                driver="GeoJSON",
            )

    def profile(self):
        """
        Generate visualization and simulation layers for Rusler2D.

        This method requires the following layers to be present in the project:
        channels & directions. These layers are populated by
        Project.surface_drainage.
        """
        # Check to see if the channel and flow direction layers exists within
        # the project layers
        if (
            "channels" in self.layers["rasters"]
            and "directions" in self.layers["rasters"]
        ):
            # Load the rasters needed to find hillslope profiles
            chn = self.layers["rasters"]["channels"]
            fdr = self.layers["rasters"]["directions"]

            # Create blank raster for hillslopes to be populated
            hls = lr.intRaster(
                chn.getRows(),
                chn.getCols(),
                chn.getXLowerLeftCorner(),
                chn.getYLowerLeftCorner(),
                chn.getCellSize(),
                -9999,
                chn.getProjectionString(),
            )

            # For keeping track of all hillslope profiles to be set to layers
            # metadata
            all_hillslopes = []

            # For creating a unique index for each hillslope profile to be set
            # to layers rasters
            hls_num = 0

            # Iterate over channels raster to find all channel cells
            for row in range(chn.getRows()):
                for col in range(chn.getCols()):
                    # If we encounter a valid channel cell
                    if chn.getValue(row, col):
                        # For recording all potential hillslope locations
                        hillslope_locations = []

                        # Check if the surrounding cells feed into the outlet
                        if fdr.getNEValue(
                            row, col
                        ) == lr.southwest and not chn.getNEValue(row, col):
                            hillslope_locations.append([row - 1, col + 1])

                        if fdr.getEValue(
                            row, col
                        ) == lr.west and not chn.getEValue(row, col):
                            hillslope_locations.append([row, col + 1])

                        if fdr.getSEValue(
                            row, col
                        ) == lr.northwest and not chn.getSEValue(row, col):
                            hillslope_locations.append([row + 1, col + 1])

                        if fdr.getSValue(
                            row, col
                        ) == lr.north and not chn.getSValue(row, col):
                            hillslope_locations.append([row + 1, col])

                        if fdr.getSWValue(
                            row, col
                        ) == lr.northeast and not chn.getSWValue(row, col):
                            hillslope_locations.append([row + 1, col - 1])

                        if fdr.getWValue(
                            row, col
                        ) == lr.east and not chn.getWValue(row, col):
                            hillslope_locations.append([row, col - 1])

                        if fdr.getNWValue(
                            row, col
                        ) == lr.southeast and not chn.getNWValue(row, col):
                            hillslope_locations.append([row - 1, col - 1])

                        if fdr.getNValue(
                            row, col
                        ) == lr.south and not chn.getNValue(row, col):
                            hillslope_locations.append([row - 1, col])

                        # Flag for including channel cell in first hillslope
                        first = True

                        # For each valid location
                        for loc in hillslope_locations:
                            # Find subwatershed from the given location
                            sub_wts = lr.findSubWatershed(
                                chn, fdr, loc[0], loc[1]
                            )

                            # Get number of cells in hillslope
                            count = sub_wts.getMax()

                            # For storing (row, col) locations in hillslope
                            hillslope = [None for x in range(count)]

                            # For each cell in the hillslope
                            for w_row in range(sub_wts.getRows()):
                                for w_col in range(sub_wts.getCols()):
                                    # Compute hillslope value and convert to 0
                                    # based index
                                    sub_wts_val = (
                                        sub_wts.getValue(w_row, w_col) - 1
                                    )

                                    # If the current location is part of the
                                    # hillslope
                                    if sub_wts_val >= 0:
                                        # Record the location
                                        hillslope[sub_wts_val] = (w_row, w_col)

                                        # Set value of hillslope index at
                                        # current location
                                        hls.setValue(w_row, w_col, hls_num)

                            # Append channel cell to first valid hillslope
                            if first:
                                hillslope.append((row, col))
                                hls.setValue(row, col, hls_num)
                                first = False

                            # Append hillslopes to list of all hillslopes
                            all_hillslopes.append(hillslope)

                            # Increment hillslope index number
                            hls_num += 1

                        # If we didn't find any neighboring cells that feed in
                        if len(hillslope_locations) == 0:
                            # Then we only need to include the channel cell
                            all_hillslopes.append([(row, col)])
                            hls.setValue(row, col, hls_num)

                            # Still need to increment hillslope index number
                            hls_num += 1

            # Store to project layers
            self.layers["metadata"]["hillslopes"] = all_hillslopes
            self.layers["rasters"]["hillslopes"] = hls
        else:
            # Exception for no channels and direction layers in project
            pass

        return

    def soil(self, soils: gpd.GeoDataFrame):
        """
        Generate soil layers needed for Rusler2D.

        Create unique soil indices for each unique soil type. Stores soil
        raster, vector, and names to project layers.

        :param soils: GeoPandas GeoDataFrame of soils downloaded from
            download.SDA.soils method
        """
        # Assign index as a new column in soils GeoDataFrame
        soils["sid"] = soils.groupby(["areasymbol", "musym"]).ngroup()

        # Get unique soil names and order them into a list by sid. This is to
        # be used in getting the soil name (soils\\usa\\<areasymbol>\\<musym>)
        # when doing a lookup of the correct Rusle2 soils xml
        unique = (
            soils[["sid", "areasymbol", "musym"]]
            .sort_values(by="sid")
            .drop_duplicates()
        )
        names = [
            f"soils\\usa\\{asb}\\{ms}"
            for asb, ms in zip(unique["areasymbol"], unique["musym"])
        ]

        # Open soil GeoDataFrame in memory with GDAL format
        soils_vector = gdal.OpenEx(soils.to_json())

        # Rasterize soils vector using GDAL
        vsi_soils = "/vsimem/soils.tif"

        opt = gdal.RasterizeOptions(
            format="GTiff",
            xRes=self.resolution,
            yRes=self.resolution,
            # outputBounds=self.bbox,
            attribute="sid",
            outputSRS=self.srs,
            noData=-9999,
            outputType=gdal.gdalconst.GDT_Int32,
        )
        gdal_soils = gdal.Rasterize(vsi_soils, soils_vector, options=opt)

        # Convert to LibRaster format
        soils_raster = gdal_to_libraster(gdal_soils)

        # Clean up memory in GDAL
        gdal.Unlink(vsi_soils)

        # Store to internal layers object
        self.layers["metadata"]["soils"] = names
        self.layers["rasters"]["soils"] = soils_raster
        self.layers["vectors"]["soils"] = soils

        return

    def surface_drainage(
        self,
        chn: lr.boolRaster,
        fac: lr.intRaster,
        fac_cmap: str,
        fdr: lr.intRaster,
        fdr_cmap: str,
        include_watershed: bool = False,
    ):
        """
        Generate surface drainage layers needed for Rusler2D.

        :param chn: Channel raster
        :param fac: Accumulation raster
        :param fac_cmap: Accumulation raster color map. For an example of a
            color map, see cmap.py
        :param fdr: Flow direction raster
        :param fdr_cmap: Flow direction raster color map. For an example of a
            color map, see cmap.py
        :param include_watershed: Flag to mark that catchment boundaries should
            be included in the project layers. Default is False.
        """
        # Create accumulation image
        vsi_fac = "/vsimem/fac.png"
        gdal_fac = libraster_to_gdal(fac, "float")
        options = gdal.DEMProcessingOptions(
            colorFilename=fac_cmap, format="PNG", addAlpha=True
        )
        fac_png = gdal.DEMProcessing(
            destName=vsi_fac,
            srcDS=gdal_fac,
            processing="color-relief",
            options=options,
        )

        # Create direction image
        vsi_fdr = "/vsimem/fdr.png"
        gdal_fdr = libraster_to_gdal(fdr, "int")
        options = gdal.DEMProcessingOptions(
            colorFilename=fdr_cmap,
            format="PNG",
            addAlpha=True,
            colorSelection="nearest_color_entry",
        )
        fdr_png = gdal.DEMProcessing(
            destName=vsi_fdr,
            srcDS=gdal_fdr,
            processing="color-relief",
            options=options,
        )

        # Begin channel network related geometries:
        # Create lr networks object to obtain link information
        lrn = lr.LRNetworks(chn, fdr, False)

        # For saving last outlet in each network to create watersheds
        outlet: lr.location

        # For drawing channels as vector
        chn_paths = []
        chn_lengths = []

        # For tracking link numbers
        chn_links = []
        chn_links_ds = []

        # For creating points at each outlet and generating watersheds
        chn_outlets = []
        chn_outlets_row = []
        chn_outlets_col = []

        # For creating points at each source
        chn_sources = []
        chn_sources_row = []
        chn_sources_col = []

        # For tracking network numbers at each link
        chn_networks = []

        # Basic raster information for converting coordinates
        info = libraster_get_info(chn)

        # Begin iterating over networks to gather link information
        for i in range(lrn.Size()):
            network = lrn.FindNetwork(i + 1)
            for j in range(network.Size()):
                link = network.FindLink(j + 1)

                # For storing coordinates of each raster cell in link
                link_path = []

                # Get location of source and convert to coordinates
                source = link.GetSourceCell().Location()
                computed_xll = info["xll"] + info["size"] * (source.col + 0.5)
                computed_yll = info["yll"] + info["size"] * (
                    info["rows"] - source.row - 0.5
                )
                point = f"POINT ({computed_xll} {computed_yll})"

                # Add source point
                chn_sources.append(wkt.loads(point))
                chn_sources_row.append(source.row)
                chn_sources_col.append(source.col)

                # For all link cells in-between source and outlet
                for k in range(link.Size()):
                    # Get location of current cell
                    cell = link.FindCell(k + 1)
                    loc = cell.Location()

                    # Convert to coordinates and add to link path
                    computed_xll = info["xll"] + info["size"] * (loc.col + 0.5)
                    computed_yll = info["yll"] + info["size"] * (
                        info["rows"] - loc.row - 0.5
                    )
                    link_path.append(f"{computed_xll} {computed_yll}")

                # Get location of outlet and add to link path
                # NOTE: This has to be done outside of the link iteration above
                #       as the outlet is not included in the link size
                loc = link.GetOutletCell().Location()
                computed_xll = info["xll"] + info["size"] * (loc.col + 0.5)
                computed_yll = info["yll"] + info["size"] * (
                    info["rows"] - loc.row - 0.5
                )
                link_path.append(f"{computed_xll} {computed_yll}")

                # Record downstream link number. If no downstream link, then
                # set to -1 to denote end of network
                data_source = link.DownstreamLink()
                chn_links_ds.append(
                    data_source.Number() if data_source else -1
                )

                # Add downstream link source node to current link path so that
                # visualization does not contain gaps at link junctions
                # NOTE: This must be done after adding the outlet cell to
                #       ensure proper order of points along line
                if data_source:
                    loc = data_source.GetSourceCell().Location()
                    computed_xll = info["xll"] + info["size"] * (loc.col + 0.5)
                    computed_yll = info["yll"] + info["size"] * (
                        info["rows"] - loc.row - 0.5
                    )
                    link_path.append(f"{computed_xll} {computed_yll}")

                # Convert link path to WKT linestring
                linestring = f"LINESTRING ({', '.join(link_path)})"

                # Add linestring to overall channel geometry
                chn_paths.append(wkt.loads(linestring))
                chn_lengths.append(link.Size() * info["size"])

                # Record link and network number at current link
                chn_links.append(j + 1)
                chn_networks.append(i + 1)

                # Record outlet location for current link to keep track of the
                # last outlet in the network (drainage point)
                outlet = loc

            # Add network's outlet to list
            chn_outlets.append(outlet)
            chn_outlets_row.append(outlet.row)
            chn_outlets_col.append(outlet.col)

        # For watershed GeoDataFrame (if include_watershed is True)
        df_wts = None

        # Temporary list to store geometries for each watershed as they are
        # made
        df_wts_a = []

        # For each outlet, create vectors for outlets/sources, create watershed
        # geometry, and compute statistics (if include_watershed is True)
        for i, chn_outlet in enumerate(chn_outlets):
            # for i in range(len(chn_outlets)):
            # Get the current outlet and compute its watershed
            outlet = chn_outlet

            if include_watershed:
                wts = lr.findWatershed(fdr, outlet.row, outlet.col, i + 1)

                # Export to temporary location to be used with gdal_polygonize
                temp = NamedTemporaryFile(suffix=".asc", delete=False)
                temp_name = temp.name
                temp.close()
                wts.exportGRIDASCII(temp_name)

                # Create GeoJSON polygon from watershed raster
                gdal_polygonize.main(
                    [
                        "-8",
                        "-f",
                        "GeoJSON",
                        temp_name,
                        temp_name.replace(".asc", ".geojson"),
                        "watershed",
                        "Network",
                    ]
                )

                # Remove temporary raster
                os.remove(temp_name)
                if os.path.exists(temp_name.replace(".asc", ".prj")):
                    os.remove(temp_name.replace(".asc", ".prj"))

            # Convert the x, y locations to coordinates now that it's safe to
            # do
            computed_xll = info["xll"] + info["size"] * (outlet.col + 0.5)
            computed_yll = info["yll"] + info["size"] * (
                info["rows"] - outlet.row - 0.5
            )
            chn_outlets[i] = wkt.loads(
                f"POINT ({computed_xll} {computed_yll})"
            )

            if include_watershed:
                # Read GeoJSON into GeoDataFrame
                dataframe = gpd.read_file(
                    temp_name.replace(".asc", ".geojson")
                )

                # Count valid cells in watershed (make a libraster function)
                area = 0
                for row in range(wts.getRows()):
                    for col in range(wts.getCols()):
                        if wts.getValue(row, col) != wts.getNODATA():
                            area += 1

                # Compute area of watershed
                area *= wts.getCellSize() ** 2
                dataframe["Area (m^2)"] = pd.Series([area])

                # Append watershed GeoDataFrame to temporary list
                df_wts_a.append(dataframe)

                # Remove temporary GeoJSON file on disk
                os.remove(temp_name.replace(".asc", ".geojson"))

        if include_watershed:
            # Concatenate all watersheds into one
            df_wts = gpd.GeoDataFrame(pd.concat(df_wts_a))

            # Some watersheds may intersect slightly due to lr.streamExtract
            # not following flow directions along map borders. This must be
            # given an insignificant buffer size to avoid overlapping
            # coordinates and dissolved into one geometry per watershed.
            # NOTE: Artifacts in gdal_polygonize may be produced because of LCP
            #       flow directions at the edges of the watershed. This may not
            #       be necessary for other stream instantiation methods
            df_wts["geometry"] = df_wts.buffer(0.01)
            df_wts = df_wts.dissolve(by="Network")
            df_wts = df_wts.set_crs(self.srs)

        # Dynamically get CRS for this
        # Create GeoDataFrame for channel network vector
        df_links = gpd.GeoDataFrame(
            pd.DataFrame(
                {
                    "Network": chn_networks,
                    "Link": chn_links,
                    "Downstream Link": chn_links_ds,
                    "Length (m)": chn_lengths,
                    "Path": chn_paths,
                }
            ),
            geometry="Path",
        ).set_crs(self.srs)

        # Dynamically get CRS for this
        # Create GeoDataFrame for outlet points
        df_outlets = gpd.GeoDataFrame(
            pd.DataFrame(
                {
                    "Network": [i + 1 for i in range(len(chn_outlets))],
                    "Row": chn_outlets_row,
                    "Col": chn_outlets_col,
                    "Point": chn_outlets,
                }
            ),
            geometry="Point",
        ).set_crs(self.srs)

        # Dynamically get CRS for this
        # Create GeoDataFrame for source points
        # NOTE: While source locations are not used to create watersheds like
        #       the outlet points are, these are recorded to allow for
        #       selections of watersheds at stream heads or link intersections
        #       at the user's discretion. To compute all watersheds from source
        #       points now would be costly, time consuming, and ultimately
        #       pointless unless a user had need of them all. Watersheds
        #       created at the outlet are far less in number and much more
        #       descriptive, as to why they're created now.
        df_sources = gpd.GeoDataFrame(
            pd.DataFrame(
                {
                    "Network": chn_networks,
                    "Link": chn_links,
                    "Downstream Link": chn_links_ds,
                    "Row": chn_sources_row,
                    "Col": chn_sources_col,
                    "Point": chn_sources,
                }
            ),
            geometry="Point",
        ).set_crs(self.srs)

        # Unlink all in-memory gdal files used in image generation
        gdal.Unlink(vsi_fdr)
        gdal.Unlink(vsi_fac)

        # Store to self.layers

        # Images
        self.layers["images"]["accumulations"] = fac_png
        self.layers["images"]["directions"] = fdr_png

        # Rasters
        self.layers["rasters"]["channels"] = chn
        self.layers["rasters"]["accumulations"] = fac
        self.layers["rasters"]["directions"] = fdr

        # Vectors
        self.layers["vectors"]["channels"] = df_links
        self.layers["vectors"]["outlets"] = df_outlets
        self.layers["vectors"]["sources"] = df_sources
        if include_watershed:
            self.layers["vectors"]["watersheds"] = df_wts

        return

    def topography(
        self,
        dem: lr.doubleRaster,
        dem_cmap: str,
        slp: lr.doubleRaster,
        slp_cmap: str,
        brightness: float = 1.0,
    ):
        """
        Generate a topography layers.

        Needed to visualize base layers of a Rusler2D simulation.

        :param dem: Digital Elevation Model raster
        :param dem_cmap: Color map to use for DEM
        :param slp: Slope raster
        :param slp_cmap: Color map to use for Slope
        :param brightness: Brightness factor for hillshade overlay. Default is
            1.0 for full brightness
        """
        # Add color to DEM
        vsi_dem = "/vsimem/dem.png"
        gdal_dem = libraster_to_gdal(dem, "float")
        options = gdal.DEMProcessingOptions(
            colorFilename=dem_cmap, format="PNG", addAlpha=True
        )
        dem_png = gdal.DEMProcessing(
            destName=vsi_dem,
            srcDS=gdal_dem,
            processing="color-relief",
            options=options,
        )

        # Create hillshade of DEM
        vsi_hls = "/vsimem/hls.png"
        options = gdal.DEMProcessingOptions(
            format="PNG", addAlpha=True, zFactor=3
        )
        dem_hls = gdal.DEMProcessing(
            destName=vsi_hls,
            srcDS=gdal_dem,
            processing="hillshade",
            options=options,
        )

        # Create grayscale of slope
        vsi_slp = "/vsimem/slp.png"
        gdal_slp = libraster_to_gdal(slp, "float")
        options = gdal.DEMProcessingOptions(
            colorFilename=slp_cmap, addAlpha=True, format="PNG"
        )
        slp_png = gdal.DEMProcessing(
            destName=vsi_slp,
            srcDS=gdal_slp,
            processing="color-relief",
            options=options,
        )

        # Prepare a GDAl GeoTiff object
        driver = gdal.GetDriverByName("GTiff")

        # Get metadata from dem raster
        info = gdal_get_info(gdal_dem)

        # Create GDAL raster in-memory to store elevation shaded relief
        vsi_elv = "/vsimem/elevation.tiff"
        elv_img = driver.Create(
            vsi_elv, info["cols"], info["rows"], 4, gdal.GDT_Byte
        )

        # Create GDAL raster's location information
        geo_transform = (
            info["xll"],
            info["size"],
            0,
            info["yll"],
            0,
            info["size"],
        )

        # Write all metadata
        elv_img.SetGeoTransform(geo_transform)
        elv_img.SetProjection(info["projection"])

        # Extract rgb bands from each image
        dem_bands = [
            dem_png.GetRasterBand(1).ReadAsArray(),
            dem_png.GetRasterBand(2).ReadAsArray(),
            dem_png.GetRasterBand(3).ReadAsArray(),
        ]
        slp_bands = [
            slp_png.GetRasterBand(1).ReadAsArray(),
            slp_png.GetRasterBand(2).ReadAsArray(),
            slp_png.GetRasterBand(3).ReadAsArray(),
        ]

        # Hillshade only has one band
        hls_band = dem_hls.GetRasterBand(1).ReadAsArray()

        # Convert bands to ratio. Hillslope only has one band, so it is done
        # separately
        hls_band = hls_band / 255

        # Convert remaining bands to ratio, and multiply all bands to get final
        # elevation image
        for i in range(3):
            # Ratio
            dem_bands[i] = dem_bands[i] / 255
            slp_bands[i] = slp_bands[i] / 255

            # Overlay by multiplication
            overlay = dem_bands[i] * slp_bands[i] * hls_band * 255 * brightness

            # Once the overlay is finished, write band back to PNG for DEM
            elv_img.GetRasterBand(i + 1).WriteArray(overlay)
            # elv_img.GetRasterBand(i+1).WriteArray(np.flipud(overlay))

        # Add transparency band
        elv_img.GetRasterBand(4).WriteArray(
            dem_png.GetRasterBand(4).ReadAsArray()
        )
        # elv_img.GetRasterBand(4).WriteArray(
        #     np.flipud(dem_png.GetRasterBand(4).ReadAsArray()))

        # Free vsi files used in processing
        gdal.Unlink(vsi_dem)
        gdal.Unlink(vsi_hls)
        gdal.Unlink(vsi_slp)
        gdal.Unlink(vsi_elv)

        # Store to self.layers

        # Images
        self.layers["images"]["elevations"] = dem_png
        self.layers["images"]["hillshade"] = dem_hls
        self.layers["images"]["slopes"] = slp_png

        # Rasters
        self.layers["rasters"]["elevations"] = dem
        self.layers["rasters"]["slopes"] = slp

        return
