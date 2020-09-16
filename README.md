# Reusing wastewater for agricultural irrigation: a Nexus assessment in the North Western Sahara Aquifer System
This is the repository containing the model scripts used for the paper "Reusing wastewater for agricultural irrigation: a Nexus assessment in the North Western Sahara Aquifer System". The entire results of the paper can be replicated by running this scripts.

## Installation instructions
Download or `git clone` this repository and install the required python packages found in the `environment.yml` file. If you use `conda`, create a new environment with the required packages by running the following command in the bash:
```
conda env create -n name-of-environment -f environment.yml
```
After succesfully creating the environment activate it running `conda activate name-of-environment`.

**Note:** replace `name-of-environment` with a name of you choice.
## GIS data preparation
Geospatial characteristics of the NWSAS were obtained from open sources as described in table 1. All data layers were converted into matching units, re-projected into the Sud Algerie Degree projection (ESRI: 102592), re-scaled to the same resolution and, in the case of the groundwater quality layer, interpolated to extend the data to the entire analysed area. Furthermore, all layers were merged into a large data frame.

**Table 1.** Geographic Information System data sources.
Layer|Coverage|Format|Resolution|Year|Source
--|--|--|--|--:|--
Population|Algeria, Tunisia, Libya|raster (tif)|100 m grid cell|2015|[WorldPop](https://dx.doi.org/10.5258/SOTON/WP00645)
Depth to groundwater|Africa|txt table|5 km grid cell|2012|[British Geological Survey](https://www.bgs.ac.uk/research/groundwater/international/africanGroundwater/mapsDownload.html)
Administrative boundaries|Algeria, Tunisia, Libya|shapefile|Level 1 (provinces) polygons|2015|[GADM](https://gadm.org/index.html)
Transboundary aquifers borders|Global|shapefile|Individual polygons|2015|[IGRAC](https://www.un-igrac.org/ggis/transboundary-aquifers-world-map)
Groundwater quality|NWSAS Basin|data points|206 data points|2016|Provided by regional Authorities<sup>1</sup>
Land cover|Africa|raster (tif)|20 m grid cell|2016|[ESA Climate Change Initiative](http://2016africalandcover20m.esrin.esa.int/)
Climate data|Global|raster (tif)|30 arc second, monthly|1970-2000|[WorldClim](http://www.worldclim.org/)

<sup>1</sup> Available upon request to the authors

More details about this process can be found in the respective publication and its supplementary material. In addition, a test dataframe `test_data_10km.csv` with 10X10 km resolution, is provided in this repository for testing purposes.

## Running the model
