# Get binary results and write as one gpkg
import rasterio as rio
import geopandas as gpd
from rasterio import MemoryFile
from rasterio.features import shapes
import numpy as np
import pandas as pd

# Inputs
multiclass_results_tif = './data/processed/RESULTS/ria_formosa/tif/RF_seagrass_ria_formosa_roi.tif'

# Output
seagrass_classified_multi = './data/processed/RESULTS/ria_formosa/RF_multi_seagrass_classification_clean.gpkg'

# Open results
raster_o = rio.open(multiclass_results_tif)
raster_r = raster_o.read()

# Read
image = raster_o.read(1).astype('float32') # first band
results = (
{'properties': {'raster_val': v}, 'geometry': s}
for i, (s, v) 
in enumerate(
    shapes(image, mask=None, transform=raster_o.transform)))           
geoms = list(results)
intertidal_gdf = gpd.GeoDataFrame.from_features(geoms)
intertidal_gdf = intertidal_gdf.loc[intertidal_gdf['raster_val'] == 1.0]
intertidal_gdf = intertidal_gdf.set_crs(4326).to_crs(3763)
intertidal_gdf['habitat_class'] = 'seagrass_intertidal'

subtidal_gdf = gpd.GeoDataFrame.from_features(geoms)
subtidal_gdf = subtidal_gdf.loc[subtidal_gdf['raster_val'] == 2.0]
subtidal_gdf = subtidal_gdf.set_crs(4326).to_crs(3763)
subtidal_gdf['habitat_class'] = 'seagrass_subtidal'

### All in one geopackage
seagrass_gdf = pd.concat([intertidal_gdf, subtidal_gdf]).drop(columns=['raster_val'])
seagrass_gdf['area_m2'] = seagrass_gdf.area

seagrass_gdf.to_file(seagrass_classified_multi, driver = 'GPKG')
