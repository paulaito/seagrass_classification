# Get area per class from classification results
# (only for comparison purposes)
import rasterio as rio
import geopandas as gpd
import numpy as np

# Inputs
# Ria Formosa results (multiclass)
ria_formosa_results_tif = './data/processed/RESULTS/ria_formosa/tif/RF_seagrass_ria_formosa_roi.tif'
ria_formosa_results_clean_gpkg = './data/processed/RESULTS/ria_formosa/RF_multi_seagrass_classification_clean.gpkg'

# CLASS AREAS FROM TIF IMAGES

# Pixel resolution and area
pixel_resol_mts = 3.2
pixel_area_m2 = pixel_resol_mts**2

# Open raster
with rio.open(ria_formosa_results_tif) as src:
    raster_r = src.read()

# Count intertidal and subtidal pixels
sg_intertidal_pixels = np.count_nonzero(raster_r == 1)
sg_subtidal_pixels = np.count_nonzero(raster_r == 2)

sg_intertidal_ha = (pixel_area_m2*sg_intertidal_pixels)/10000
sg_subtidal_ha = (pixel_area_m2*sg_subtidal_pixels)/10000
sg_ha = sg_intertidal_ha + sg_subtidal_ha

print("TIF\n- Seagrass intertidal area: {} ha\n- Seagrass subtidal area: {} ha\n- Total seagrass area: {} ha".format(sg_intertidal_ha,sg_subtidal_ha, (sg_intertidal_ha + sg_subtidal_ha)))

# GPKG seagrass data
clean_seagrass = gpd.read_file(ria_formosa_results_clean_gpkg)
sum(clean_seagrass.area/10000)

seagrass_clean_intertidal = clean_seagrass.loc[clean_seagrass['habitat_class'] == 'seagrass_intertidal']
seagrass_clean_subtidal = clean_seagrass.loc[clean_seagrass['habitat_class'] == 'seagrass_subtidal']

sg_clean_intertidal_ha = sum(seagrass_clean_intertidal.area)/10000
sg_clean_subtidal_ha = sum(seagrass_clean_subtidal.area)/10000
sg_clean_ha = sg_clean_intertidal_ha + sg_clean_subtidal_ha

print("Seagrass intertidal area: {} ha\n- Seagrass subtidal area: {} ha\n- Total seagrass area: {} ha".format(sg_clean_intertidal_ha,sg_clean_subtidal_ha))
