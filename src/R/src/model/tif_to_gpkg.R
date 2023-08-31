# Convert tif as geopackage
library(terra)
library(dplyr)

raster <- rast('./data/processed/RESULTS/ria_formosa/tif/RF_seagrass_ria_formosa_R.tif')
out_vector <- './data/processed/RESULTS/ria_formosa/RF_seagrass_ria_formosa_R.gpkg'

vector <- as.polygons(raster, aggregate=TRUE, round=TRUE, na.rm=TRUE)

df <- as.data.frame(vector)

df %>%
  mutate(habitat_class = case_when(
    RF_seagrass_ria_formosa_R == 0 ~ NA,
    (RF_seagrass_ria_formosa_R == 1) ~ "seagrass_intertidal",
    (RF_seagrass_ria_formosa_R == 2) ~ "seagrass_subtidal"
  )) %>% vect()

writeVector(vector, out_vector)

