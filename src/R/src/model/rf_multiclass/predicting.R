# Predicting: classifying seagrass habitats
library(raster)

# Inputs
model <- readRDS('./R/model/RF_ria_formosa_model.RDS')
ria_formosa_raster <- raster::brick('./data/processed/bands_stack/ria_formosa/masked_stack.tif')

# Outputs
seagrass_classified <- './data/processed/RESULTS/ria_formosa/tif/RF_seagrass_ria_formosa.tif'

# Name bands and predict
names(ria_formosa_raster) <- c('blue', 'green', 'red', 'nir', 'ndvi', 'dem')
pr <- predict(ria_formosa_raster, model, type ='raw') 

# Export predict result as raster
predictions <- factor(pr, levels = c('0','1','2'))
map <- raster(ria_formosa_raster)

map <- setValues(map, as.matrix(pr))

raster::writeRaster(
  map, 
  filename = seagrass_classified,
  overwrite = TRUE)
