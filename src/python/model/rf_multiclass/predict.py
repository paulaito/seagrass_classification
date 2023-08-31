import pandas as pd
from osgeo import gdal 
import numpy as np
import time
from splitting import features
from hyperparameters_tuning import best_model, time_0

inpRas = "./data/processed/bands_stack/ria_formosa/masked_stack.tif"
outRas = "./data/processed/RESULTS/ria_formosa/RF_seagrass_ria_formosa2.tif"

### 4: PREDICTING ON SATELLITE IMAGE
# Open raster to be predicted
ds = gdal.Open(inpRas, gdal.GA_ReadOnly)
array = ds.ReadAsArray()

# Get raster info
rows = ds.RasterYSize
cols = ds.RasterXSize
bands = ds.RasterCount
geo_transform = ds.GetGeoTransform()
projection = ds.GetProjectionRef()

# Modify structure
array_s = np.stack(array,axis=2)
array_s = np.reshape(array_s, [rows*cols,bands])
test = pd.DataFrame(array_s, columns=features)

y_pred = best_model.predict(test)
del test
classification = y_pred.reshape((rows,cols))
del y_pred

def createGeotiff(outRaster, data, geo_transform, projection):
    # Create a GeoTIFF file with the given data
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    rasterDS = driver.Create(outRaster, cols, rows, 1, gdal.GDT_Int32)
    rasterDS.SetGeoTransform(geo_transform)
    rasterDS.SetProjection(projection)
    band = rasterDS.GetRasterBand(1)
    band.WriteArray(data)
    rasterDS = None


# Export classified image
createGeotiff(outRas,classification,geo_transform,projection)

# Get time(i) and total seconds to apply model
time_i = time.time()

time_to_process = time_i - time_0

print(time_to_process)