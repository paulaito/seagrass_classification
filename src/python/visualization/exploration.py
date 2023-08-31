# Explore datasets
import geopandas as gpd
import rasterio as rio
import numpy as np

# Get % of virtual points from all points
gt = gpd.read_file('C:/Users/pa-ak/OneDrive - Universidade do Algarve/algarve/data/clean/mapping/all-systems/seagrass_quadrats/seagrass_truth_data.gpkg')
gt.columns

alvor = gt[gt['system'] == 'alvor']
arade = gt[gt['system'] == 'arade']
guadiana = gt[gt['system'] == 'guadiana']

alvor_p_virtual = len(alvor[alvor['source']=='virtual'])/len(alvor)
arade_p_virtual = len(arade[arade['source']=='virtual'])/len(arade)
guadiana_p_virtual = len(guadiana[guadiana['source']=='virtual'])/len(guadiana)

# Reproject: 3857 to 4326
ortho = "c:/Users/pa-ak/OneDrive - Universidade do Algarve/CCMAR/Algae/RiaFormosaMapping/wordir/1_datasets/aerial/drones/orthophotomosaics/drone_004_praiafaro_20190505.tif"

from rasterio.warp import calculate_default_transform, reproject, Resampling
import os 

os.environ['CHECK_WITH_INVERT_PROJ'] = 'YES'
ortho_no_tif = ortho.split('.')[0] 
ortho_3857 = "{}_3857.tif".format(ortho_no_tif)

dst_crs = 'EPSG:3857'

with rio.open(ortho) as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rio.open(ortho_3857, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rio.band(src, i),
                destination=rio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)
            
            
# Get drone orthophotos areas

ortho_o = rio.open(ortho_3857)
pixel_area = ortho_o.transform[0] * (-ortho_o.transform[4])

ortho_r = ortho_o.read(1)
pixel_n = np.count_nonzero(ortho_r < 255)

ortho_area_ha = (pixel_area*pixel_n)/10000
print(ortho_area_ha/2)
