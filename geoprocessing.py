'''
Created by the authors.
@wallissoncarvalho
@machadoyang
'''

import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping
import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_from_mask(mask_dir,raster_dir):
    shapefile = gpd.read_file(mask_dir)
    Name = []
    elev_mean = []
    elev_std = []
    for i in tqdm(range(len(shapefile.geometry))):
        Name.append(shapefile.Name[i])
        # extract and transform the geometry into GeoJSON format
        geoms = [mapping(shapefile.geometry.values[i])]
        # extract the raster values values within the polygon 
        with rasterio.open(raster_dir) as src:
         out_image, out_transform = mask(src, geoms, crop=True)
        # no data values of the original raster
        no_data=src.nodata
        data=out_image[0]
        row, col = np.where(data != no_data) 
        elev = np.extract(data != no_data, data)
        elev_mean.append(elev.mean())
        elev_std.append(elev.std())
    df = pd.DataFrame({'Name':Name,'Elev_Mean':elev_mean,'Elev_std':elev_std})
    return df


def physiographic_data(watersheds_dir,longestflowpath_dir, raster_dir):
    watersheds = gpd.read_file(watersheds_dir)
    watersheds.index = watersheds.HydroID
    watersheds = watersheds[['Name','AreaKm2','PerimetKm']]
    longestflowpath = gpd.read_file(longestflowpath_dir)
    longestflowpath.index =longestflowpath.DrainID
    longestflowpath = longestflowpath[['Slp1085','LengthKm']]
    longestflowpath['Slp1085']=longestflowpath['Slp1085']*100
    df = pd.concat([watersheds,longestflowpath],axis=1)
    df.index = df.Name
    df['KF'] = df['AreaKm2']/(df['LengthKm']**2)
    df['KC'] = df['PerimetKm']*0.28/(df['AreaKm2']**0.5)
    df.pop('Name')
    df.pop('LengthKm')
    df.pop('PerimetKm')
    df_raster = extract_from_mask(watersheds_dir,raster_dir)
    df_raster.index = df_raster.Name
    df_raster = df_raster[['Elev_Mean','Elev_std']]
    df = pd.concat([df,df_raster],axis=1)
    return df
