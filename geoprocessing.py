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
import os
import xarray as xr 
from shapely.geometry import Point
from osgeo import ogr

watersheds_dir = r'shapefiles\Watersheds.shp'
longestflowpath_dir=r'shapefiles\LongestFlowPath.shp'
raster_dir = r'raster/mdt_23s.tif'


def physiographic_data(watersheds_dir,longestflowpath_dir, raster_dir):
    watersheds = gpd.read_file(watersheds_dir)
    watersheds.index = watersheds.HydroID
    watersheds = watersheds[['Name','AreaKm2','PerimeterK']]
    longestflowpath = gpd.read_file(longestflowpath_dir)
    longestflowpath.index =longestflowpath.DrainID
    longestflowpath = longestflowpath[['Slp1085','LengthKm']]
    df = pd.concat([watersheds,longestflowpath],axis=1)
    df.index = df.Name
    df['KF'] = df['AreaKm2']/(df['LengthKm']**2)
    df['KC'] = df['PerimeterK']*0.28/(df['AreaKm2']**0.5)
    df.pop('Name')
    df.pop('LengthKm')
    df.pop('PerimeterK')
    df_raster = extract_from_mask(watersheds_dir,raster_dir)
    df_raster.index = df_raster.Name
    df_raster = df_raster[['Elev_Mean','Elev_std']]
    df = pd.concat([df,df_raster],axis=1)
    return df



def extract_from_mask(mask_dir,raster_dir):
    shapefile = gpd.read_file(mask_dir)
    Name = []
    elev_mean = []
    elev_std = []
    for i in range(len(shapefile.geometry)):
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
        print(shapefile.Name[i])
    df = pd.DataFrame({'Name':Name,'Elev_Mean':elev_mean,'Elev_std':elev_std})
    return df

def data_netcdf(net_cdf_dir,shape_geometry,buffer=0):
    dados=xr.open_dataset(net_cdf_dir)
    dados=dados.dropna(dim='latitude',how='all')
    dados=dados.dropna(dim='longitude',how='all')
    latitudes=dados.coords['latitude'].values
    longitudes=dados.coords['longitude'].values
    if longitudes.max()>180:
        dados['longitude'] = ((dados.longitudes + 180) % 360 - 180).sortby(dados.longitudes)    
    x,y=[],[]
    for lati in latitudes:
        longitudes=dados.sel(latitude=lati)              
        longitudes=longitudes.dropna(dim='longitude',how='all')
        longitudes=longitudes.coords['longitude'].values
        for long in longitudes:
            x.append(long)
            y.append(lati)    
    pontos=[Point(x) for x in zip(x,y)]
    crs={'proj':'latlong','ellps':'WGS84','datum':'WGS84','no_def':True} #SC WGS 8
    pixels=gpd.GeoDataFrame(crs=crs,geometry=pontos)            
    pixels_bacia=pixels[pixels.geometry.within(shape_geometry.buffer(buffer))]
    buffer=0
    flag=0
    while len(pixels_bacia)==0:
        buffer+=.015
        pixels_bacia=pixels[pixels.geometry.within(shape_geometry.buffer(buffer))]
        flag=1
    if flag==1 and len(pixels_bacia)>0:
        print(len(pixels_bacia))
    pixels=pixels_bacia
    pixels['Latitude']=pixels.geometry.y
    pixels['Longitude']=pixels.geometry.x
    x,y=[],[]
    serie=pd.DataFrame()
    for pixel in pixels.geometry:
        lat=pixel.y
        lon=pixel.x
        x.append(lon)
        y.append(lat)
        serie=pd.concat([serie,dados.sel(latitude=pixel.y,longitude=pixel.x).to_dataframe()])
    serie=serie.groupby(serie.index).mean()
    return pixels,serie

def data_netcdf_concat(net_cdf_folder,shape_geometry,var):
    serie_completa=pd.DataFrame()
    for file in os.listdir(net_cdf_folder):
        if file[0:len(var)]==var:
            net_cdf_dir = net_cdf_folder+file
            pixels,serie = data_netcdf(net_cdf_dir,shape_geometry)
            serie_completa=pd.concat([serie_completa,serie])
    serie_completa=serie_completa[var].to_frame()
    return serie_completa

def mean_prec_watersheds(net_cdf_folder,shape_dir,var):
    shapefile = gpd.read_file(shape_dir).to_crs({'proj':'latlong','ellps':'WGS84','datum':'WGS84','no_def':True})
    Name = []
    Mean_Prec =[]
    for i in range(len(shapefile.geometry)):
        print(shapefile.Name[i])
        Name.append(shapefile.Name[i])
        prec = data_netcdf_concat(net_cdf_folder,shapefile.geometry[i],'prec')
        Mean_Prec.append(prec.mean()[0])
    df = pd.DataFrame({'Name':Name,'Mean_Prec':Mean_Prec})
    return df

def drainage_density(shape_watersheds_dir,shape_drainage_line_dir):
    watersheds = gpd.read_file(shape_watersheds_dir)
    drainage_line = gpd.read_file(shape_drainage_line_dir)
    Name = []
    Drainage_Density = []
    for i in range(len(watersheds.geometry)):
        print(watersheds.Name[i])
        Name.append(watersheds.Name[i])
        drainage_density=drainage_line[drainage_line.geometry.within(watersheds.geometry[i])]
        Drainage_Density.append(drainage_density.Comprimeto.sum()/watersheds.Area_1[i])
    df =pd.DataFrame({'Name':Name,'Drainage_Density':Drainage_Density})
    return df

def shape_estacoes_area(dados,inventario_hidroweb_dir,shape_area_dir,buffer=0):
    ''' Retorna o shapefile da estações contidas no dataframe dados e que estão presentes em uma área de interesse.            
    '''   
    import baixar
    from shapely.geometry import Point
    import geopandas as gpd
    postos_area=baixar.postos_na_area(inventario_hidroweb_dir,shape_area_dir,buffer=0)
    postos_dados=[] 
    for column in dados.columns:
        postos_dados.append(int(column))
    postos_dados={'Codigo':postos_dados}
    postos_dados=pd.DataFrame(data=postos_dados)
    postos_dados=postos_area.merge(postos_dados,on=['Codigo'])
    shape_postos_area=[Point(x) for x in zip(postos_dados['Longitude'],postos_dados['Latitude'])]
    crs={'proj':'latlong','ellps':'WGS84','datum':'WGS84','no_def':True} #SC WGS 8
    shape_postos_area=gpd.GeoDataFrame(crs=crs,geometry=shape_postos_area)
    shape_postos_area['Codigo']=postos_dados['Codigo']
    shape_postos_area['Latitude']=postos_dados['Latitude']
    shape_postos_area['Longitude']=postos_dados['Longitude']
    return shape_postos_area

