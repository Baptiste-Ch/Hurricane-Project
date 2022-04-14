

import cdsapi
import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier #Réseau de forêt aléatoire
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
from cartopy.util import add_cyclic_point
import cartopy
import datetime as dt
import os
import imageio
from sklearn.metrics import accuracy_score
from Optimization import modelisation

   
# API
    # API directement fournie par ERA5.
    # Mise sous forme de fonction pour être utilisable en boucle
def API(year, month, day, hour, name):
    '''
    format exemple : 2020, 10, 20, 18:00, wilfred.nc
    '''
    
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'year': '{}'.format(year),
            'month': '{}'.format(month),
            'day': '{}'.format(day),
            'time': '{}:00'.format(hour),
            'area': [40, -100, 7, 30],
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
                '2m_temperature', 'evaporation', 'mean_sea_level_pressure',
                'sea_surface_temperature', 'surface_pressure', 'total_precipitation',
                '100m_u_component_of_wind', '100m_v_component_of_wind'],
            },
        '{}'.format(name))
    return nc.Dataset(name)

# NETCDF to DF TRANSFORMATION
    # L'étape de transformation des données netcdf brutes en df est automatisée par une fonction
def to_df(ds):
    '''
    ds : name of the netcdf dataset
    '''

    lon = pd.DataFrame(ds.variables['longitude'][:])
    lat = pd.DataFrame(ds.variables['latitude'][:])
    uwind = pd.DataFrame(ds.variables['u10'][0,:,:])
    vwind = pd.DataFrame(ds.variables['v10'][0,:,:])
    dewp = pd.DataFrame(ds.variables['d2m'][0,:,:])
    tsurf = pd.DataFrame(ds.variables['t2m'][0,:,:])
    evap = pd.DataFrame(ds.variables['e'][0,:,:])
    mslp = pd.DataFrame(ds.variables['msl'][0,:,:])
    sst = pd.DataFrame(ds.variables['sst'][0,:,:])
    psurf = pd.DataFrame(ds.variables['sp'][0,:,:])
    prcp = pd.DataFrame(ds.variables['tp'][0,:,:])
    u100 = pd.DataFrame(ds.variables['u100'][0,:,:])
    v100 = pd.DataFrame(ds.variables['v100'][0,:,:])


    df_lat =np.array([lat,]*len(lon)).ravel()
    df_lon = np.array([lon,]*len(lat)).transpose().ravel()
    df_uwind = np.array(uwind.transpose())
    df_uwind = df_uwind.ravel()
    df_vwind = np.array(vwind.transpose())
    df_vwind = df_vwind.ravel()
    df_dewp = np.array(dewp.transpose())
    df_dewp = df_dewp.ravel()
    df_tsurf = np.array(tsurf.transpose())
    df_tsurf = df_tsurf.ravel()
    df_evap = np.array(evap.transpose())
    df_evap = df_evap.ravel()
    df_mslp = np.array(mslp.transpose())
    df_mslp = df_mslp.ravel()
    df_sst = np.array(sst.transpose())
    df_sst = df_sst.ravel()
    df_psurf = np.array(psurf.transpose())
    df_psurf = df_psurf.ravel()
    df_prcp = np.array(prcp.transpose())
    df_prcp = df_prcp.ravel()
    df_uwind100 = np.array(u100.transpose())
    df_uwind100 = df_uwind100.ravel()
    df_vwind100 = np.array(v100.transpose())
    df_vwind100 = df_vwind100.ravel()
    df_ws = abs(df_uwind) + abs(df_vwind)
    df_ws100 = abs(df_uwind100) + abs(df_vwind100)
    df_RH = np.array(611**(5423*((1/273)-(1/df_dewp)))/611**(5423*((1/273)-(1/df_tsurf)))*100)

    df_all = np.array([df_lat, df_lon, df_psurf, df_sst, df_mslp, df_evap, df_prcp, df_tsurf, df_RH, df_ws, df_ws100])
    df_all = df_all.transpose()
    df_all = pd.DataFrame(df_all, columns = ['latitude', 'longitude', 'psurf', 'sst', 'mslp', 
                                             'evaporation', 'prcp', 'tsurf', 'RH', 'ws', 'ws100'], dtype = 'float64')
    df_all = df_all.dropna()
    df_all = df_all[(df_all['latitude'] > 15.75) | (df_all['longitude'] > -83.75)]
    df_all = df_all[(df_all['latitude'] > 8.75) | (df_all['longitude'] > -77.25)]
    return df_all


# ML PREDICTION + PLOTTING
def hurricane_plot(df_all, row_event, name_event, ds, i):
    '''
    df_all : données physiques de la carte
    row_event : une ligne du df de l'évènement d'ouragan qui a été isolé
    name_event : nom de l'évènement
    ds : nectdf dataset
    i : paramètre pour permettre une boucle (sinon i = None)
    '''
    
    Prediction = rfc.predict(df_all.iloc[:,2:])
    df_all['prediction'] = Prediction

    sst = ds.variables['sst'][0, :, :]
    lats = ds.variables['latitude'][:]
    lons = ds.variables['longitude'][:]

    levels = list(np.arange(290, 307, 1))
    field, long = add_cyclic_point(sst, coord=lons)

    fig = plt.figure(figsize=(15,10))
    ax = plt.subplot(projection=ccrs.PlateCarree()) # Permet de sélectionner la projection souhaitée
    ax.set_extent([-100, -5, 7, 40], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND) # ajoute d'autres styles, ici les frontières
    cf = plt.contourf(lons, lats, sst, transform=ccrs.PlateCarree(),  # On dit que la projection des data est à la base en PlateCarree
                      cmap=mpl.cm.RdYlBu_r, levels = levels) 
    cs = ax.contour(long, lats, field, colors='k', levels=levels, linewidths=0.5,
                    transform=ccrs.PlateCarree())
    
    plt.scatter(df_all[df_all['prediction'] == 1]['longitude'],
                df_all[df_all['prediction'] == 1]['latitude'], transform=ccrs.PlateCarree(),
                label=None, s = 2, zorder = 2)
    ax.plot(row_event['longitude'], row_event['latitude'], 'bo', 
            markersize=12, color = 'black', transform=ccrs.PlateCarree())
    plt.annotate('{}'.format(name_event), (row_event['longitude'], row_event['latitude']), 
                 textcoords = 'offset points', xytext=(0, 25), ha = 'left', size = 14,
                 bbox=dict(boxstyle="square", fc="white", ec="black", alpha = 0.8))

    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    cb = plt.colorbar(cf, shrink = 0.5, cax=cax)
    cb.ax.set_title('SST [K]')
    ax.set_title("Hurricane Prediction of {} : {}".format(name_event, row_event.iloc[0,0]), size = 16)
    
    plt.savefig('plots/katrina_{:02d}.png'.format(i), dpi = 100)
    plt.show()

  
# GIF CREATION
def gif(png_dir, images, name):
    '''
    png_dir : chemin d'accès aux images png
    images : liste vide
    name : nom d'enregistrement sans le format (ex : katrina_movie)'
    '''

    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
            print(file_name)
    imageio.mimsave('{}/{}.gif'.format(png_dir, name), images, duration = 0.25)

    
    
    ##### TEST DE PREDICTION DE L'OURAGAN KATRINA 2005  #####
    
df = pd.read_csv('hurricane_data.csv', sep = ',')
     
    
# On isole les évènements de Katrina
df['time'] = pd.to_datetime(df['time'])
df_kat = df[(df['name'] == 'KATRI2') & (df['time'].dt.year == 2005)]
df_kat = df_kat.dropna()
df_try = pd.merge(df, df_kat, how= 'outer', indicator = True)
df_try = df.loc[df_try._merge == 'left_only']
    
 
# On entraine le modèle sur les données historiques (sans les évènements Katrina)
rfc = modelisation('hurricane_data', 298, 100500)   


for i in range(len(df_kat)):
    a = pd.DataFrame(df_kat.iloc[i,:]).T 
    hour = str(a.time.dt.hour.values[0]).zfill(2)
    day = str(a.time.dt.day.values[0]).zfill(2)
    month = str(a.time.dt.month.values[0]).zfill(2)
    year = str(a.time.dt.year.values[0]).zfill(2)
    ds = API(year, month, day, hour, '2005.nc')
    df_all = to_df(ds)
    df_norm = (df_all - df_all.min())/(df_all.max() - df_all.min())
    df_norm['latitude'] = df_all['latitude']
    df_norm['longitude'] = df_all['longitude']
    hurricane_plot(df_norm, a, 'KATRINA', ds, i)
         
    
png_dir = 'C:/Users/baptc/Desktop/Master_SOAC/Master_1/S2/Projet_numerique/Data/plots'
images = []
gif(png_dir, images, 'Katrina_movie')

