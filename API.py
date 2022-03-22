

import cdsapi
import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier #Réseau de forêt aléatoire
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
from cartopy.util import add_cyclic_point
import cartopy


df = pd.read_csv('data_4.csv', sep = ';')

df['RH'] = 611**(5423*((1/273)-(1/df['dewp'])))/611**(5423*((1/273)-(1/df['tsurf'])))*100
df = df[df['sst'] != -32767]
df['ws'] = np.sqrt(df['uwind']**2 + df['vwind']**2)
df = df[df['name'] != 'NOT_2MED']

df.loc[df.sst < 300, 'nature'] = 0
df.loc[df.prcp < 0.0005, 'nature'] = 0
df.loc[df.psurf > 100000, 'nature'] = 0


dfi = df[df['nature'] == 1]
dfii = df[df['nature'] == 0]


# TEST DE PREDICTION DE L'OURAGAN KATRINA 2005

df['time'] = pd.to_datetime(df['time'])
df_kat = df[(df['name'] == 'WILFRED') & (df['time'].dt.year == 2020)]
duplicates = pd.merge(df, df_kat, how='inner',left_on=['Unnamed: 0'], right_on=['Unnamed: 0'])
df_try = df.drop(duplicates['Unnamed: 0'])
drop_columns = ['index', 't', 'time', 'name', 'latitude', 'longitude', 
                'Unnamed: 0', 'uwind', 'vwind', 'dewp']
df_try = df_try.drop(drop_columns, axis = 1)      
# print(df_try.corr())




# MODELE
X_train, X_test, y_train, y_test = train_test_split(df_try.loc[:, df_try.columns != 'nature'],
                                                    df_try['nature'], random_state = 0)

clf = RandomForestClassifier(n_estimators = 200, criterion = 'gini', random_state=1)
clf.fit(X_train, y_train)
scores = cross_val_score(clf, df_try.loc[:, df_try.columns != 'nature'], df_try['nature'], cv=5)
print(scores)


   
# API
c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'year': '2020',
        'month': '09',
        'day': '20',
        'time': '18:00',
        'area': [ 40, -100, 7, 30],
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
            '2m_temperature', 'evaporation', 'mean_sea_level_pressure',
            'sea_surface_temperature', 'surface_pressure', 'total_precipitation' ],
    },
    'wilfred.nc')

ds = nc.Dataset('katrina.nc')

ds = nc.Dataset('adaptor.mars.internal-1647624533.0713866-13377-7-733f0755-0b67-4c35-95df-e5a5a612d873.nc')

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
df_ws = df_uwind + df_vwind
df_RH = np.array(611**(5423*((1/273)-(1/df_dewp)))/611**(5423*((1/273)-(1/df_tsurf)))*100)

df_all = np.array([df_lat, df_lon, df_psurf, df_sst, df_mslp, df_evap, df_prcp, df_tsurf, df_RH, df_ws])
df_all = df_all.transpose()
df_all = pd.DataFrame(df_all, columns = ['latitude', 'longitude', 'psurf', 'sst', 'mslp', 
                                         'evaporation', 'prcp', 'tsurf', 'RH', 'ws'], dtype = 'float64')
df_all = df_all.dropna()


# Forecasting
Prediction = clf.predict(df_all.iloc[:,2:])

df_all['prediction'] = Prediction

# VISUALISATION

sst = ds.variables['sst'][0, :, :]
lats = ds.variables['latitude'][:]
lons = ds.variables['longitude'][:]

levels = list(np.arange(290, 307, 1))

field, long = add_cyclic_point(sst, coord=lons)


fig = plt.figure(figsize=(15,10))
ax = plt.subplot(projection=ccrs.PlateCarree()) # Permet de sélectionner la projection souhaitée
ax.set_extent([-100, -5, 7, 40], crs=ccrs.PlateCarree())
# ax.coastlines() # on ajoute des jolies côtes
ax.add_feature(cfeature.LAND) # ajoute d'autres styles, ici les frontières

cf = plt.contourf(lons, lats, sst, transform=ccrs.PlateCarree(),  # On dit que la projection des data est à la base en PlateCarree
         cmap=mpl.cm.RdYlBu_r, levels = levels) 
cs = ax.contour(long, lats, field, colors='k', levels=levels, linewidths=0.5,
                transform=ccrs.PlateCarree())
# lb = plt.clabel(cs, fontsize=8, inline=True, fmt='%r', levels = [290, 292, 294, 296, 298, 300, 302, 304, 306]);

plt.scatter(df_all[df_all['prediction'] == 1]['longitude'],
            df_all[df_all['prediction'] == 1]['latitude'], transform=ccrs.PlateCarree(),
            label=None, s = 8)
ax.plot(-89.6, 29.3, 'bo', markersize=12, color = 'black', transform=ccrs.PlateCarree())
plt.annotate('KATRINA', (-89.6, 29.3), textcoords = 'offset points',
             xytext=(0, 25), ha = 'left', size = 12, bbox=dict(boxstyle="square", fc="none", ec="gray"))
#sm = plt.cm.ScalarMappable(cmap=mpl.cm.RdYlBu_r) # ,norm=plt.Normalize(0,1)
#sm._A = []
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
cb = plt.colorbar(cf, shrink = 0.5, cax=cax)
cb.ax.set_title('SST [K]')
ax.set_title("Prediction de l'ouragan Katrina [29/08/2005 11:00]", size = 16)

#plt.colorbar(sm,cax=cax)
plt.savefig('katrina.png', dpi = 1000)
plt.show()

