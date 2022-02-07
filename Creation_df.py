#### HISTORICAL DATA HURRICANE #####

# PACKAGES

import pandas as pd
import numpy as np
from numpy.random import randint
from numpy.random import seed
import netCDF4 as nc
import datetime
from netCDF4 import Dataset

# IMPORT AND MERGE NETCDF FILES

p_surf = nc.MFDataset(['E:/Data_hurrican/surface_pressure_1.nc', 
                       'E:/Data_hurrican/surface_pressure_2.nc'], aggdim = 'time')
sst = nc.MFDataset(['E:/Data_hurrican/surface_temperature_1.nc', 
                    'E:/Data_hurrican/surface_temperature_2.nc'], aggdim = 'time')
prcp = nc.MFDataset(['E:/Data_hurrican/prcp_1.nc', 
                     'E:/Data_hurrican/prcp_2.nc'], aggdim = 'time')
mslp = nc.MFDataset(['E:/Data_hurrican/mslevel_pressure_1.nc', 
                       'E:/Data_hurrican/mslevel_pressure_2.nc'], aggdim = 'time')
evap = nc.MFDataset(['E:/Data_hurrican/evaporation_1.nc', 
                       'E:/Data_hurrican/evaporation_2.nc'], aggdim = 'time')


# ETABLISSEMENT DE L HEURE & COORDONNEES

time = pd.DataFrame(p_surf.variables['time'][:], columns = ['t'])

time_inter = []
for i in time.t:
    x = datetime.datetime(1900,1,1,0) + datetime.timedelta(hours = i)
    time_inter.append(x)    
time['time'] = time_inter
time = time.reset_index()


lat = pd.DataFrame(p_surf.variables['latitude'][:], columns = ['latitude'])
lat = lat.reset_index()

lon = pd.DataFrame(p_surf.variables['longitude'][:], columns = ['longitude'])
lon = lon.reset_index()





# EXTRACTION DES DONNEES

df = pd.read_csv('IBTrACS.csv', sep = ';', na_values=' ')
df['date'] =  pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df2 = df.iloc[:,[5,7,8,9,10,11,24]] # selection des colonnes
df2 = df2[df2['nature'] == 1]

merge = time.merge(df2, left_on = 'time', right_on = 'date', how = 'inner')
merge = merge.reset_index()


inter_psurf = []
inter_sst =   []
inter_mslp =  []
inter_evap =  []
inter_prcp =  []
full_list = [inter_psurf, inter_sst, inter_mslp, inter_evap, inter_prcp]
full_header = ['psurf', 'sst', 'mslp', 'evaporation', 'prcp']

for j in range(len(merge)):
    y = lat[lat['index'] == lat['latitude'].sub(merge.latitude[j]).abs().idxmin()]
    z = lon[lon['index'] == lon['longitude'].sub(merge.longitude[j]).abs().idxmin()]
    
    x_psurf = p_surf.variables['sp'][merge.iloc[j,1], y['index'], z['index']]
    inter_psurf.append(x_psurf)
    
    x_sst = sst.variables['sst'][merge.iloc[j,1], y['index'], z['index']]
    inter_sst.append(x_sst)
    
    x_mslp = mslp.variables['msl'][merge.iloc[j,1], y['index'], z['index']]
    inter_mslp.append(x_mslp)
    
    x_evap = evap.variables['e'][merge.iloc[j,1], y['index'], z['index']]
    inter_evap.append(x_evap)
    
    x_prcp = prcp.variables['tp'][merge.iloc[j,1], y['index'], z['index']]
    inter_prcp.append(x_prcp)
    
    
for i in range(len(full_list)):
    merge['{}'.format(full_header[i])] = np.concatenate(full_list[i], axis = 0)




# ADDITION OF RANDOM METEOROLOGICAL STATES
