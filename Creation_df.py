#### HISTORICAL DATA HURRICANE #####

# PACKAGES

import pandas as pd
import numpy as np
from numpy.random import randint
from numpy.random import seed
import netCDF4 as nc
import datetime
from netCDF4 import Dataset
import matplotlib.pyplot as plt


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
vwind = nc.MFDataset(['E:/Data_hurrican/evaporation_1.nc', 
                       'E:/Data_hurrican/evaporation_2.nc'], aggdim = 'time')
uwind = nc.MFDataset(['E:/Data_hurrican/evaporation_1.nc', 
                       'E:/Data_hurrican/evaporation_2.nc'], aggdim = 'time')
dew = nc.MFDataset(['E:/Data_hurrican/evaporation_1.nc', 
                       'E:/Data_hurrican/evaporation_2.nc'], aggdim = 'time')
t_surf = nc.MFDataset(['E:/Data_hurrican/evaporation_1.nc', 
                       'E:/Data_hurrican/evaporation_2.nc'], aggdim = 'time')

# ESTABLISHMENT OF COORDINATES AND TIME

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




# HURRICANE DATA EXTRACTION

df = pd.read_csv('IBTrACS.csv', sep = ';', na_values=' ')
df['date'] =  pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df2 = df.iloc[:,[5,7,8,9,24]] # selection des colonnes
df2 = df2[df2['nature'] == 1]

merge = time.merge(df2, left_on = 'time', right_on = 'date', how = 'inner')
merge = merge.drop('date', 1)




# ADDITION OF RANDOM METEOROLOGICAL ROWS
    
added_df = pd.DataFrame(data = None, columns=['index', 't', 'time','latitude', 'longitude'])
for row in range(3000):
    new_lon = lon['longitude'].sample()
    new_lat = lat['latitude'].sample()
    new_index = pd.DataFrame(time['index'].sample())
    new_time = time.loc[time['index'] == new_index.iloc[0,0]]
    new_time['latitude'] = new_lat.values
    new_time['longitude'] = new_lon.values
    #new_time = new_time.values.tolist()
    added_df = pd.concat([new_time, added_df])        
    

full = merge.append(added_df)
full['nature'] = full['nature'].fillna(0)




# ASSIGNING ALL PARAMETERS WE NEED TO OUR DATAFRAME

inter_psurf = []
inter_sst =   []
inter_mslp =  []
inter_evap =  []
inter_prcp =  []
inter_vwind = []
inter_uwind = []
inter_dew =   []
inter_tsurf = []
full_list = [inter_psurf, inter_sst, inter_mslp, inter_evap, inter_prcp, 
             inter_vwind, inter_uwind, inter_dew, inter_tsurf]
full_header = ['psurf', 'sst', 'mslp', 'evaporation', 'prcp', 'uwind', 'vwind', 'dewp', 'tsurf']

for j in range(len(full)):
    y = lat[lat['index'] == lat['latitude'].sub(full.latitude[j]).abs().idxmin()]
    z = lon[lon['index'] == lon['longitude'].sub(full.longitude[j]).abs().idxmin()]
    
    x_psurf = p_surf.variables['sp'][full.iloc[j,1], y['index'], z['index']]
    inter_psurf.append(x_psurf)
    
    x_sst = sst.variables['sst'][full.iloc[j,1], y['index'], z['index']]
    inter_sst.append(x_sst)
    
    x_mslp = mslp.variables['msl'][full.iloc[j,1], y['index'], z['index']]
    inter_mslp.append(x_mslp)
    
    x_evap = evap.variables['e'][full.iloc[j,1], y['index'], z['index']]
    inter_evap.append(x_evap)
    
    x_prcp = prcp.variables['tp'][full.iloc[j,1], y['index'], z['index']]
    inter_prcp.append(x_prcp)
    
    x_uwind = uwind.variables['u'][full.iloc[j,1], y['index'], z['index']]
    inter_uwind.append(x_uwind)
    
    x_vwind = vwind.variables['v'][full.iloc[j,1], y['index'], z['index']]
    inter_vwind.append(x_vwind)
    
    x_dew = dew.variables['dew'][full.iloc[j,1], y['index'], z['index']]
    inter_dew.append(x_dew)
    
    x_tsurf = t_surf.variables['st'][full.iloc[j,1], y['index'], z['index']]
    inter_tsurf.append(x_tsurf)
    
for i in range(len(full_list)):
    full['{}'.format(full_header[i])] = np.concatenate(full_list[i], axis = 0)



