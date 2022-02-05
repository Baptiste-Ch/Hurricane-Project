#### HISTORICAL DATA HURRICANE #####

# PACKAGES

import pandas as pd
import numpy as np
import netCDF4 as nc
import datetime


ds2 = nc.Dataset('E:/Data_hurrican/surface_pressure_1.nc')

# print(ds2.variables.keys())

# ETABLISSEMENT DE L HEURE
time_psurf = pd.DataFrame(ds2.variables['time'][:], columns = ['t'])

time = []
for i in time_psurf.t:
    x = datetime.datetime(1900,1,1,0) + datetime.timedelta(hours = i)
    time.append(x)    
time_psurf['time'] = time
time_psurf = time_psurf.reset_index()







# ETABLISSEMENT COORDONNEES
lat = pd.DataFrame(ds2.variables['latitude'][:], columns = ['latitude'])
lat = lat.reset_index()

lon = pd.DataFrame(ds2.variables['longitude'][:], columns = ['longitude'])
lon = lon.reset_index()






# EXTRACTION DES DONNEES

df = pd.read_csv('IBTrACS.csv', sep = ';', na_values=' ')
df['date'] =  pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df2 = df.iloc[:,[5,7,8,9,10,11,24]] # selection des colonnes
df2 = df2[df2['nature'] == 1]

merge = time_psurf.merge(df2, left_on = 'time', right_on = 'date', how = 'inner')
merge = merge.reset_index()

psurf = []
for j in range(len(merge)):
    y = lat[lat['index'] == lat['latitude'].sub(merge.latitude[j]).abs().idxmin()]
    z = lon[lon['index'] == lon['longitude'].sub(merge.longitude[j]).abs().idxmin()]
    x = ds2.variables['sp'][merge.iloc[j,1], y['index'], z['index']]
    psurf.append(x)
merge['psurf'] = np.concatenate(psurf, axis = 0)



# y = lat[lat['index'] == lat['latitude'].sub(merge.latitude[0]).abs().idxmin()]

