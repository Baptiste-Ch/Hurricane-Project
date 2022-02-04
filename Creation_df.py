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

# EXTRACTION DES DONNEES

df = pd.read_csv('IBTrACS.csv', sep = ';', na_values=' ')
df['date'] =  pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df2 = df.iloc[:,[5,7,8,9,10,11,24]] # selection des colonnes



merge = time_psurf.merge(df2, left_on = 'time', right_on = 'date', how = 'inner')

merge = pd.concat(time_psurf, df, join = 'inner')

psurf = pd.DataFrame(ds2.variables['sp'][0,:,:])
