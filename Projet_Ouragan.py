######### HURRICANE PROJECT #############

        # IMPORTS

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import netCDF4 as nc


        #### METEO DATA ####
ds = nc.Dataset('gdas.t12z.sfcf006.nc')

print(ds.variables.keys())

keys = ds.variables
for key, value in keys.items() :
    print (key, value)
    print( )

# hgt_hyblev1, hpbl, soilt1, ssrun_acc  

        # GET ALL USEFULL VARIABLES
      
lon = ds.variables['lon'][0,1110:1430]
lat = ds.variables['lat'][209:341, 0]
time = ds.variables['time'] 
time_data = ds.variables['time'][:] 
pres = ds.variables['pressfc']
pres_data = ds.variables['pressfc'][0,209:341,1110:1430]
wind_data = ds.variables['vgrd_hyblev1'][0,209:341, 1110:1430]

        # VARIABLES PREPARATION
df_pres = pres_data.transpose()
df_pres = df_pres.ravel() * 0.01
df_lon = -(360 - np.array([lon,]*len(lat)).transpose().ravel())
df_lat =np.array([lat,]*len(lon)).ravel()
df_wind = wind_data.transpose()
df_wind = abs(df_wind.ravel()) * 1.94384
df_year = np.array([time.units[12:16]]*len(lat)*len(lon))
df_month = np.array([time.units[17:19]]*len(lat)*len(lon))
df_day = np.array([time.units[20:22]]*len(lat)*len(lon))
df_hour = np.array([float(time.units[23:25])+time_data[0]]*len(lat)*len(lon))


        # GET INTO DATAFRAME
df_meteo = np.array([df_lat, df_lon, df_pres, df_wind, df_day, df_month, df_year, df_hour])
df_meteo = df_meteo.transpose()
df_meteo = pd.DataFrame(df_meteo, columns = ['latitude', 'longitude', 'pressure', 'wind', 'day', 'month', 'year', 'hour'], dtype = 'float64')

        # GET INTO CSV
df_meteo.to_csv('daily_meteo_data.csv', sep = ';', index = False)



        ### HURRICAN DATA ###

df = pd.read_csv('IBTrACS.csv', sep = ';', na_values=' ')

df.dtypes
df2 = df.iloc[:,[1,6,7,8,9,10,11, 12, 13,20,21,22,23]] # selection des colonnes
df3 = df2.dropna() # les Na's sont enlevees
df_model = df3.iloc[162:,[2,5,6]]  #3,4,9,10,11,12
        # SPLIT DATA

X_train, X_test, y_train, y_test = train_test_split(df_model.loc[:, df_model.columns != 'nature'],
                                                    df_model['nature'], random_state = 0)

        # REGRESSOR

regressor = RandomForestClassifier(n_estimators=200, random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print("Training set : {:.3f}".format(regressor.score(X_train, y_train)))
print("Test set : {:.3f}".format(regressor.score(X_test, y_test)))

        # TESTING METEO DATA

y_pred_meteo = regressor.predict(df_meteo.iloc[:,2:4])


df_meteo['prediction'] = pd.Series(y_pred_meteo)
