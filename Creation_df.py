#### HISTORICAL DATA HURRICANE #####

# PACKAGES

import pandas as pd
import numpy as np
import netCDF4 as nc
import datetime


ds2 = nc.Dataset('E:/Data_hurrican/surface_pressure_1.nc')                              # import des data Netcdf

# print(ds2.variables.keys())

# ETABLISSEMENT DE L HEURE
time_psurf = pd.DataFrame(ds2.variables['time'][:], columns = ['t'])                    # je selectionne que le panel de dates qu on etudie et j en fais un df

time = []
for i in time_psurf.t:
    x = datetime.datetime(1900,1,1,0) + datetime.timedelta(hours = i)
    time.append(x)    
time_psurf['time'] = time
time_psurf = time_psurf.reset_index()                                                   # tout ca c est pour trouver la vraie date issue de la colonne de base qu on







# ETABLISSEMENT COORDONNEES
lat = pd.DataFrame(ds2.variables['latitude'][:], columns = ['latitude'])
lat = lat.reset_index()                                                                 # les deux lignes pour definir le panel de latitudes qu on a et surtout definir quelle latitude se trouve en quel position du jeu Netcdf

lon = pd.DataFrame(ds2.variables['longitude'][:], columns = ['longitude'])
lon = lon.reset_index()                                                                 # pareil ici pour la longitude






# EXTRACTION DES DONNEES

df = pd.read_csv('IBTrACS.csv', sep = ';', na_values=' ')                               # on joue avec ibtracs mtn
df['date'] =  pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df2 = df.iloc[:,[5,7,8,9,10,11,24]] # selection des colonnes
df2 = df2[df2['nature'] == 1]                                                           # tout ca c est pour selectionner que ce qui m interesse dans les donnees

merge = time_psurf.merge(df2, left_on = 'time', right_on = 'date', how = 'inner')       # je fusionne ibtracs et les indexs de temps qu on a defini dans time_psurf (ligne 24)
merge = merge.reset_index()                                                             

psurf = []                                                                              # je prepare un colonne vide pour mettre la pression de chaque ligne d Ibtracs
for j in range(len(merge)):
    y = lat[lat['index'] == lat['latitude'].sub(merge.latitude[j]).abs().idxmin()]      # On obtient la latitude du Netcdf la plus proche de la latitude issue de la ligne j de ibtracs (df merge)
    z = lon[lon['index'] == lon['longitude'].sub(merge.longitude[j]).abs().idxmin()]    # pareil pour la longitude
    x = ds2.variables['sp'][merge.iloc[j,1], y['index'], z['index']]                    # mtn je cherche la valeur de 's' correspondant a la lat/lon de la ligne j et au temps egal au temps de la ligne j
    psurf.append(x)                                                                     # je met l info dans l array psurf et on laisse la boucle remplir l array
merge['psurf'] = np.concatenate(psurf, axis = 0)                                        # petite astuce pour bien faire rentrer les valeurs de pression dans le dossier Ibtracs (merge)
