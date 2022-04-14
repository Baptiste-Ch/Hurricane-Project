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
import random

# IMPORT DES JEUX DE DONNEES EN FORMAT NETCDF. CHAQUE VARIABLE EST IMPORTEE SEPAREMENT
    # Source : https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview

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
vwind = nc.MFDataset(['E:/Data_hurrican/vwind_1.nc', 
                       'E:/Data_hurrican/vwind_2.nc'], aggdim = 'time')
uwind = nc.MFDataset(['E:/Data_hurrican/uwind_1.nc', 
                       'E:/Data_hurrican/uwind_2.nc'], aggdim = 'time')
dew = nc.MFDataset(['E:/Data_hurrican/dewpoint_1.nc', 
                       'E:/Data_hurrican/dewpoint_2.nc'], aggdim = 'time')
t_surf = nc.MFDataset(['E:/Data_hurrican/temperature2m_1.nc', 
                       'E:/Data_hurrican/temperature2m_2.nc'], aggdim = 'time')
uwind100 = nc.MFDataset(['E:/Data_hurrican/uwind100_1.nc', 
                       'E:/Data_hurrican/uwind100_2.nc'], aggdim = 'time')
vwind100 = nc.MFDataset(['E:/Data_hurrican/vwind100_1t.nc', 
                       'E:/Data_hurrican/vwind100_2.nc'], aggdim = 'time')

# ETABLISSEMENT EN DF DES COORDONNEES SPATIALES ET TEMPORELLES
    # Temps
time = pd.DataFrame(p_surf.variables['time'][:], columns = ['t'])
time_inter = []
for i in time.t:
    x = datetime.datetime(1900,1,1,0) + datetime.timedelta(hours = i)
    time_inter.append(x)    
time['time'] = time_inter
time = time.reset_index()

    # Coordonnées
lat = pd.DataFrame(p_surf.variables['latitude'][:], columns = ['latitude'])
lat = lat.reset_index()

lon = pd.DataFrame(p_surf.variables['longitude'][:], columns = ['longitude'])
lon = lon.reset_index()




# EXTRACTION DES DONNEES D'OURAGANS

df = pd.read_csv('IBTrACS.csv', sep = ';', na_values=' ')
df['date'] =  pd.to_datetime(df[['year', 'month', 'day', 'hour']]) # formalisation de la date
df2 = df.iloc[:,[5,7,8,9,24]]                                      # sélection des colonnes
df2 = df2[df2['nature'] == 1]                                      # sélection des données d'ouragans (NATURE = 1)

# PHASE D'ASSOCIATION
    # Les indexes temporels physiques sont associés avec les dates d'ouragans
merge = time.merge(df2, left_on = 'time', right_on = 'date', how = 'inner')
merge = merge.drop('date', 1)
merge = merge[merge['name'] != 'NOT_2MED']
merge = merge[merge['longitude'] > -100]
merge = merge[merge['latitude'] < 41]


# ADDITION DE DONNEES METEOROLOGIQUES ALEATOIRES

    # Pour que le modèle puisse apprendre il est nécessaire de lui présenter des données
    # d'états 'normaux'
    # Cette étape peut prendre plusieurs dizaines de minutes

np.random.seed(0)   
added_df = pd.DataFrame(data = None, columns=['index', 't', 'time','latitude', 'longitude']) # df vide
for row in range(10000):                                           # Création de 20 000 lignes
    new_lon = lon['longitude'].sample()                            # Echantillonnage aléatoire de longitude
    new_lat = lat['latitude'].sample()                             # Même chose pour la latitude
    new_index = pd.DataFrame(time['index'].sample())               # De même pour la date
    new_time = time.loc[time['index'] == new_index.iloc[0,0]]
    new_time['latitude'] = new_lat.values
    new_time['longitude'] = new_lon.values
    #new_time = new_time.values.tolist()
    added_df = pd.concat([new_time, added_df])                     # On concatène les coordonnées     
    print(row)

full = merge.append(added_df)                                      # Ajout des données ouragans avec les données météo 
full['nature'] = full['nature'].fillna(0)                          # 0 = Absence d'ouragan
full = full.reset_index(drop=True)

# AJOUT DES PARAMETRES PHYSIQUES POUR CHAQUE POINT DE DONNEE

    # Le programme suivant vise à chercher, pour chaque coordonnée du fichier 'full', les paramètres physiques
    # dans les métadonnées
    
    # ATTENTION : Cette étape nécessite plusieurs heures de compilation


inter_psurf = []
inter_sst =   []
inter_mslp =  []
inter_evap =  []
inter_prcp =  []
inter_vwind = []
inter_uwind = []
inter_dew =   []
inter_tsurf = []
inter_vwind100 = []
inter_uwind100 = []
full_list = [inter_psurf, inter_sst, inter_mslp, inter_evap, inter_prcp, 
             inter_vwind, inter_uwind, inter_dew, inter_tsurf, inter_uwind100, inter_vwind100]
full_header = ['psurf', 'sst', 'mslp', 'evaporation', 'prcp', 'uwind', 'vwind', 'dewp', 'tsurf', 'uwind100', 'vwind100']


for j in range(len(full)):
    y = lat[lat['index'] == lat['latitude'].sub(full.latitude[j]).abs().idxmin()]    # Cherche la latitude des métadonnées la plus proche des latitude du df ouragans
    z = lon[lon['index'] == lon['longitude'].sub(full.longitude[j]).abs().idxmin()]  # Idem pour la longitude
   
    x_psurf = p_surf.variables['sp'][full.iloc[j,0], y['index'], z['index']]         # Prends le paramètres associé aux coordonnées précédement définis
    inter_psurf.append(x_psurf)                                                      # On rempli les listes vides
    x_sst = sst.variables['sst'][full.iloc[j,0], y['index'], z['index']]             # Idem pour chaque paramètres
    inter_sst.append(x_sst)
    x_mslp = mslp.variables['msl'][full.iloc[j,0], y['index'], z['index']]
    inter_mslp.append(x_mslp)
    x_evap = evap.variables['e'][full.iloc[j,0], y['index'], z['index']]
    inter_evap.append(x_evap)
    x_prcp = prcp.variables['tp'][full.iloc[j,0], y['index'], z['index']]
    inter_prcp.append(x_prcp)
    x_uwind = uwind.variables['u10'][full.iloc[j,0], y['index'], z['index']]
    inter_uwind.append(x_uwind)
    x_vwind = vwind.variables['v10'][full.iloc[j,0], y['index'], z['index']]
    inter_vwind.append(x_vwind)
    x_dew = dew.variables['d2m'][full.iloc[j,0], y['index'], z['index']]
    inter_dew.append(x_dew)
    x_tsurf = t_surf.variables['t2m'][full.iloc[j,0], y['index'], z['index']]
    inter_tsurf.append(x_tsurf)
    x_uwind100 = uwind100.variables['u100'][full.iloc[j,0], y['index'], z['index']]
    inter_uwind100.append(x_uwind100)
    x_vwind100 = vwind100.variables['v100'][full.iloc[j,0], y['index'], z['index']]
    inter_vwind100.append(x_vwind100)     
    print('etape :',j,'/', len(full))
    
for i in range(len(full_list)):                                                      # Ajoute les données physiques aux df d'ouragans
    full['{}'.format(full_header[i])] = np.concatenate(full_list[i], axis = 0)





full.to_csv('full_df_2.csv', index=False)    # On enregistre la compilation brute au cas où


# FINALISATION DU JEU DE DONNEES A ENTRAINER
    # Attribution de l'Humidité Relative dépendante de la température de Point de Rosée
full['RH'] = 611**(5423*((1/273)-(1/full['dewp'])))/611**(5423*((1/273)-(1/full['tsurf'])))*100

    # Les données continentales identifiées par les informations abérantes de sst sont supprimés
full = full[full['sst'] != -32767]

    # Le vent calculé
full['ws'] = np.sqrt(full['uwind']**2 + full['vwind']**2)
full['ws100'] = np.sqrt(full['uwind100']**2 + full['vwind100']**2)
full = full[full['name'] != 'NOT_2MED']

drop_columns = ['index', 't', 'uwind', 'vwind', 'dewp', 'uwind100', 'vwind100']
full = full.drop(drop_columns, axis = 1)                                             # On supprime les colonnes non essentielles

# ENREGISTREMENT DU CSV


full.to_csv('hurricane_data.csv', index = False)                                                       

