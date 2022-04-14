import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression #Classificateur binaire
from sklearn.ensemble import RandomForestClassifier #Réseau de forêt aléatoire
from sklearn.metrics import mean_squared_error #pour calculer REQM
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

filename = 'data.csv'
df = pd.read_csv(filename)
#On crée une colonne vitesse du vent ws qui est la sqrt( (uwind)**2 + (vwind)**2 )
df['ws'] = np.sqrt(df['uwind']**2 + df['vwind']**2) #m/s
pd.options.mode.chained_assignment = None



#On remplace les valeurs abérantes par des NaN
df['sst'] = df['sst'].replace(-32767.000000,np.nan)
#On enlève les colonnes qui nous intéressent pas
drop_columns = ['index', 't', 'time', 'name', 'latitude', 'longitude', 'Unnamed: 0']
df_ss_col = df.drop(drop_columns, axis = 1)
#On mélange tout et on se place sur le seed 1
np.random.seed(1)
df = df_ss_col.loc[np.random.permutation(len(df_ss_col))]
#On se débarasse des lignes avec des NaN
clean_df = df.dropna(axis=0)
df = clean_df
df[df['nature']==0]


col = df.columns.to_list()
for i in df[col]:
    print(i)
    print(df[i].max())
    print(df[i].min())
    
    

#On scinde notre df en test et colonne
df_test = df.iloc[0:15000]
df_train = df.iloc[15000:]

#On choisit les colonnes que l'on entraine et celle sur laquelle on teste notre modèle
features = df.columns.to_list()
features.remove('nature')
target = 'nature'

#On paramètre notre modèle et on l'entraîne
rfc =  RandomForestClassifier(n_estimators=200, random_state = 0)
rfc.fit(df_train[features], df_train[target])
prediction = rfc.predict(df_test[features])
df_test['predicted_nature'] = prediction
matches = df_test['predicted_nature'] == df_test['nature']
correct_predictions = df_test[matches]
accuracy = len(correct_predictions) / len(df_test)
print(accuracy)
print(df.corr())

reqm = mean_squared_error(df_test[target],prediction,squared = False)
print(reqm)




#On scinde notre df en test et colonne
df_test = df.iloc[0:15000]
df_train = df.iloc[15000:]

#On choisit les colonnes que l'on entraine et celle sur laquelle on teste notre modèle
target = 'nature'
features = ['psurf','sst', 'mslp', 'evaporation', 'prcp', 'uwind', 'vwind', 'dewp', 'tsurf', 'ws']

rmse_values = []
precision_values = []
x =  [i for i in range(0,10)]

for i in range(10):
    print(features[i])
    features.remove(features[i])
    print(features)
    rfc =  RandomForestClassifier(n_estimators=200, random_state = 0)
    rfc.fit(df_train[features], df_train[target])
    prediction = rfc.predict(df_test[features])
    df_test['predicted_nature'] = prediction
    matches = df_test['predicted_nature'] == df_test['nature']
    correct_predictions = df_test[matches]
    precision_values.append(len(correct_predictions) / len(df_test))
    rmse_values.append(mean_squared_error(df_test[target],prediction,squared = False))
    features  = ['psurf','sst', 'mslp', 'evaporation', 'prcp', 'uwind', 'vwind', 'dewp', 'tsurf', 'ws']


rmse_dic = {}
precision_dic = {}
for i in range(len(features)):
    rmse_dic[features[i]] = rmse_values[i]

for k in range(len(features)):
    precision_dic[features[k]] = precision_values[k]

plt.figure()
myList = rmse_dic.items()
x, y = zip(*myList) 
plt.xticks(rotation=60)
plt.title("RMEQ")
plt.scatter(x, y)
plt.show()

plt.figure()
myList_bis = precision_dic.items()
a, b = zip(*myList_bis) 
plt.xticks(rotation=60)
plt.title("Précision")
plt.scatter(a, b)
plt.show()




##### PART 2

filename = 'data.csv'
df = pd.read_csv(filename)
#On crée une colonne vitesse du vent ws qui est la sqrt( (uwind)**2 + (vwind)**2 )
df['ws'] = np.sqrt(df['uwind']**2 + df['vwind']**2) #m/s
pd.options.mode.chained_assignment = None

#On remplace les valeurs abérantes par des NaN
df['sst'] = df['sst'].replace(-32767.000000,np.nan)
#On enlève les colonnes qui nous intéressent pas
drop_columns = ['index', 't', 'time', 'name', 'latitude', 'longitude', 'Unnamed: 0', 'ws', 'psurf', 'mslp', 'evaporation', 'tsurf']
df_ss_col = df.drop(drop_columns, axis = 1)
#On mélange tout et on se place sur le seed 1
np.random.seed(1)
df = df_ss_col.loc[np.random.permutation(len(df_ss_col))]
#On se débarasse des lignes avec des NaN
clean_df = df.dropna(axis=0)
df = clean_df 

#On choisit les colonnes que l'on entraine et celle sur laquelle on teste notre modèle
features = df.columns.to_list()
features.remove('nature')
target = 'nature'

#On scinde notre df en test et colonne
df_test = df.iloc[0:15000]
df_train = df.iloc[15000:]

#On paramètre notre modèle et on l'entraîne
rfc =  RandomForestClassifier(n_estimators=200, random_state = 0)
rfc.fit(df_train[features], df_train[target])
prediction = rfc.predict(df_test[features])

df_test['predicted_nature'] = prediction
matches = df_test['predicted_nature'] == df_test['nature']
correct_predictions = df_test[matches]
accuracy = len(correct_predictions) / len(df_test)
print(accuracy)


##### PART 3

filename = 'data.csv'
df = pd.read_csv(filename)
#On crée une colonne vitesse du vent ws qui est la sqrt( (uwind)**2 + (vwind)**2 )
df['ws'] = np.sqrt(df['uwind']**2 + df['vwind']**2) #m/s
pd.options.mode.chained_assignment = None

#On remplace les valeurs abérantes par des NaN
df['sst'] = df['sst'].replace(-32767.000000,np.nan)
#On enlève les colonnes qui nous intéressent pas
drop_columns = ['index', 't', 'time', 'name', 'latitude', 'longitude', 'Unnamed: 0']
df_ss_col = df.drop(drop_columns, axis = 1)
#On mélange tout et on se place sur le seed 1
np.random.seed(1)
df = df_ss_col.loc[np.random.permutation(len(df_ss_col))]
#On se débarasse des lignes avec des NaN
clean_df = df.dropna(axis=0)
df = clean_df

#On choisit les colonnes que l'on entraine et celle sur laquelle on teste notre modèle
features = df.columns.to_list()
features.remove('nature')
target = 'nature'

#On normalise
nature = df['nature'] 
df_ss_nature = df.drop(['nature'], axis=1)
df_ss_nature = (df_ss_nature - df_ss_nature.min())/(df_ss_nature.max() - df_ss_nature.min())
df_ss_nature['nature'] = nature
df = df_ss_nature

#On scinde notre df en test et colonne
df_test = df.iloc[0:15000]
df_train = df.iloc[15000:]

#On paramètre notre modèle et on l'entraîne
rfc =  RandomForestClassifier(n_estimators=200, random_state = 0)
rfc.fit(df_train[features], df_train[target])
prediction = rfc.predict(df_test[features])

df_test['predicted_nature'] = prediction
matches = df_test['predicted_nature'] == df_test['nature']
correct_predictions = df_test[matches]
accuracy = len(correct_predictions) / len(df_test)
print(accuracy)



##### PART 4

filename = 'data.csv'
df = pd.read_csv(filename)
#On crée une colonne vitesse du vent ws qui est la sqrt( (uwind)**2 + (vwind)**2 )
df['ws'] = np.sqrt(df['uwind']**2 + df['vwind']**2) #m/s
pd.options.mode.chained_assignment = None

#On remplace les valeurs abérantes par des NaN
df['sst'] = df['sst'].replace(-32767.000000,np.nan)
#On enlève les colonnes qui nous intéressent pas
drop_columns = ['index', 't', 'time', 'name', 'latitude', 'longitude', 'Unnamed: 0']
df_ss_col = df.drop(drop_columns, axis = 1)
#On mélange tout et on se place sur le seed 1
np.random.seed(1)
df = df_ss_col.loc[np.random.permutation(len(df_ss_col))]
#On se débarasse des lignes avec des NaN
clean_df = df.dropna(axis=0)
df = clean_df 

#On choisit les colonnes que l'on entraine et celle sur laquelle on teste notre modèle
features = df.columns.to_list()
features.remove('nature')
target = 'nature'

#On normalise
nature = df['nature'] 
df_ss_nature = df.drop(['nature'], axis=1)
df_ss_nature = (df_ss_nature - df_ss_nature.mean())/(df_ss_nature.std())
df_ss_nature['nature'] = nature
df = df_ss_nature

#On scinde notre df en test et colonne
df_test = df.iloc[0:15000]
df_train = df.iloc[15000:]

#On paramètre notre modèle et on l'entraîne
rfc =  RandomForestClassifier(n_estimators=200, random_state = 0)
rfc.fit(df_train[features], df_train[target])
prediction = rfc.predict(df_test[features])

df_test['predicted_nature'] = prediction
matches = df_test['predicted_nature'] == df_test['nature']
correct_predictions = df_test[matches]
accuracy = len(correct_predictions) / len(df_test)
print(accuracy)



##### PART 5

filename = 'data.csv'
df = pd.read_csv(filename)
#On crée une colonne vitesse du vent ws qui est la sqrt( (uwind)**2 + (vwind)**2 )
df['ws'] = np.sqrt(df['uwind']**2 + df['vwind']**2) #m/s
pd.options.mode.chained_assignment = None

#On remplace les valeurs abérantes par des NaN
df['sst'] = df['sst'].replace(-32767.000000,np.nan)
#On enlève les colonnes qui nous intéressent pas
drop_columns = ['index', 't', 'time', 'name', 'latitude', 'longitude', 'Unnamed: 0']
df_ss_col = df.drop(drop_columns, axis = 1)
#On mélange tout et on se place sur le seed 1
np.random.seed(1)
df = df_ss_col.loc[np.random.permutation(len(df_ss_col))]
#On se débarasse des lignes avec des NaN
clean_df = df.dropna(axis=0)
df = clean_df

#On choisit les colonnes que l'on entraine et celle sur laquelle on teste notre modèle
features = df.columns.to_list()
features.remove('nature')
target = 'nature'

#On normalise
nature = df['nature'] 
df_ss_nature = df.drop(['nature'], axis=1)
df_ss_nature = (df_ss_nature - df_ss_nature.min())/(df_ss_nature.max() - df_ss_nature.min())
df_ss_nature['nature'] = nature
df = df_ss_nature

#On scinde notre df en test et colonne
df_test = df.iloc[0:15000]
df_train = df.iloc[15000:]

#On paramètre notre modèle et on l'entraîne
rfc =  RandomForestClassifier(n_estimators=200, random_state = 0)
rfc.fit(df_train[features], df_train[target])
prediction = rfc.predict(df_test[features])

df_test['predicted_nature'] = prediction
matches = df_test['predicted_nature'] == df_test['nature']
correct_predictions = df_test[matches]
accuracy = len(correct_predictions) / len(df_test)
print(accuracy)

kf = KFold(n_splits = 5, shuffle = True, random_state = 1)
precision = cross_val_score(rfc, df[features], df[target], cv = kf)
#mse_fold = cross_val_score(rfc, df[features], df[target], scoring = 'neg_mean_squared_error', cv = kf) 
#rmse_fold = np.sqrt(np.absolute(mse_fold))


print(precision)



##### PART 6

filename = 'data.csv'
df = pd.read_csv(filename)
#On crée une colonne vitesse du vent ws qui est la sqrt( (uwind)**2 + (vwind)**2 )
df['ws'] = np.sqrt(df['uwind']**2 + df['vwind']**2) #m/s
pd.options.mode.chained_assignment = None

#On remplace les valeurs abérantes par des NaN
df['sst'] = df['sst'].replace(-32767.000000,np.nan)
#On enlève les colonnes qui nous intéressent pas
drop_columns = ['index', 't', 'time', 'name', 'latitude', 'longitude', 'Unnamed: 0']
df_ss_col = df.drop(drop_columns, axis = 1)
#On mélange tout et on se place sur le seed 1
np.random.seed(1)
df = df_ss_col.loc[np.random.permutation(len(df_ss_col))]
#On se débarasse des lignes avec des NaN
clean_df = df.dropna(axis=0)
df = clean_df

#On choisit les colonnes que l'on entraine et celle sur laquelle on teste notre modèle
features = df.columns.to_list()
features.remove('nature')
target = 'nature'

#On normalise
nature = df['nature'] 
df_ss_nature = df.drop(['nature'], axis=1)
df_ss_nature = (df_ss_nature - df_ss_nature.min())/(df_ss_nature.max() - df_ss_nature.min())
df_ss_nature['nature'] = nature
df = df_ss_nature

#On scinde notre df en test et colonne
df_test = df.iloc[0:15000]
df_train = df.iloc[15000:]

#On paramètre notre modèle et on l'entraîne
rfc =  RandomForestClassifier(n_estimators=200, random_state = 0)
rfc.fit(df_train[features], df_train[target])
prediction = rfc.predict(df_test[features])

df_test['predicted_nature'] = prediction
matches = df_test['predicted_nature'] == df_test['nature']
correct_predictions = df_test[matches]
accuracy = len(correct_predictions) / len(df_test)
print(accuracy)

kf = KFold(n_splits = 10, shuffle = True, random_state = 1)
precision = cross_val_score(rfc, df[features], df[target], cv = kf)
#mse_fold = cross_val_score(rfc, df[features], df[target], scoring = 'neg_mean_squared_error', cv = kf) 
#rmse_fold = np.sqrt(np.absolute(mse_fold))


print(precision )



###### PART 7

from sklearn.metrics import mean_squared_error #pour calculer REQM
filename = 'data.csv'
df = pd.read_csv(filename)
#On crée une colonne vitesse du vent ws qui est la sqrt( (uwind)**2 + (vwind)**2 )
df['ws'] = np.sqrt(df['uwind']**2 + df['vwind']**2) #m/s
pd.options.mode.chained_assignment = None

#On remplace les valeurs abérantes par des NaN
df['sst'] = df['sst'].replace(-32767.000000,np.nan)
#On enlève les colonnes qui nous intéressent pas
drop_columns = ['index', 't', 'time', 'name', 'latitude', 'longitude', 'Unnamed: 0']
df_ss_col = df.drop(drop_columns, axis = 1)
#On mélange tout et on se place sur le seed 1
np.random.seed(1)
df = df_ss_col.loc[np.random.permutation(len(df_ss_col))]
#On se débarasse des lignes avec des NaN
clean_df = df.dropna(axis=0)
df = clean_df 

#On choisit les colonnes que l'on entraine et celle sur laquelle on teste notre modèle
features = df.columns.to_list()
features.remove('nature')
target = 'nature'

#On paramètre notre modèle et on l'entraîne
cb =  LogisticRegression(solver = 'liblinear')
cb.fit(df[features], df[target])
prediction =cb.predict(df[features])
df['predicted_nature'] = prediction

#matches
matches = df['predicted_nature'] == df['nature']
correct_predictions = df[matches]
accuracy = len(correct_predictions) / len(df)
print(accuracy)



##### PART 8

filename = 'data.csv'
df = pd.read_csv(filename)
#On crée une colonne vitesse du vent ws qui est la sqrt( (uwind)**2 + (vwind)**2 )
df['ws'] = np.sqrt(df['uwind']**2 + df['vwind']**2) #m/s
pd.options.mode.chained_assignment = None

#On remplace les valeurs abérantes par des NaN
df['sst'] = df['sst'].replace(-32767.000000,np.nan)
#On enlève les colonnes qui nous intéressent pas
drop_columns = ['index', 't', 'time', 'name', 'latitude', 'longitude', 'Unnamed: 0']
df_ss_col = df.drop(drop_columns, axis = 1)
#On mélange tout et on se place sur le seed 1
np.random.seed(1)
df = df_ss_col.loc[np.random.permutation(len(df_ss_col))]
#On se débarasse des lignes avec des NaN
clean_df = df.dropna(axis=0)
df = clean_df

# Application des conditions aux limites
df.loc[df.sst < 300, 'nature'] = 0          
df.loc[df.prcp < 0.001, 'nature'] = 0
df.loc[df.psurf > 100500, 'nature'] = 0
df.loc[df.ws < 7, 'nature'] = 0

#On choisit les colonnes que l'on entraine et celle sur laquelle on teste notre modèle
features = df.columns.to_list()
features.remove('nature')
target = 'nature'

#On normalise
nature = df['nature'] 
df_ss_nature = df.drop(['nature'], axis=1)
df_ss_nature = (df_ss_nature - df_ss_nature.min())/(df_ss_nature.max() - df_ss_nature.min())
df_ss_nature['nature'] = nature
df = df_ss_nature

#On scinde notre df en test et colonne
df_test = df.iloc[0:15000]
df_train = df.iloc[15000:]

#On paramètre notre modèle et on l'entraîne
rfc =  RandomForestClassifier(n_estimators=200, random_state = 0)
rfc.fit(df_train[features], df_train[target])
prediction = rfc.predict(df_test[features])

df_test['predicted_nature'] = prediction
matches = df_test['predicted_nature'] == df_test['nature']
correct_predictions = df_test[matches]
accuracy = len(correct_predictions) / len(df_test)
print(accuracy)