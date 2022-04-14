

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression #Classificateur binaire
from sklearn.ensemble import RandomForestClassifier #Réseau de forêt aléatoire
from sklearn.metrics import mean_squared_error #pour calculer REQM
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def modelisation(name, sst, psurf):
    filename = '{}.csv'.format(name)
    df = pd.read_csv(filename)

    drop_columns = ['time', 'name', 'latitude', 'longitude'] #On enlève les colonnes qui nous intéressent pas
    df_ss_col = df.drop(drop_columns, axis = 1)
    np.random.seed(1) #On mélange tout et on se place sur le seed 1
    df = df_ss_col.loc[np.random.permutation(len(df_ss_col))]

# Application des conditions aux limites
    df.loc[df.sst < sst, 'nature'] = 0          
    df.loc[df.psurf > psurf, 'nature'] = 0

#On choisit les colonnes que l'on entraine et celle sur laquelle on teste notre modèle
    features = df.columns.to_list()
    features.remove('nature')
    target = 'nature'

# On normalise
    nature = df['nature'] 
    df_ss_nature = df.drop(['nature'], axis=1)
    df_ss_nature = (df_ss_nature - df_ss_nature.min())/(df_ss_nature.max() - df_ss_nature.min())
    df_ss_nature['nature'] = nature
    df = df_ss_nature

# On scinde notre df en test et colonne si besoin d'entrainement
    #df_test = df.iloc[0:5000]
    #df_train = df.iloc[5000:]
    
#On paramètre notre modèle et on l'entraîne
    rfc =  RandomForestClassifier(n_estimators=200, random_state = 0, class_weight = 'balanced')
    rfc.fit(df[features], df[target])
    #prediction = rfc.predict(df_test[features])
    #df_test['prediction'] = prediction
    #df_test['predicted_nature'] = prediction
    #matches = df_test['predicted_nature'] == df_test['nature']
    #correct_predictions = df_test[matches]
    #accuracy = len(correct_predictions) / len(df_test)
    #print(accuracy)

    return rfc

