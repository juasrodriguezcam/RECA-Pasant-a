# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from mpl_toolkits import mplot3d
from hdbscan import HDBSCAN

#Cargamos el dataframe
#_________________________________________________________________________________________

#Importamos el archivo .csv a través de Pandas
data = pd.read_csv(r'Taurus_ANG3MT.csv')
#Se transforma la información en un DataFrame
df_cluster = pd.DataFrame(data, columns = ['pmra', 'pmdec', 'parallax', 'lgg'])
#Encontramos las filas que tengan valores NaN 
nan_rows = np.where(np.isnan(df_cluster) == True)
#Eliminamos esas filas para evitar errores
df_cluster = df_cluster.drop(nan_rows[0], axis = 0)

indexes = np.empty((1,2))
for i in range(5,105,5):
    for j in range(5,105,5):
        #Instanciamos el algoritmo
        #_________________________________________________________________________________________
        hdbscan = HDBSCAN(min_cluster_size=i,
                         min_samples=j)
                         
        #Entrenamos y predecimos
        #_________________________________________________________________________________________
        preds_2 = hdbscan.fit_predict(df_cluster)
        
        
        #Métricas
        #_________________________________________________________________________________________
        a = silhouette_score(df_cluster, preds_2)
        calinski_harabasz_score(df_cluster, preds_2)
        
        #Instanciamos el algoritmo
        #_________________________________________________________________________________________
        hdbscan_1 = HDBSCAN(min_cluster_size=i,
                         min_samples=j,
                         cluster_selection_method = 'leaf')
                         
        #Entrenamos y predecimos
        #_________________________________________________________________________________________
        preds_3 = hdbscan_1.fit_predict(df_cluster)
        
        #Métricas
        #_________________________________________________________________________________________
        b = silhouette_score(df_cluster, preds_3)
        calinski_harabasz_score(df_cluster, preds_3)
        
        indexes = np.append(indexes, [[a,b]], axis = 0)
        
print(df_cluster)
#Graficamos
#_________________________________________________________________________________________
fig, axes = plt.pyplot.subplots(nrows=2, ncols=3)
df_cluster.plot(ax=axes[0,0], kind='scatter', x='pmra', y='pmdec', c=preds_2,
                cmap='Accent_r', figsize=(16,10)); axes[0,0].set_title('A')

df_cluster.plot(ax=axes[0,1], kind='scatter', x='pmra', y='parallax', c=preds_2,
                cmap='Accent_r', figsize=(16,10)); axes[0,1].set_title('B')

df_cluster.plot(ax=axes[0,2], kind='scatter', x='pmra', y='lgg', c=preds_2,
                cmap='Accent_r', figsize=(16,10)); axes[0,2].set_title('C')

df_cluster.plot(ax=axes[1,0], kind='scatter', x='pmdec', y='parallax', c=preds_2,
                cmap='Accent_r', figsize=(16,10)); axes[1,0].set_title('D')

df_cluster.plot(ax=axes[1,1], kind='scatter', x='pmdec', y='lgg', c=preds_2,
                cmap='Accent_r', figsize=(16,10)); axes[1,1].set_title('E')

df_cluster.plot(ax=axes[1,2], kind='scatter', x='pmra', y='pmdec', c=preds_3,
                cmap='Accent_r', figsize=(16,10)); axes[1,2].set_title('F')
pass



