# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:58:59 2023

@author: matah
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

indicator = "gdppercapita.xls"

#reading excel file with the required years 1981 and 2021
df = pd.read_excel(indicator, skiprows=3,
                           usecols=[0,25,65])
#drop missing rows
df = df.dropna()
# To reset the indices
df = df.reset_index(drop = True)
print(df)

#visualising data
#df.plot('1981', '2021', kind = 'scatter', color = 'blue')
#plt.title('GDP per Capita')

#new dataframe for clustering
df_clust = df[['1981','2021']].copy()


# Normalize the data
max_value = df_clust.max()
min_value = df_clust.min()
#normalized data
df_norm = (df_clust - min_value) / (max_value - min_value)
print(df_norm)

sse = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_norm)
    sse.append(kmeans.inertia_)
    
#plotting to check for appropriate number of clusters using elbow method
plt.figure(dpi=140)
plt.plot(range(1, 11), sse, color = 'red')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.savefig('clust.png')
plt.show()    

#finding the Kmeans clusters
ncluster = 3
kmeans = KMeans(n_clusters=ncluster, init='k-means++', max_iter=300, n_init=10,
                random_state=0)

# Fit the model to the data
kmeans.fit(df_norm)

#labels
labels = kmeans.labels_

# Use the silhouette score to evaluate the quality of the clusters
print(f'Silhouette Score: {silhouette_score(df_norm, labels)}')

#Extracting clusters centers
cen = kmeans.cluster_centers_
print(cen)

cen = pd.DataFrame(cen, columns=['1981','2021'])


# Plot the scatter plot of the clusters
plt.scatter(df_norm['1981'], df_norm['2021'], c=labels)
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

#Getting the Kmeans
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, 
                random_state=0)
y_predict = kmeans.fit_predict(df_norm)
print(y_predict)

#creating new dataframe with the labels for each country
df['cluster'] = y_predict
print(df)


# plot using the labels to select colour
plt.figure(figsize=(10.0, 10.0))
col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
for l in range(ncluster): # loop over the different labels
    plt.plot(df_norm[labels==l]["1981"], df_norm[labels==l]["2021"], "o", markersize=3, color=col[l])
#
# show cluster centres
for ib in range(ncluster):
    xb, yb = cen[ib,:]
    plt.plot(xb, yb, "dk", markersize=10)
    plt.xlabel("1981")
    plt.ylabel("2021")
    plt.show()