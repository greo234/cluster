# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:58:59 2023

@author: Omoregbe Olotu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#filefolders
indicator = "gdppercapita.xls"
file = 'TotalPopulation.xls'


#reading excel file with the required years 1981 and 2021 for clustering
df = pd.read_excel(indicator, skiprows=3,
                           usecols=[0,25,65])
#drop missing rows
df = df.dropna()
# To reset the indices
df = df.reset_index(drop = True)
print(df)

#visualising data
df.plot('1981', '2021', kind = 'scatter', color = 'blue')
plt.title('GDP per Capita')

#new dataframe for clustering
df_clust = df[['1981','2021']].copy()

#data in array format
c = df[['1981', '2021']].values
print(c)


# Normalize the data
max_value = df_clust.max()
min_value = df_clust.min()
#normalized data
df_norm = (df_clust - min_value) / (max_value - min_value)

#converting to array
n = df_norm.values

#getting the sse for elbow method to find the number of clusters to use
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

#converting to dataframe
cent = pd.DataFrame(cen, columns=['1981','2021'])
print(cent)
print(cent)


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
df_norm['cluster'] = y_predict
print('\n Countries and Cluster Category:\n', df)


#plotting the normalised GDPshowing the cluster centres
plt.figure()
plt.scatter(n[y_predict == 0, 0], n[y_predict == 0, 1], s = 50, c = 'blue',label = 'cluster 0')
plt.scatter(n[y_predict == 1, 0], n[y_predict == 1, 1], s = 50, c = 'red',label = 'cluster 1')
plt.scatter(n[y_predict == 2, 0], n[y_predict == 2, 1], s = 50, c = 'green',label = 'cluster 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 50, c = 'black', label = 'Centroids')
plt.title('GDP/Capita (Normalised)')
plt.xlabel('1981')
plt.ylabel('2021')
plt.legend()
plt.show()


# Logistic function forecasting
def logistic(t, n0, g, t0):
    """ This fuction calculates the logistic function with scale factor n0
    and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f



def read(filename):
    """ This function reads a file in its original and transponsed format"""
    data = pd.read_excel(filename,skiprows=3)
    data.drop(data.columns[[1,2,3]], axis=1, inplace=True)
    return data, data.transpose()
    

#reading the Total population file from the world bank format
pop, pop2 = read(file)
print(pop2)

#rename the columns
df2 = pop2.rename(columns=pop2.iloc[0])

#drop the country name
df2 = df2.drop(index=df2.index[0], axis=0)
df2['Year'] = df2.index

print(df2)

#fitting the data
df_fit = df2[['Year', 'China']].apply(pd.to_numeric, 
                                               errors='coerce')

#fits the logistic data
param, covar = opt.curve_fit(logistic, df_fit['Year'], df_fit['China'], 
                             p0=(3e12, 0.03, 2000))

sigma = np.sqrt(np.diag(covar))

df_fit['fit'] = logistic(df_fit['Year'], *param)

# Plot the forecast
plt.plot(df_fit["Year"], df_fit["China"], 
         label="Population")
plt.plot(df_fit["Year"], df_fit["fit"], label="fit")
plt.xlabel("Year")
plt.ylabel("Population")
plt.legend()
plt.title('CHINA')
plt.show()


# Forecast for the next 10 years
year = np.arange(1960, 2031)
forecast = logistic(year, *param)

plt.plot(df_fit["Year"], df_fit["China"], 
         label="Population")
plt.plot(year, forecast, label="Forecast")
plt.xlabel("Year")
plt.ylabel("Population")
plt.legend()
plt.title('CHINA')
plt.show()

# Error ranges calculation
def err_ranges(x, func, param, sigma):
    """
    This function calculates the upper and lower limits of function, parameters and
    sigmas for a single value or array x. The function values are calculated for 
    all combinations of +/- sigma and the minimum and maximum are determined.
    This can be used for all number of parameters and sigmas >=1.
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    # Create a list of tuples of upper and lower limits for parameters
    uplow = []   
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    # calculate the upper and lower limits
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper

# calculates the error ranges 
low, up = err_ranges(year, logistic, param, sigma)

#plotting China's Total Population
plt.figure(dpi=140)
plt.plot(df_fit["Year"], df_fit["China"], 
         label="Population")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("Year")
plt.ylabel("Population")
plt.legend()
plt.title('China Population Forecast - Confidence Interval')
plt.show()

#prints the error ranges
print(err_ranges(2031, logistic, param, sigma))


#fitting the United States data
df_US = df2[['Year', 'United States']].apply(pd.to_numeric, 
                                               errors='coerce')
#fits the logistic data
param, covar = opt.curve_fit(logistic, df_US['Year'], df_US['United States'], 
                             p0=(3e12, 0.03, 2000))

sigma = np.sqrt(np.diag(covar))

df_US['fit'] = logistic(df_US['Year'], *param)

# Plot the forecast
plt.plot(df_US["Year"], df_US["United States"], 
         label="Population")
plt.plot(df_US["Year"], df_US["fit"], label="fit")
plt.xlabel("Year")
plt.ylabel("Population")
plt.legend()
plt.title('United States')
plt.show()


# Forecast for the next 10 years
year = np.arange(1960, 2031)
forecast = logistic(year, *param)

plt.plot(df_US["Year"], df_US["United States"], 
         label="Population")
plt.plot(year, forecast, label="Forecast")
plt.xlabel("Year")
plt.ylabel("Population")
plt.legend()
plt.title('United States')
plt.show()


