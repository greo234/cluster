# -*- coding: utf-8 -*-
"""
An analysis of how population growth affects GDP per capita using Kmeans 
clustering for classification of Total Population and predicting GDP per
capita with curve fitting models


Created on Sun Jan 22 13:44:47 2023

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
file1 = 'TotalPopulation.xls'
file2 = "gdppercapita.xls"


def read(filename):
    """ This function reads a file in its original and transponsed format"""
    data = pd.read_excel(filename, skiprows=3)
    data.drop(data.columns[[1, 2, 3]], axis=1, inplace=True)
    return data, data.transpose()


# reading the Total Population file where pop is the original format and popT the transposed format
pop, popT = read(file1)
print(pop)

year1 = '1981'
year2 = '2021'


# listing regional countries that may distort in data
regional_list = ['Africa Eastern and Southern', 'North America', 'Arab World', 'Caribbean small states', 'Central African Republic', 'Central Europe and the Baltics',
                 'Early-demographic dividend', 'East Asia & Pacific',
                 'East Asia & Pacific (IDA & IBRD countries)',
                 'East Asia & Pacific (excluding high income)', 'Europe & Central Asia',
                 'Europe & Central Asia (IDA & IBRD countries)',
                 'Europe & Central Asia (excluding high income)', 'European Union',
                 'Fragile and conflict affected situations', 'French Polynesia', 'Heavily indebted poor countries (HIPC)',
                 'High income', 'IBRD only',
                 'IDA & IBRD total', 'IDA blend', 'IDA only', 'IDA total', 'Late-demographic dividend',
                 'Latin America & Caribbean',
                 'Latin America & Caribbean (excluding high income)',
                 'Latin America & the Caribbean (IDA & IBRD countries)',
                 'Least developed countries: UN classification', 'Low & middle income', 'Low income', 'Lower middle income',
                 'Middle East & North Africa',
                 'Middle East & North Africa (IDA & IBRD countries)',
                 'Middle East & North Africa (excluding high income)',
                 'Middle income', 'Not classified',
                 'OECD members', 'Other small states',
                 'Pacific island small states', 'Post-demographic dividend',
                 'Pre-demographic dividend', 'Small states', 'South Asia (IDA & IBRD)', 'Sub-Saharan Africa',
                 'Sub-Saharan Africa (IDA & IBRD countries)',
                 'Sub-Saharan Africa (excluding high income)', 'Upper middle income', 'West Bank and Gaza',
                 'World', 'Africa Western and Central', 'South Asia', 'Euro area']

# extract the required data for the clustering and drop missing values
pop_list = pop.loc[pop.index, ['Country Name', year1, year2]].dropna()
print(pop_list)

# excluding regional data
pop_clust = pop_list[~pop_list['Country Name'].isin(regional_list)]
print(pop_clust)

# convert the dataframe to an array
p = pop_clust[[year1, year2]].values
print(p)

# visualising data
plt.figure(dpi=140)
pop_clust.plot(year1, year2, kind='scatter', color='blue')
plt.title('Total Population')


# Normalize the data
max_value = p.max()
min_value = p.min()
# normalized data
p_norm = (p - min_value) / (max_value - min_value)

print('\n Normalised Population:\n', p_norm)

sse = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit(p_norm)
    sse.append(kmeans.inertia_)

# plotting to check for appropriate number of clusters using elbow method
plt.figure(dpi=140)
plt.plot(range(1, 11), sse, color='red')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.savefig('clust.png')
plt.show()

# finding the Kmeans clusters
ncluster = 3
kmeans = KMeans(n_clusters=ncluster, init='k-means++', max_iter=300, n_init=10,
                random_state=0)

# Fit the model to the data
kmeans.fit(p_norm)

# labels
labels = kmeans.labels_

# Use the silhouette score to evaluate the quality of the clusters
print(f'Silhouette Score: {silhouette_score(p_norm, labels)}')

# Extracting clusters centers
cen = kmeans.cluster_centers_
print(cen)


# Plot the scatter plot of the clusters
plt.figure(dpi=140)
plt.scatter(p_norm[:, 0], p_norm[:, 1], c=kmeans.labels_)
plt.title('K-Means Clustering')
plt.xlabel('1981')
plt.ylabel('2021')
plt.show()

# Getting the Kmeans
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10,
                random_state=0)
y_predict = kmeans.fit_predict(p_norm)
print(y_predict)

# creating new dataframe with the labels for each country
pop_clust['cluster'] = y_predict
print('\n Countries and Cluster Category:\n', pop_clust)


# plotting the normalised Population
plt.figure(dpi=140)
plt.scatter(p_norm[y_predict == 0, 0], p_norm[y_predict ==
            0, 1], s=50, c='blue', label='cluster 0')
plt.scatter(p_norm[y_predict == 1, 0], p_norm[y_predict ==
            1, 1], s=50, c='red', label='cluster 1')
plt.scatter(p_norm[y_predict == 2, 0], p_norm[y_predict ==
            2, 1], s=50, c='green', label='cluster 2')
plt.scatter(cen[:, 0], cen[:, 1], s=50, c='black', label='Centroids')
plt.title('Total Population (Normalised)')
plt.xlabel('1981')
plt.ylabel('2021')
plt.legend()
plt.show()

# converting centroid to it's unnormalized form
cent = cen * (max_value - min_value) + min_value

# plotting the Population in their clusters with the centroid points
plt.figure(dpi=140)
plt.scatter(p[y_predict == 0, 0], p[y_predict == 0, 1],
            s=50, c='blue', label='cluster 0')
plt.scatter(p[y_predict == 1, 0], p[y_predict == 1, 1],
            s=50, c='red', label='cluster 1')
plt.scatter(p[y_predict == 2, 0], p[y_predict == 2, 1],
            s=50, c='green', label='cluster 2')
plt.scatter(cent[:, 0], cent[:, 1], s=50, c='black', label='Centroids')
plt.title('Total Population')
plt.xlabel('1981')
plt.ylabel('2021')
plt.legend()
plt.show()

# reading the Total population file from the world bank format
pop, pop2 = read(file1)
print(pop2)

# rename the the transposed data columns for population
df2 = pop2.rename(columns=pop2.iloc[0])

# drop the country name
df2 = df2.drop(index=df2.index[0], axis=0)
df2['Year'] = df2.index

print(df2)

# fitting the for China's population data
df_fit = df2[['Year', 'China']].apply(pd.to_numeric,
                                      errors='coerce')


# Logistic function for curve fitting and forecasting the Total Population
def logistic(t, n0, g, t0):
    """ This fuction calculates the logistic function with scale factor n0
    and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f

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
    for p, s in zip(param, sigma):
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


# fits the logistic data
param_ch, covar_ch = opt.curve_fit(logistic, df_fit['Year'], df_fit['China'],
                                   p0=(3e12, 0.03, 2000))

# calculating the standard deviation
sigma_ch = np.sqrt(np.diag(covar_ch))

# creating a new column with the fit data
df_fit['fit'] = logistic(df_fit['Year'], *param_ch)

# Forecast for the next 10 years
year = np.arange(1960, 2031)
forecast = logistic(year, *param_ch)

# calculates the error ranges
low_ch, up_ch = err_ranges(year, logistic, param_ch, sigma_ch)

# plotting China's Total Population
plt.figure(dpi=140)
plt.plot(df_fit["Year"], df_fit["China"],
         label="Population")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low_ch, up_ch, color="yellow",
                 alpha=0.7, label='Error ranges')
plt.xlabel("Year")
plt.ylabel("Population")
plt.legend()
plt.title('China Population Forecast')
plt.show()

# prints the error ranges
print(err_ranges(2031, logistic, param_ch, sigma_ch))

# fitting the United State's Population data
us = df2[['Year', 'United States']].apply(pd.to_numeric,
                                          errors='coerce')

# fits the US logistic data
param_us, covar_us = opt.curve_fit(logistic, us['Year'], us['United States'],
                                   p0=(3e12, 0.03, 2031))

# calculates the standard deviation for United States data
sigma_us = np.sqrt(np.diag(covar_us))

# Forecast for the next 10 years
forecast_us = logistic(year, *param_us)

# calculate error ranges
low_us, up_us = err_ranges(year, logistic, param_us, sigma_us)

# plotting United State's Total Population
plt.figure(dpi=140)
plt.plot(us["Year"], us["United States"],
         label="Population")
plt.plot(year, forecast_us, label="forecast")
plt.fill_between(year, low_us, up_us, color="yellow",
                 alpha=0.7, label="Error Ranges")
plt.xlabel("Year")
plt.ylabel("Population")
plt.legend()
plt.title('US Population Forecast - Confidence Interval')
plt.show()

# fitting the data
Gh = df2[['Year', 'Ghana']].apply(pd.to_numeric,
                                  errors='coerce')

# fits the Ghana's logistic data
param_gh, covar_gh = opt.curve_fit(logistic, Gh['Year'], Gh['Ghana'],
                                   p0=(3e12, 0.03, 2031))

# sigma is the standard deviation
sigma_gh = np.sqrt(np.diag(covar_gh))

# Forecast for the next 10 years
forecast_gh = logistic(year, *param_gh)

# calculate error ranges
low_gh, up_gh = err_ranges(year, logistic, param_gh, sigma_gh)

# creates a new column for the fit data
Gh['fit'] = logistic(Gh['Year'], *param_gh)

# plotting Ghana's Total Population
plt.figure(dpi=140)
plt.plot(Gh["Year"], Gh["Ghana"],
         label="Population")
plt.plot(year, forecast_gh, label="forecast")
plt.fill_between(year, low_gh, up_gh, color="yellow",
                 alpha=0.7, label='Error Ranges')
plt.xlabel("Year")
plt.ylabel("Population")
plt.legend()
plt.title('Ghana Population Forecast ')
plt.show()

#reading the GDP/Capita file from the world bank format
gdp, gdpT = read(file2)
print(gdpT)

#rename the columns
g = gdpT.rename(columns=gdpT.iloc[0])

#drop the country name
g = g.drop(index=g.index[0], axis=0)
g['Year'] = g.index

#fitting the data
g_chi = g[['Year', 'China']].apply(pd.to_numeric, 
                                               errors='coerce')

# poly function for forecasting GDP/Capita
def poly(x, a, b, c):
    
    """ Function for fitting
    x: independent variable
    a, b, c: parameters to be fitted
    """
    return a*x**2 + b*x + c

#fits the linear data
param_cg, cov_cg = opt.curve_fit(poly, g_chi['Year'], g_chi['China'] )

#calculates the standard deviation
sigma_cg = np.sqrt(np.diag(cov_cg))

#creates a new column for the fit figures
g_chi['fit'] = poly(g_chi['Year'], *param_cg)

#forecasting the fit figures
forecast_cg = poly(year, *param_cg)

# calculate error ranges 
low_cg, up_cg = err_ranges(year, poly, param_cg, sigma_cg)

#Plotting
plt.plot(g_chi["Year"], g_chi["China"], 
         label="Population")
plt.plot(year, forecast_cg, label="Forecast")

plt.xlabel("Year")
plt.ylabel("Population")
plt.legend()
plt.title('China')
plt.show()


#fitting the data
g_us = g[['Year', 'United States']].apply(pd.to_numeric, 
                                               errors='coerce')

#fits the linear data
param_usg, cov_usg = opt.curve_fit(poly, g_us['Year'], g_us['United States'], 
                )
#calculates the standard deviation
sigma_usg = np.sqrt(np.diag(cov_usg))

#creates a column for the fit data
g_us['fit'] = poly(g_us['Year'], *param_usg)

#forecasting for the next 10 years
forecast_usg = poly(year, *param_usg)

#plotting
plt.plot(g_us["Year"], g_us["United States"], 
         label="Population")
plt.plot(year, forecast_usg, label="Forecast")
plt.xlabel("Year")
plt.ylabel("GDP per Capita ('US$')")
plt.legend()
plt.title('United States')
plt.show()

#fitting the data
g_gh = g[['Year', 'Ghana']].apply(pd.to_numeric, 
                                               errors='coerce')

#fits the linear data
param_ghg, cov_ghg = opt.curve_fit(poly, g_gh['Year'], g_gh['Ghana'])

#calculates the standard deviation
sigma_ghg = np.sqrt(np.diag(cov_ghg))

#creates a new column for the fit data
g_gh['fit'] = poly(g_gh['Year'], *param_ghg)

#forescast paramaters for the next 10 years
forecast_ghg = poly(year, *param_ghg)

#plotting
plt.plot(g_gh["Year"], g_gh["Ghana"], 
         label="Population")
plt.plot(year, forecast_ghg, label="Forecast")
plt.xlabel("Year")
plt.ylabel("GDP per Capita ('US$')")
plt.legend()
plt.title('Ghana')
plt.show()
