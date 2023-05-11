# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cluster_tools as ct
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import errors as err
import scipy.optimize as opt

# Load the climate change data from the World Bank dataset
url = 'Climate.csv'
df = pd.read_csv(url, skiprows=4)

# Select relevant columns for analysis
columns = ['Country Name', "Country Code", 'Indicator Name', '1980', '2020']
df = df[columns]

# Print the DataFrame containing the selected columns
print(df)

# Create a DataFrame for the 'Forest area (% of land area)' indicator
forest = ['Forest area (% of land area)']
df_forest = df.loc[df['Indicator Name'].isin(forest)]

# Print the DataFrame containing the 'Forest area (% of land area)' indicator
print(df_forest)

# Create a DataFrame for the 'Arable land (% of land area)' indicator
Arable = ['Arable land (% of land area)']
df_arable = df.loc[df['Indicator Name'].isin(Arable)]

# Print the DataFrame containing the 'Arable land (% of land area)' indicator
print(df_arable)

# Print summary statistics for the 'Arable land (% of land area)' DataFrame
print(df_arable.describe())

# Print summary statistics for the 'Forest area (% of land area)' DataFrame
print(df_forest.describe())


# Drop rows with NaN values in the '2020' column of the 'df_arable' DataFrame
df_arable = df_arable[df_arable["2020"].notna()]

# Print summary statistics for the 'df_arable' DataFrame
print(df_arable.describe())

# Drop rows with NaN values in the '2020' column of the 'df_forest' DataFrame
# using an alternative method
df_forest = df_forest.dropna(subset=["2020"])

# Print summary statistics for the 'df_forest' DataFrame
print(df_forest.describe)

# Create new DataFrames containing only the relevant columns for '2020'
df_arable2020 = df_arable[["Country Name", "Country Code", "2020"]].copy()
df_forest2020 = df_forest[["Country Name", "Country Code", "2020"]].copy()

# Print summary statistics for the 'df_arable2020' DataFrame
print(df_arable2020.describe())

# Print summary statistics for the 'df_forest2020' DataFrame
print(df_forest2020.describe())

# Merge the 'df_arable2020' and 'df_forest2020' DataFrames based on
# 'Country Name'. The 'how="outer"' argument specifies that entries
# not found in both DataFrames should be included
df_2020 = pd.merge(df_arable2020, df_forest2020,
                   on="Country Name", how="outer")

# Print summary statistics for the 'df_2020' DataFrame
print(df_2020.describe())

# Save the 'df_2020' DataFrame to an Excel file named "agr_for2020.xlsx"
df_2020.to_excel("agr_for2020.xlsx")

# Drop entries with NaN values from the 'df_2020' DataFrame
# Entries with only one data point or less are considered useless
df_2020 = df_2020.dropna()

# Print summary statistics for the updated 'df_2020' DataFrame
print(df_2020.describe())

# Rename columns in the 'df_2020' DataFrame
df_2020 = df_2020.rename(columns={"2020_x": "Arable", "2020_y": "Forest"})

# Print the updated 'df_2020' DataFrame
print(df_2020)

# Create a scatter matrix plot of the 'df_2020' DataFrame
pd.plotting.scatter_matrix(df_2020, figsize=(10, 6), s=10, alpha=0.8)

# Compute the correlation matrix of the 'df_2020' DataFrame
print(df_2020.corr())

# Create a new DataFrame containing only the 'Arable' and 'Forest' columns
# of the 'df_2020' DataFrame
df_cluster = df_2020[["Arable", "Forest"]].copy()


# normalise the data using the scaler function from the ct module, this ensures
# that each feature has the same scale, which is necessary for clustering
df_cluster, df_min, df_max = ct.scaler(df_cluster)

# print the silhouette score for different numbers of clusters
print("n    score")
for ncluster in range(2, 10):
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)

    # fit the data to the clusterer, storing the results in the kmeans object
    kmeans.fit(df_cluster)

    # get the labels assigned to each data point by the clusterer
    labels = kmeans.labels_

    # get the estimated cluster centers
    cen = kmeans.cluster_centers_

    # calculate the silhouette score for the current number of clusters
    score = skmet.silhouette_score(df_cluster, labels)

    # print the number of clusters and the silhouette score
    print(ncluster, score)

# set the number of clusters to 5 (chosen based on the silhouette scores)
ncluster = 5

# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=ncluster)

# fit the data to the clusterer, storing the results in the kmeans object
kmeans.fit(df_cluster)

# get the labels assigned to each data point by the clusterer
labels = kmeans.labels_

# get the estimated cluster centers
cen = kmeans.cluster_centers_

# get the x-coordinates of the cluster centers
xcen = cen[:, 0]

# get the y-coordinates of the cluster centers
ycen = cen[:, 1]


# create a figure cluster by cluster
plt.figure(figsize=(10.0, 6.0))

cm = plt.cm.get_cmap('tab10')
plt.scatter(df_cluster["Arable"], df_cluster["Forest"],
            40, labels, marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.xlabel("Arable")
plt.ylabel("Forest")
plt.show()


# move the cluster centres to the original scale
cen = ct.backscale(cen, df_min, df_max)
xcen = cen[:, 0]
ycen = cen[:, 1]

# create a figure cluster by cluster
plt.figure(figsize=(10.0, 6.0))

cm = plt.cm.get_cmap('tab10')
plt.scatter(df_2020["Arable"], df_2020["Forest"],
            40, labels, marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.xlabel("Arable")
plt.ylabel("Forest")
plt.show()

# Define a linear model


def linear_model(x, m, c):
    return m * x + c


# Define the variables to be fitted
x_data = df_2020["Arable"]
y_data = df_2020["Forest"]

df_arable = pd.read_csv("SL_Arable.csv")

df_arable.plot("Year", "Arable land (% of land area)")
plt.show()


def exponential(t, n0, g):
    """
    Calculates exponential function with scale factor n0 and growth rate g.
    """

    t = t - 1990
    f = n0 * np.exp(g*t)

    return f


print(type(df_arable["Year"].iloc[1]))
df_arable["Year"] = pd.to_numeric(df_arable["Year"])
print(type(df_arable["Year"].iloc[1]))
param, covar = opt.curve_fit(
    exponential, df_arable["Year"], df_arable["Arable land (% of land area)"],
    p0=(1.2e12, 0.03))

print("Arable land (% of land area) 1990", param[0]/1e9)
print("growth rate", param[1])


df_arable["fit"] = exponential(df_arable["Year"], *param)

df_arable.plot("Year", ["Arable land (% of land area)", "fit"])
plt.show()


def logistic(t, n0, g, t0):
    """
    Calculates the logistic function with scale factor n0 and growth rate g
    """

    f = n0 / (1 + np.exp(-g*(t - t0)))

    return f


param, covar = opt.curve_fit(logistic, df_arable["Year"], 
                             df_arable["Arable land (% of land area)"],
                             p0=(1, 0.03, 1990.0))

sigma = np.sqrt(np.diag(covar))

df_arable["fit"] = logistic(df_arable["Year"], *param)

df_arable.plot("Year", ["Arable land (% of land area)", "fit"])
plt.show()

print("turning point", param[2], "+/-", sigma[2])
print("Arable land (% of land area) at turning point",
      param[0]/1e9, "+/-", sigma[0]/1e9)
print("growth rate", param[1], "+/-", sigma[1])


year = np.arange(1960, 2031)
forecast = logistic(year, *param)

plt.figure()
plt.plot(df_arable["Year"], df_arable["Arable land (% of land area)"],
         label="Arable land (% of land area)")
plt.plot(year, forecast, label="forecast")

plt.xlabel("year")
plt.ylabel("Arable land (% of land area)")
plt.legend()
plt.show()

low, up = err.err_ranges(year, logistic, param, sigma)


def poly(x, a, b, c, d, e):
    """
    Calulates polynominal
    """
    x = x - 1990
    f = a + b*x + c*x**2 + d*x**3 + e*x**4

    return f


param, covar = opt.curve_fit(
    poly, df_arable["Year"], df_arable["Arable land (% of land area)"])

sigma = np.sqrt(np.diag(covar))
print(sigma)
year = np.arange(1960, 2031)
forecast = poly(year, *param)
low, up = err.err_ranges(year, poly, param, sigma)

df_arable["fit"] = poly(df_arable["Year"], *param)

plt.figure()
plt.plot(df_arable["Year"], df_arable["Arable land (% of land area)"],
         label="Arable land (% of land area)")
plt.plot(year, forecast, label="forecast")

plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("Arable land (% of land area)")
plt.legend()
plt.show()
