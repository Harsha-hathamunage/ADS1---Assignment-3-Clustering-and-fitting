# -*- coding: utf-8 -*-
"""
Created on Mon May  8 18:14:37 2023

@author: LENOVO
"""

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from scipy.stats import skew, kurtosis


# =============================================================================
# This section consist of all the function definitions
# =============================================================================


# create function for read file
def read_electricity_production_data(filename):
    """
    Docstring
    """

    # read data from csv
    df_data = pd.read_csv(filename, skiprows=4)

    # create a new dataframe to filter five usefull indicators
    df_electricity_production = df_data[
        (df_data["Indicator Name"] == "Renewable electricity output (% of total electricity output)") |
        (df_data["Indicator Name"] == "Electricity production from oil sources (% of total)") |
        (df_data["Indicator Name"] == "Electricity production from nuclear sources (% of total)") |
        (df_data["Indicator Name"] == "Electricity production from natural gas sources (% of total)") |
        (df_data["Indicator Name"] == "Electricity production from coal sources (% of total)") |
        (df_data["Indicator Name"] ==
         "Electricity production from hydroelectric sources (% of total)")
    ].reset_index(drop=True)

    df_electricity_production["Indicator Name"] = df_electricity_production["Indicator Name"].replace(
        ["Renewable electricity output (% of total electricity output)",
         "Electricity production from oil sources (% of total)",
         "Electricity production from nuclear sources (% of total)",
         "Electricity production from natural gas sources (% of total)",
         "Electricity production from coal sources (% of total)",
         "Electricity production from hydroelectric sources (% of total)"],
        ["Renewable",
         "Oil Sources",
         "Nuclear Sources",
         "Natural Gas Sources",
         "Coal Sources",
         "Hydroelectric Sources"
         ])

    # drop all unnecessary columns
    df_electricity_production = df_electricity_production.drop(["Indicator Code",
                                                                "Unnamed: 66",
                                                                "2015",
                                                                "2016",
                                                                "2017",
                                                                "2018",
                                                                "2019",
                                                                "2020",
                                                                "2021"], axis=1)

    # drop the years between 1960 to 1990
    df_electricity_production = df_electricity_production.drop(df_electricity_production.iloc[:, 3:42],
                                                               axis=1)

    # create a dataframe to get years as columns
    df_year = df_electricity_production.copy()

    # remove all NaNs to clean the dataframe
    df_year = df_electricity_production.dropna(axis=0)

    # set the country name as index
    df_electricity_production = df_electricity_production.set_index(
        "Country Name")

    # transpose the dataframe to get countries as columns
    df_country = df_electricity_production.transpose()

    # clean the transposed dataframe
    df_country = df_country.dropna(axis=1)

    # return both year and country dataframes
    return df_year, df_country


# function to extract data for specific countries
def extract_country_data(country_name):
    """
    This function get the country name as an argument and create a new
    dataframe with data of given country
    """

    # extract the given country data
    df_state = df_country[country_name]

    # use iloc to extract columns for new df
    df_cols = df_state.iloc[1]

    # take the data less the header row
    df_state = df_state[2:]

    # assign new columns to the df
    df_state.columns = df_cols

    # convert data types to numeric
    df_state = df_state.apply(pd.to_numeric, errors="coerce")

    # return the dataframe
    return df_state


# function to compare statistical properties of each indicators per state
def individual_country_statisctic(country_name):
    """
    This function get the country name as an argument and produce the
    comparison of the statistical properties of indicators for given country
    """

    # call thefunction to create country dataframe
    df_state = extract_country_data(country_name)

    # extract statistical properties
    df_describe = df_state.describe().round(2)

    # extract column headers
    cols = df_describe.columns

    # get the half of length of ech column
    lencols = [int(len(c)/2) for c in cols]

    # get column names into 2 lines based on the length of each column
    df_describe.columns = pd.MultiIndex.from_tuples(tuple((c[:ln], c[ln:])
                                                          for c, ln in zip(
        cols,
        lencols)
    ))

    # print the statistics
    print("========== The summary statistics for", country_name, "===========")
    print("\n")
    print(df_describe)
    print("==================================================================")
    print("\n")

    # return the statistics for each country
    return


# function to compare statistical properties of each countries per indicator
def individual_indicator_statistics(indicator_name):
    """
    This function get the indicator name as an argument and produce the
    comparison of the statistical properties of countries for given indicator
    """

    # extract given indicator data
    df_indicator = df_year[df_year["Indicator Name"] == indicator_name]

    # drop unneccesary columns
    df_indicator = df_indicator.drop(["Country Code", "Indicator Name"],
                                     axis=1)

    # extract the useful countries for further analysis
    df_indicator = df_indicator[
        (df_indicator["Country Name"] == "Netherlands") |
        (df_indicator["Country Name"] == "Finland") |
        (df_indicator["Country Name"] == "Germany") |
        (df_indicator["Country Name"] == "Spain") |
        (df_indicator["Country Name"] == "Russian Federation")
    ].reset_index(drop=True)

    # set the country name as index
    df_indicator = df_indicator.set_index("Country Name")

    # transpose the df
    df_indicator = df_indicator.T

    # extract statistical properties
    df_describe = df_indicator.describe().round(2)

    # print the statistics
    print("========= The summary statistics for", indicator_name, "==========")
    print("\n")
    print(df_describe)
    print("==================================================================")
    print("\n")

    # return the statistics for each country
    return


# function to get correlation over time
def correlation_per_year(country_name):
    """
    This function get the country name as an argument and produce the
    correlation over time for selected indicators
    """

    # define the window size
    window_size = 5

    # extract the country data
    df_data = df_countries[df_countries["Country Name"] == country_name]

    # filter the dataframe for the indicators you want to analyze
    indicators = ['Renewable',
                  'Oil Sources',
                  'Nuclear Sources']

    df_filtered = df_data[df_data['Indicator Name'].isin(indicators)]

    df_filtered['Year'] = df_filtered['Year'].astype(int)

    # pivot the dataframe to have year as index and indicators as columns
    df_pivot = df_filtered.pivot(index='Year',
                                 columns='Indicator Name',
                                 values='Total')

    # calculate the correlation matrix for each rolling window
    corr_matrix_over_time = df_pivot.rolling(window_size).corr()

    # select only the correlations between different indicators
    corr_matrix_over_time = corr_matrix_over_time.unstack().iloc[:,
                                                                 window_size-4::window_size]

    # drop null values
    corr_matrix_over_time.dropna()

    # print the result
    print(corr_matrix_over_time)

    # end the function
    return


# function to create multiple line charts for electricity production from oil source
def plt_oil_sources_line_chart(df):
    """ 
    Docstring
    """

    # create dataframes for countries
    df_Netherlands = df[df["Country Name"] == "Netherlands"]
    df_Russia = df[df["Country Name"] == "Russian Federation"]
    df_Finland = df[df["Country Name"] == "Finland"]
    df_Spain = df[df["Country Name"] == "Spain"]
    df_Germany = df[df["Country Name"] == "Germany"]
    df_UK = df[df["Country Name"] == "United Kingdom"]

    # make the figure
    plt.figure()

    # use multiple x and y for plot multiple lines
    plt.plot(df_Netherlands["Year"],
             df_Netherlands["Total"], label="Netherlands")
    plt.plot(df_Finland["Year"], df_Finland["Total"], label="Finland")
    plt.plot(df_Russia["Year"], df_Russia["Total"], label="Russia")
    plt.plot(df_Spain["Year"], df_Spain["Total"], label="Spain")
    plt.plot(df_Germany["Year"], df_Germany["Total"], label="Germany")
    plt.plot(df_UK["Year"], df_UK["Total"], label="UK")

    # labeling
    plt.xlabel("Year", labelpad=(10), fontweight="bold")
    plt.ylabel("Electricity production(% of total)",
               labelpad=(10), fontweight="bold")

    # add a title and legend
    plt.title("Total electricity production from oil sources by country ",
              fontweight="bold", y=1.1)
    plt.legend(loc='center left',
               bbox_to_anchor=(1, 0.5),
               fancybox=True,
               shadow=True)

    plt.xticks(rotation=90)

    # save the plot as png
    plt.savefig("oil_sources_line_chart.png")

    # show the plot
    plt.show()

    # end the function
    return


# function to create multiple line charts for Renewable electricity output (% of total electricity output)
def plot_renewable_electricity_line_chart(df):
    """ 
    docstring
    """

    # create dataframes for countries
    df_Netherlands = df[df["Country Name"] == "Netherlands"]
    df_Russia = df[df["Country Name"] == "Russian Federation"]
    df_Finland = df[df["Country Name"] == "Finland"]
    df_Spain = df[df["Country Name"] == "Spain"]
    df_Germany = df[df["Country Name"] == "Germany"]
    df_UK = df[df["Country Name"] == "UK"]

    # make the figure
    plt.figure()

    # use multiple x and y for plot multiple lines
    plt.plot(df_Netherlands["Year"],
             df_Netherlands["Total"], label="Netherlands")
    plt.plot(df_Finland["Year"], df_Finland["Total"], label="Finland")
    plt.plot(df_Russia["Year"], df_Russia["Total"], label="Russia")
    plt.plot(df_Spain["Year"], df_Spain["Total"], label="Spain")
    plt.plot(df_Germany["Year"], df_Germany["Total"], label="Germany")
    plt.plot(df_UK["Year"], df_UK["Total"], label="UK")

    # labeling
    plt.xlabel("Year", labelpad=(10), fontweight="bold")
    plt.ylabel("Electricity production(% of total)",
               labelpad=(10), fontweight="bold")

    # add a title and legend
    plt.title("Renewable electricity output (% of total electricity output) by country ",
              fontweight="bold", y=1.1)
    plt.legend(loc='center left',
               bbox_to_anchor=(1, 0.5),
               fancybox=True,
               shadow=True)

    plt.xticks(rotation=90)

    # save the plot as png
    plt.savefig("renewable_line_chart.png")

    # show the plot
    plt.show()

    # end the function
    return


# function to create multiple line charts for electricity production from natural gas
def plot_natural_gas_line_chart(df):
    """ 
    docstring
    """

    # create dataframes for countries
    df_Switzerland = df[df["Country Name"] == "Switzerland"]
    df_Netherlands = df[df["Country Name"] == "Netherlands"]
    df_Finland = df[df["Country Name"] == "Finland"]
    df_Russia = df[df["Country Name"] == "Russian Federation"]
    df_Spain = df[df["Country Name"] == "Spain"]
    df_japan = df[df["Country Name"] == "Japan"]
    df_Germany = df[df["Country Name"] == "Germany"]
    df_UK = df[df["Country Name"] == "United Kingdom"]

    # make the figure
    plt.figure()

    # use multiple x and y for plot multiple lines
    plt.plot(df_Switzerland["Year"],
             df_Switzerland["Total"],
             linestyle='dashed',
             label="Switzerland")
    plt.plot(df_Netherlands["Year"],
             df_Netherlands["Total"],
             linestyle='dashed',
             label="Netherlands")
    plt.plot(df_Finland["Year"],
             df_Finland["Total"],
             label="Finland")
    plt.plot(df_Russia["Year"],
             df_Russia["Total"],
             linestyle='dashed',
             label="Russia")
    plt.plot(df_Spain["Year"],
             df_Spain["Total"],
             linestyle='dashed',
             label="Spain")
    plt.plot(df_japan["Year"],
             df_japan["Total"],
             linestyle='dashed',
             label="Japan")
    plt.plot(df_Germany["Year"],
             df_Germany["Total"],
             label="Germany")
    plt.plot(df_UK["Year"],
             df_UK["Total"],
             linestyle='dashed',
             label="UK")

    # labeling
    plt.xlabel("Year", labelpad=(10), fontweight="bold")
    plt.ylabel("Electricity production(% of total)",
               labelpad=(10), fontweight="bold")

    # add a title and legend
    plt.title("Total Electricity production from natural gas sources by country ",
              fontweight="bold", y=1.1)
    plt.legend(loc='center left',
               bbox_to_anchor=(1, 0.5),
               fancybox=True,
               shadow=True)

    plt.xticks(rotation=90)

    # save the plot as png
    plt.savefig("natural_gas_line_chart.png")

    # show the plot
    plt.show()

    # end the function
    return


def plot_heat_map(country_name):
    """ This ia a function to create a heatmap for country specific indicators.
    This function takes country as an argument, and use plot correlation
    between indicators"""

    # extract the given country data
    df_data = extract_country_data(country_name)

    # create correlation matrix
    corr_matrix = df_data.corr()

    # plot heatmap
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)

    # rotate the x-axis labels by 45 degrees
    plt.xticks(rotation=45)

    # set the plot title
    plt.title('Correlation between Indicators in ' +
              country_name, fontweight="bold", y=1.05)
    plt.xlabel("")
    plt.ylabel("")

    # save the plot as png
    plt.savefig("heat_map.png")

    # show the plot
    plt.show()

    # end the function
    return


def plot_heat_map2(country_name):
    """ This ia a function to create a heatmap for country specific indicators.
    This function takes country as an argument, and use plot correlation
    between indicators"""

    # extract the given country data
    df_data = extract_country_data(country_name)

    # create correlation matrix
    corr_matrix = df_data.corr()

    # plot heatmap
    sns.heatmap(corr_matrix, cmap='YlGnBu', annot=True)

    # rotate the x-axis labels by 45 degrees
    plt.xticks(rotation=90)
    plt.yticks()

    # set the plot title
    plt.title('Correlation between Indicators in ' + country_name,
              fontweight="bold",
              y=1.05)
    plt.xlabel("")
    plt.ylabel("")

    # save the plot as png
    plt.savefig("heat_map2.png")

    # show the plot
    plt.show()

    # end the function
    return


# =============================================================================
# This section is the main program of this code. In here all the pre
# processing requirements, statistical comparisons and calling functions done
# =============================================================================


# call the function for read file and generate 2 dataframes
df_year, df_country = read_electricity_production_data("Climate.csv")

# call the function to extract stat properties of each indicator per state
individual_country_statisctic("Netherlands")
individual_country_statisctic("Russian Federation")
individual_country_statisctic("Germany")

# call the function to extract stat properties of each country per indicator
individual_indicator_statistics("Renewable")
individual_indicator_statistics("Oil Sources")
individual_indicator_statistics("Nuclear Sources")

# extrcact useful countries
df_Netherlands = extract_country_data("Netherlands")
df_Russia = extract_country_data("Russian Federation")
df_Germany = extract_country_data("Germany")

# assign the columns into new dataframe
df_bcols = df_Netherlands.columns
df_gcols = df_Russia.columns
df_ucols = df_Germany.columns

# find the skewness, round it to 2 decimals and put it into dictionary
Netherlands_skew = df_Netherlands.apply(skew).round(2).to_dict()
Russia_skew = df_Russia.apply(skew).round(2).to_dict()
Germany_skew = df_Germany.apply(skew).round(2).to_dict()

# find the kutosis, round it to 2 decimals and put it into dictionary
Netherlands_kurtosis = df_Netherlands.apply(kurtosis).round(2).to_dict()
Russia_kurtosis = df_Russia.apply(kurtosis).round(2).to_dict()
Germany_kurtosis = df_Germany.apply(kurtosis).round(2).to_dict()

# ignore warning
warnings.filterwarnings("ignore", message="Precision loss occurred in moment\
                        calculation due to catastrophic cancellation")

# create dictionary to store summary statistics
stats = {
    ("Variance", "Netherlands"): {
        c: round(np.var(df_Netherlands[c]), 2) for c in df_bcols
    },
    ("Variance", "Russian Federation"): {
        c: round(np.var(df_Russia[c]), 2) for c in df_gcols
    },
    ("Variance", "Germany"): {
        c: round(np.var(df_Germany[c]), 2) for c in df_ucols
    },
    ("Skewness", "Netherlands"): Netherlands_skew,
    ("Skewness", "Russian Federation"): Russia_skew,
    ("Skewness", "Germany"): Germany_skew,
    ("Kutosis", "Netherlands"): Netherlands_kurtosis,
    ("Kutosis", "Russian Federation"): Russia_kurtosis,
    ("Kutosis", "Germany"): Germany_kurtosis
}

# assign statistics into a dataframe
df_statistics = pd.DataFrame(stats)

# print the summary statistics
print(df_statistics)

# create a df with usufull countries
df_countries = ["Netherlands", "Finland",
                "Russian Federation", "Spain", "Germany"]

# create a loop to iterate over countries df
for c in df_countries:

    # extract the country data
    df_country1 = extract_country_data(c)

    # calculate the correlation
    df_corr = df_country1.corr()

    # print all correlation matrices
    print("Correlation matrix for indicators in",
          c, ":", "\n", "\n", df_corr, "\n")

# transform the df_year dataframe seperate years columns into one year column
df_year_new = pd.melt(df_year,
                      id_vars=["Country Name",
                               "Country Code",
                               "Indicator Name"
                               ],
                      value_vars=df_year.iloc[:, 3:-1].columns,
                      var_name="Year",
                      value_name=("Total"))

# explore the new dataframe
print(df_year_new.head())

# create a list of countrues for further analysis
countries = ["Switzerland", "Netherlands", "Finland",
             "Russian Federation", "Spain", "Japan", "United Kingdom", "Germany"]

# crete new dataframe with reuqired country data
df_countries = df_year_new.groupby(
    'Country Name').filter(lambda x: x.name in countries)

# extract data for co2 emission
df_countries_oil_sources = df_countries[df_countries["Indicator Name"]
                                        == "Oil Sources"]

# explore new dataframe
print(df_countries_oil_sources.head())

# extract data for population growth
df_countries_Renewable = df_countries[
    df_countries["Indicator Name"] == "Renewable"
]

# change some country names into aabbreviations
df_countries_Renewable.loc[df_countries_Renewable["Country Name"]
                           == "Germany", "Country Name"] = "Germany"
df_countries_Renewable.loc[df_countries_Renewable["Country Name"]
                           == "United Kingdom", "Country Name"] = "UK"


# explore new dataframe
print(df_countries_Renewable.head())

# extrace data for renew. energy comsump.
df_countries_renew_energy = df_countries[
    df_countries['Indicator Name'] == "Renew. energy consump(%)"
].sort_values(by="Country Name", ascending=True)

# explore new dataframe
print(df_countries_renew_energy.head())

# extrace data for oil_sources
df_countries_oil_sources = df_countries[df_countries["Indicator Name"]
                                        == "Oil Sources"]

# explore new dataframe
print(df_countries_oil_sources.head())

# extrace data for aggricultural land
df_countries_natural_gas_sources = df_countries[df_countries["Indicator Name"] ==
                                   "Natural Gas Sources"]

# explore new dataframe
print(df_countries_natural_gas_sources)

# call function to create oil source multiple line chart
plt_oil_sources_line_chart(df_countries_oil_sources)

# call function to create population growth boxplots
plot_renewable_electricity_line_chart(df_countries_Renewable)

# call function to create natural gas sources multiple line charts
plot_natural_gas_line_chart(df_countries_natural_gas_sources)

# call the function to create correlation heatmap for Finland and Germany
plot_heat_map("Finland")
plot_heat_map2("Germany")
