import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


def load_data(path):
    data = pd.read_csv(path)
    return(data)

def print_info(data):
    print("The data is composed of " + str(data.shape[0]) + " rows and " + str(data.shape[1]) + " columns")
    print("The attribute list for this dataset is :", data.columns)
    
    
def central_trend(data, column_name):
    # Sorting the column values
    sorted_data = data[column_name].sort_values(ascending=True)
    
    # Calculate the median
    if len(sorted_data) % 2 == 0:
        median = (sorted_data.iloc[len(sorted_data) // 2 - 1] + sorted_data.iloc[len(sorted_data) // 2]) / 2
    else:
        median = sorted_data.iloc[len(sorted_data) // 2]
    
    print(column_name + ':')
    print("Median:", median)
    
    # Calculate the mean
    mean = np.mean(sorted_data)
    print("Mean:", mean)
    
    # Calculate mode
    frequency = {}
    for value in sorted_data:
        if value in frequency:
            frequency[value] += 1
        else:
            frequency[value] = 1
    
    # Find the value(s) with the highest frequency
    modes = [value for value, freq in frequency.items() if freq == max(frequency.values())]
    print("Mode(s) values:", modes)

def quartile(column_data):
    # Sorting the column values
    sorted_data = column_data.sort_values(ascending=True)
    # Dictionary to store quartile values
    quartile_dict = {}
    # Calculate Q0 (min)
    Q0 = sorted_data.iloc[0]
    quartile_dict['Q0'] = Q0
    # Calculate Q1
    n = len(sorted_data)
    if n % 4 == 0:
        Q1 = (sorted_data.iloc[n // 4 - 1] + sorted_data.iloc[n // 4]) / 2
    else:
        Q1 = sorted_data.iloc[n // 4]
    quartile_dict['Q1'] = Q1
    # Calculate Q2 (median)
    if n % 2 == 0:
        Q2 = (sorted_data.iloc[n // 2 - 1] + sorted_data.iloc[n // 2]) / 2
    else:
        Q2 = sorted_data.iloc[n // 2]
    quartile_dict['Q2'] = Q2
    # Calculate Q3
    if n % 4 == 0:
        Q3 = (sorted_data.iloc[3 * (n // 4) - 1] + sorted_data.iloc[3 * (n // 4)]) / 2
    else:
        Q3 = sorted_data.iloc[3 * (n // 4)]
    quartile_dict['Q3'] = Q3
    # Calculate Q4 (max)
    Q4 = sorted_data.iloc[n - 1]
    quartile_dict['Q4'] = Q4
    # Print the results
    print(f"Quartile values for the column:\n{quartile_dict}")
    return quartile_dict         #print(result_dict['Column']['Q1'])  to access to the quaetile

def missing_value(column_data):
    # Dictionary to store missing values and their percentages for the column
    missing_values_dict = {}
    # get the data shape
    n = len(column_data)
    # Calculate the number of missing values
    nb_valeurs_manquantes = column_data.isnull().sum()
    # Calculate the percentage of missing values
    pourcentage_valeurs_manquantes = (nb_valeurs_manquantes / n) * 100
    # Store the information in the dictionary
    missing_values_dict['Number of missing values'] = nb_valeurs_manquantes
    missing_values_dict['Percentage of missing values'] = pourcentage_valeurs_manquantes
    # Print the information
    print(f"Column: {column_data.name}\nNumber of missing values: {nb_valeurs_manquantes}\nPercentage of missing values: {pourcentage_valeurs_manquantes:.2f}%\n")
    return missing_values_dict     #result_dict['Column']['Percentage of missing values']

def Scatter_plot(data, att1 , att2):

    plt.scatter(data[att1],data[att2], marker='o')
    plt.title("Scatter plot")
    plt.xlabel(att1)
    plt.ylabel(att2)
    
def histogramme(data , attribut):
    plt.hist(data[attribut], bins=10, color='b', edgecolor='k')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Histogramme")
    
def box_plot(attribut):
    plt.boxplot(attribut)
    plt.title("Box")
    plt.show












        
        