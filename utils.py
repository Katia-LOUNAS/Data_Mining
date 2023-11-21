import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_data(path):
    data = pd.read_csv(path)
    return(data)

def print_info(data):
    dataset_description = {}
    dataset_description["Number of rows"] = data.shape[0]
    dataset_description["Number of colonns"] = data.shape[1]
    dataset_description["Memory size"] = str(data.memory_usage(index=False).sum() / 1024) + " ko"
    dataset_description["Data type"] = list(map(str, data.dtypes.unique().tolist()))
    return dataset_description

def describe_column(data):
    colonnes_description = []
    for d in data:
        colonnes_description.append([d, data[d].count(), str(data.dtypes[d])])
    return colonnes_description
    
    
def central_trend(data, column_name):
    # Filter out non-numeric values
    numeric_values = pd.to_numeric(data[column_name], errors='coerce').dropna()
    
    # Sorting the column values
    sorted_data = numeric_values.sort_values(ascending=True)
    
    # Calculate the median
    if len(sorted_data) % 2 == 0:
        median = (sorted_data.iloc[len(sorted_data) // 2 - 1] + sorted_data.iloc[len(sorted_data) // 2]) / 2
    else:
        median = sorted_data.iloc[len(sorted_data) // 2]
    
    # Calculate the mean
    mean = np.mean(sorted_data)
    
    # Calculate mode
    frequency = {}
    for value in sorted_data:
        if value in frequency:
            frequency[value] += 1
        else:
            frequency[value] = 1
    
    # Find the value(s) with the highest frequency
    modes = [value for value, freq in frequency.items() if freq == max(frequency.values())]

    # Return a dictionary with central tendency measures
    central_tendency = {
        'column_name': column_name,
        'median': median,
        'mean': mean,
        'modes': modes
    }

    return central_tendency

def symetrie(ct):
    mode = max(ct["Modes"]) if type(ct["Modes"]) == list else ct["Modes"]

    if round(ct["Mean"]) == round(ct["Median"]) == round(mode):
        return "Distribution symetrique"
    elif ct["Mean"] < ct["Median"] < mode:
        return "Distribution d'asymetrie negative"
    elif ct["Mean"] > ct["Median"] > mode:
        return "Distribution d'asymetrie positive"
    else:
        return "Distribution non identifie"

def quartile(column_data):
    # Sorting the column values
    sorted_data = column_data.sort_values(ascending=True)
    # List to store quartile values
    quartile_list = []
    # Calculate Q0 (min)
    Q0 = sorted_data.iloc[0]
    quartile_list.append(('Q0', Q0))
    # Calculate Q1
    n = len(sorted_data)
    if n % 4 == 0:
        Q1 = (sorted_data.iloc[n // 4 - 1] + sorted_data.iloc[n // 4]) / 2
    else:
        Q1 = sorted_data.iloc[n // 4]
    quartile_list.append(('Q1', Q1))
    # Calculate Q2 (median)
    if n % 2 == 0:
        Q2 = (sorted_data.iloc[n // 2 - 1] + sorted_data.iloc[n // 2]) / 2
    else:
        Q2 = sorted_data.iloc[n // 2]
    quartile_list.append(('Q2', Q2))
    # Calculate Q3
    if n % 4 == 0:
        Q3 = (sorted_data.iloc[3 * (n // 4) - 1] + sorted_data.iloc[3 * (n // 4)]) / 2
    else:
        Q3 = sorted_data.iloc[3 * (n // 4)]
    quartile_list.append(('Q3', Q3))
    # Calculate Q4 (max)
    Q4 = sorted_data.iloc[n - 1]
    quartile_list.append(('Q4', Q4))
    # Return the results as a list of tuples
    return quartile_list        

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
    
def histogramme_sns(data, attribut, ax):
    sns.histplot(data[attribut], bins=10, color='b', edgecolor='k', ax=ax)
    ax.set(xlabel="X", ylabel="Y", title=f'Histogramme {attribut}')
    
def box_plot(attribut, title):
    plt.boxplot(attribut)
    plt.title(title)
    plt.show
    
from pandas.api.types import is_numeric_dtype

def box_plot_sns(data, attribute, ax):
    if is_numeric_dtype(data[attribute]):
        sns.boxplot(x=data[attribute], ax=ax)
        ax.set_title(f'Box Plot of {attribute}')












        
        