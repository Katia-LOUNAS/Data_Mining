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
    return nb_valeurs_manquantes, missing_values_dict     #result_dict['Column']['Percentage of missing values']

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

def find_outliers_iqr(column_data):
    # Get quartiles
    quartiles = quartile(column_data)

    # Calculate the IQR 
    iqr = quartiles[3][1] - quartiles[1][1]

    # Lower and upper bounds 
    lower_bound = quartiles[1][1] - 1.5 * iqr
    upper_bound = quartiles[3][1] + 1.5 * iqr

    # Find the indices of outliers
    outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)].index.tolist()

    return outliers

def find_outliers_zscore(column_data, threshold=3):
    # Calculate the Z-Score for each observation
    z_scores = (column_data - column_data.mean()) / column_data.std()

    # Find the indices of outliers based on the Z-Score
    outliers = z_scores[abs(z_scores) > threshold].index.tolist()

    return outliers

def winsorize(column_data):

    winsorized_column = column_data.copy()
    quartiles = quartile(winsorized_column)

    # Calculate the winsorization limits
    lower_limit = quartiles[1][1] - 1.5 * (quartiles[3][1] - quartiles[1][1])
    upper_limit = quartiles[3][1] + 1.5 * (quartiles[3][1] - quartiles[1][1])

    # Winsorize values below the lower limit
    winsorized_column[winsorized_column < lower_limit] = lower_limit

    # Winsorize values above the upper limit
    winsorized_column[winsorized_column > upper_limit] = upper_limit

    return winsorized_column

def min_max_normalize(data):
    normalized_data = data.copy()

    for column in normalized_data.columns:
        # Calculate min and max values for the column
        min_value = normalized_data[column].min()
        max_value = normalized_data[column].max()

        # Normalize the column using Min-Max scaling
        normalized_data[column] = (normalized_data[column] - min_value) / (max_value - min_value)

    return normalized_data

def z_score_normalize(data):
    normalized_data = data.copy()

    for column in normalized_data.columns:
        # Calculate mean and standard deviation for the column
        ct_result = central_trend(data, column)
        mean_value = ct_result['mean']
        std_dev = normalized_data[column].std()

        # Normalize the column using Z-score scaling
        normalized_data[column] = (normalized_data[column] - mean_value) / std_dev

    return normalized_data





        
        