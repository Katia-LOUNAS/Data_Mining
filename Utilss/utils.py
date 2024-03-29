import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from pandas.api.types import is_numeric_dtype
import itertools

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
    if modes == sorted_data.unique().tolist():
        modes = "Aucun mode"
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

def missing_values_info(data):
    missing_values_count = data.isnull().sum()
    missing_percentage = (missing_values_count / len(data)) * 100
    
    missing_info_df = pd.DataFrame({
        'Column': missing_values_count.index,
        'Missing_Values_Count': missing_values_count.values,
        'Missing_Values_Percentage': missing_percentage.values
    })

    return missing_info_df

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
    plt.show()

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

def bins_huntsberger(data):
    return int(1 + (10 / 3) * math.log10(data.shape[0]))

def equal_width_disc(data, column_name, label):
    num_bins = bins_huntsberger(data)
    # Check if the column is not already numeric
    if data[column_name].dtype != 'float64':
        # Handle commas as decimal separators and convert to numeric
        data[column_name] = pd.to_numeric(data[column_name].astype(str).str.replace(',', '.'), errors='coerce')
        

    # Copy the original DataFrame to avoid modifying the original data
    discretized_data = data.copy()

    # Calculate the width of each bin
    bin_width = (data[column_name].max() - data[column_name].min()) / num_bins

    # Define bin edges and round to a precision
    bin_edges = [round(data[column_name].min() + i * bin_width, 5) for i in range(num_bins + 1)]

    # Define bin labels
    if label == 1:
        bin_labels = [f'{i + 1}' for i in range(num_bins)]
    else:
        bin_labels = [data[column_name].loc[(data[column_name] >= bin_edges[i]) & (data[column_name] < bin_edges[i + 1])].mean() for i in range(num_bins)]

    # Perform discretization using cut function
    discretized_data['Discretized'] = pd.cut(data[column_name], bins=bin_edges, labels=bin_labels, include_lowest=True)

    return discretized_data

def equal_frequency_discretization(data, column_name, label):
    num_bins = int(1 + 3.322 * math.log10(data.shape[0]))

    # Check if the column is already numeric
    if data[column_name].dtype != 'float64':
        # Handle commas as decimal separators and convert to numeric
        data[column_name] = pd.to_numeric(data[column_name].str.replace(',', '.'), errors='coerce')


    # Copy the original DataFrame to avoid modifying the original data
    discretized_data = data.copy()

    # Calculate the number of data points per bin
    points_per_bin = data.shape[0] // num_bins

    # Sort the data
    sorted_data = data[column_name].sort_values()

    # Define bin edges
    bin_edges = [sorted_data.iloc[i * points_per_bin] for i in range(num_bins)]
    bin_edges.append(sorted_data.max())  # Add the maximum value as the last edge

     # Define bin labels
    if (label == 1):
        bin_labels = [f'{i + 1}' for i in range(num_bins)]
    else:
        bin_labels = [data[column_name].loc[(data[column_name] >= bin_edges[i]) & (data[column_name] < bin_edges[i + 1])].mean() for i in range(num_bins)]

    # Perform discretization using cut function
    discretized_data['Discretized'] = pd.cut(data[column_name], bins=bin_edges, labels=bin_labels, include_lowest=True)

    return discretized_data

def plot_before_after_discretization(data, column_name, discretized_data, discretized_data2):
    plt.figure(figsize=(18, 6))

    # Plot before discretization
    plt.subplot(1, 3, 1)
    plt.hist(data[column_name], bins=20, color='blue', alpha=0.7)
    plt.title(f'Before Discretization - {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')

    # Plot after discretization (equal_width)
    plt.subplot(1, 3, 2)
    plt.hist(discretized_data['Discretized'], bins=20, color='green', alpha=0.7)
    plt.title(f'After Discretization (equal_width) - {column_name}')
    plt.xlabel('Discretized')
    plt.ylabel('Frequency')

    # Plot after discretization (equal_frequency)
    plt.subplot(1, 3, 3)
    plt.hist(discretized_data2['Discretized'], bins=20, color='orange', alpha=0.7)
    plt.title(f'After Discretization (equal_frequency) - {column_name}')
    plt.xlabel('Discretized')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def calculate_lift(support_A, support_B, support_AB, num_transactions):
    return (num_transactions * support_AB) / (support_A * support_B)

def calculate_correlation(support_A, support_B, support_AB, num_transactions):
    if support_A == 0 or support_B == 0 or support_A == num_transactions or support_B == num_transactions:
        return 0.0  # Return 0 if any of the denominators is zero or equal to the number of transactions
    correlation = (support_AB * num_transactions - support_A * support_B) / (
        (num_transactions - support_A) * (num_transactions - support_B) * support_A * support_B) ** 0.5
    return correlation

def calculate_cosine_similarity(support_A, support_B, support_AB):
    return support_AB / ((support_A * support_B) ** 0.5)

def calculate_support(itemset, transactions):
    count = 0
    for transaction in transactions:
        if itemset.issubset(transaction):
            count += 1
    return count
    
def generate_association_rules(L, min_confidence, transactions):
    association_rules = []    
    for itemset in L:
        itemset_list = list(itemset)
        for i in range(1, len(itemset_list)):
            for combination in itertools.combinations(itemset_list, i):
                A = set(combination)
                B = itemset - A
                support_A = calculate_support(A, transactions)
                support_B = calculate_support(B, transactions)
                support_AB = calculate_support(itemset, transactions)
                confidence = support_AB / support_A
                if confidence >= min_confidence:
                    lift = (support_AB * len(transactions)) / (support_A * support_B)
                    correlation = calculate_correlation(support_A, support_B, support_AB, len(transactions))
                    cosine_similarity = calculate_cosine_similarity(support_A, support_B, support_AB)
                    rule = (A, B, confidence, lift, correlation, cosine_similarity)
                    association_rules.append(rule)
    return association_rules

def outliers_mean(df,column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    if not outliers.empty:
        df[column][outliers.index] = df[column].mean()

def outliers_drop(df, columns):
    filtered_df = df.copy()  
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_df = filtered_df[(filtered_df[column] >= lower_bound) & (filtered_df[column] <= upper_bound)]
    return filtered_df

def moyenne(df,att):
    return (sum(df[att]) / len(df[att]))

def med(df,att):
    col_sorted = sorted(df[att])
    if len(col_sorted) % 2 != 0:
        return col_sorted[int(len(col_sorted)//2)]
    else:
        s = len(col_sorted)//2
        return ((col_sorted[s]+col_sorted[s+1])/2)

def confusion_matrix(y_true, y_pred, num_classes=3):
    matrix = pd.DataFrame(0, index=range(num_classes), columns=range(num_classes))

    for true_label, pred_label in zip(y_true, y_pred):
        matrix.loc[true_label, pred_label] += 1

    return matrix

def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt=".0f", cmap="Blues", cbar=False,
                xticklabels=np.arange(0, 3), yticklabels=np.arange(0, 3))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def parameters(cm):
    TP = cm.values.diagonal()
    FP = cm.sum(axis=1) - TP
    FN = cm.sum(axis=0) - TP
    TN = cm.values.sum() - (TP + FP + FN)

    return {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}

def average(data,len=3):
    moy=0
    for i in data:
        moy+=i
    return moy/len

def class_accuracy(y_true, y_pred, class_label):
    # Convert y_true to a NumPy array
    y_true = np.array(y_true)
    
    # Filter indices where true label matches the specified class_label
    class_indices = np.where(y_true == class_label)[0]
    
    # Extract predictions and true labels for the specified class
    class_true_labels = y_true[class_indices]
    class_pred_labels = y_pred[class_indices]

    # Calculate accuracy for the specified class
    correct_predictions = sum(1 for true_label, pred_label in zip(class_true_labels, class_pred_labels) if true_label == pred_label)
    total_predictions = len(class_true_labels)
    
    # Handle the case where there are no predictions for the specified class
    if total_predictions == 0:
        return 0.0
    
    accuracy = correct_predictions / total_predictions
    return accuracy

def accuracy(y_true, y_pred):
    correct_predictions = sum(1 for true_label, pred_label in zip(y_true, y_pred) if true_label == pred_label)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy

def specificity_class(cm, class_label):
    p = parameters(cm)
    TN = p['TN'][class_label]
    FP = p['FP'][class_label]
    return TN / (TN + FP) if (TN + FP) > 0 else 0

def specificity(cm,classes = 3):
    moy = 0
    for my_class in range(classes):
        moy+=specificity_class(cm,my_class)
    return moy/classes

def precision_class(cm, class_label):
    TP = cm.loc[class_label, class_label]
    FP = cm.loc[:, class_label].sum() - TP
    return TP / (TP + FP) if (TP + FP) > 0 else 0

def precision(cm,classes = 3):
    moy=0
    for my_class in range(classes):
        moy+=precision_class(cm,my_class)
    return moy/classes

def recall_class(cm, class_label):
    TP = cm.loc[class_label, class_label]
    FN = cm.loc[class_label, :].sum() - TP
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def recall(cm,classes=3):
    moy=0
    for my_class in range(classes):
        moy+=recall_class(cm,my_class)
    return moy/classes

def f_score_class(cm, class_label):
    prec = precision_class(cm, class_label)
    rec = recall_class(cm, class_label)

    f1_score = 2 * (prec * rec) / (prec + rec) if np.any(prec + rec != 0) else 0
    return f1_score
def f_score(cm,classes=3):
    moy=0
    for my_class in range(classes):
        moy+=f_score_class(cm,my_class)
    return moy/classes

def silhouette_score_scratch(X, labels):
    n_samples = len(X)
    cluster_labels = np.unique(labels)
    num_clusters = len(cluster_labels)
    
    if num_clusters == 1:
        return 0  
    a_values = np.zeros(n_samples)
    b_values = np.zeros(n_samples)

    for i in range(n_samples):
        cluster_i = labels[i]
        a_values[i] = np.mean([np.linalg.norm(X[i] - X[j]) for j in range(n_samples) if labels[j] == cluster_i and j != i])

        b_values[i] = min(
            np.mean([np.linalg.norm(X[i] - X[j]) for j in range(n_samples) if labels[j] == cluster_j]) 
            for cluster_j in cluster_labels if cluster_j != cluster_i
        )

    silhouette_values = (b_values - a_values) / np.maximum(a_values, b_values)
    
    return np.mean(silhouette_values)

def silhouette_sample_scratch(X, labels):
    n_samples = len(X)
    cluster_labels = np.unique(labels)
    num_clusters = len(cluster_labels)
    
    if num_clusters == 1:
        return np.zeros(n_samples)  # All points in the same cluster, silhouette score is 0 for each.

    silhouette_values = np.zeros(n_samples)

    for i in range(n_samples):
        cluster_i = labels[i]
        a_i = np.mean([np.linalg.norm(X[i] - X[j]) for j in range(n_samples) if labels[j] == cluster_i and j != i])

        b_i_values = [np.mean([np.linalg.norm(X[i] - X[j]) for j in range(n_samples) if labels[j] == cluster_j]) 
                      for cluster_j in cluster_labels if cluster_j != cluster_i]
        b_i = min(b_i_values) if b_i_values else 0

        silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)

    return silhouette_values