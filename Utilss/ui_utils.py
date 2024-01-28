import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import utils 
from io import BytesIO
from PIL import Image
import tempfile
import utils.appriori as appriori


global custom_colors 
custom_colors =  ['#0d9488', '#344B48','#B87436']

def opendatafile(file,pre):
    if file == "Dataset 1":
        if pre == "before preprocessing":
            file = "./Data/Dataset11.csv"
        else:
            file = "./Data/Dataset1_final.csv"
    elif file == "Dataset 2":
        if pre == "before preprocessing":
            file = "./Data/Dataset2.csv"
        else:
            file = "./Data/Dataset2_final.csv"
    elif file == "Dataset 3":
        if pre == "before preprocessing":
            file = "./Data/Dataset3.csv"
        elif pre == "after preprocessing":
            file = "./Data/Dataset3_final.csv"
        
    data = pd.read_csv(file)
    return data 

def describe_column(file,pre):
    data = opendatafile(file,pre)
    colonnes_description = []
    for d in data:
        colonnes_description.append([d, data[d].count(), str(data.dtypes[d])])
    df = pd.DataFrame(colonnes_description, columns = ["Name","Non-null value","Type"])
    return df


import pandas as pd

def print_info(file,pre):
    data = opendatafile(file,pre)
    dataset_description = pd.DataFrame(columns=["Property", "Value"])

    dataset_description.loc[len(dataset_description)] = ["Number of rows", data.shape[0]]
    dataset_description.loc[len(dataset_description)] = ["Number of columns", data.shape[1]]
    dataset_description.loc[len(dataset_description)] = ["Memory size", f"{data.memory_usage(index=False).sum() / 1024} ko"]
    dataset_description.loc[len(dataset_description)] = ["Data types", ', '.join(map(str, data.dtypes.unique().tolist()))]

    missing_values_count = data.isnull().sum()
    missing_percentage = (missing_values_count / len(data)) * 100
    overall_missing_percentage = (missing_values_count.sum() / (len(data) * data.shape[1])) * 100

    dataset_description.loc[len(dataset_description)] = ["Missing Values percentage", str(f"{overall_missing_percentage:.4f} %")]

    return dataset_description

def show_cent_trends(file,pre):
    data = opendatafile(file,pre)
    attributes = data.select_dtypes(include='number').columns
    if file == "Dataset 2":
        attributes = attributes.drop(['zcta'])
    central_trend_dict = {}

    for attribute in attributes:
        ct_result = utils.central_trend(data, attribute)
        tc = {
            'Median': ct_result['median'],
            'Mean': ct_result['mean'],
            'Modes': ct_result['modes'],
        }
        tc['symetrie'] = utils.symetrie(tc)
        # central_trend_dict[attribute] = tc
        central_trend_dict[attribute] = {**{'Attribute': attribute}, **tc}
    df = pd.DataFrame.from_dict(central_trend_dict, orient='index')
    return df



def dataset_description(file,pre):
    if file == "Dataset 2":
        html = """<div style="background-color: #135461; color: white; padding: 15px; border-radius: 10px; text-align: justify;">
                  <h2 style="text-align: center; font-family: 'Arial', sans-serif; font-size: 24px; font-weight: bold; color: #cbfbf3;">Coded Chronicles of COVID-19</h2>
                  <p>Dataset 2 serves as an essential resource for understanding the temporal evolution of the number of COVID-19 cases, organized by postal code. Each entry in the dataset provides key information such as postal code, time period, local population, start and end dates, total case count, test count, positive tests, and crucial indicators like case rate, test rate, and positivity rate. These data allow for in-depth analysis of the virus's spread, offering researchers and public health officials valuable insights to assess epidemiological trends, make informed decisions, and implement effective strategies to contain the spread of COVID-19 at the local level.</p>
                  </div>
                """
    elif file == "Dataset 1":
        html = """<div style="background-color: #135461; color: white; padding: 15px; border-radius: 10px; text-align: justify;">
                <h2 style="text-align: center; font-family: 'Arial', sans-serif; font-size: 24px; font-weight: bold; color: #cbfbf3;">Comprehensive Soil Fertility Analysis</h2>
                <p>Dataset 1 represents a crucial resource for evaluating soil fertility through static data encompassing soil properties. Each entry in the dataset provides key information on soil characteristics, including nutrient levels such as nitrogen (N), phosphorus (P), potassium (K), soil pH, electrical conductivity (EC), organic carbon (OC), sulfur (S), zinc (Zn), iron (Fe), copper (Cu), manganese (Mn), boron (B), organic matter (OM), and soil fertility. This dataset offers a detailed view of the chemical, physical, and biological properties of the soil, enabling researchers and agronomy experts to analyze soil composition and assess its capacity to support crop growth. This static dataset provides a solid foundation for in-depth analyses aimed at improving agricultural practices, maximizing yields, and promoting sustainable land use.</p>
                </div>
                """
    else:
        html = """<div style="background-color: #135461; color: white; padding: 15px; border-radius: 10px; text-align: justify;">
                <h2 style="text-align: center; font-family: 'Arial', sans-serif; font-size: 24px; font-weight: bold; color: #cbfbf3;">Exploring Relationships in Environmental and Agricultural Data</h2>
                <p>Our objective is to analyze and extract frequent patterns, association rules, and correlations from Dataset 3. We aim to illuminate the existing relationships among climate attributes (Temperature, Humidity, Precipitation), soil characteristics, vegetation, and fertilizer usage. This analysis will provide essential information to make informed decisions in the context of environmental and agricultural resource management. The chosen attribute for discretization is rainfall. The dataset includes variables such as Temperature, Humidity, Rainfall, Soil, Crop, and Fertilizer, which will contribute to a comprehensive understanding of the interconnections influencing environmental and agricultural dynamics.</p>
                </div>
                """ 
    return html

def graphe_2_1(file,pre):
    data = opendatafile(file,pre)
    data_grouped = data.groupby('zcta')[['case count', 'positive tests']].sum()
    # custom_colors = ['#0d9488', '#344B48','#B87436']
    fig, ax = plt.subplots(figsize=(12, 6))
    data_grouped.plot(kind='bar', ax = ax, color=custom_colors)
    plt.title('Distribution of the total number of confirmed cases and positive tests by ZIPCODE')
    plt.xlabel('ZIPCODE')
    plt.legend(["confirmed cases", "positive tests"])
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        plt.savefig(temp_file, format='png')
        temp_file_path = temp_file.name
    plt.close(fig)
    return temp_file_path
def graphe_2_2(file,pre,zipcode,period):
    # custom_colors = ['#0d9488', '#344B48','#B87436']
    data = opendatafile(file,pre)
    zone_data = data[data['zcta'] == int(zipcode)]
    zone_data['Start date'] = pd.to_datetime(zone_data['Start date'])
    if period == "WEEKLY":
        zone_data_weekly = zone_data.set_index('Start date').resample('W').sum()
        print(zone_data_weekly)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(zone_data_weekly.index, zone_data_weekly['test count'], label='test count', color=custom_colors[0])
        ax.plot(zone_data_weekly.index, zone_data_weekly['positive tests'], label='positive tests', color=custom_colors[1])
        ax.plot(zone_data_weekly.index, zone_data_weekly['case count'], label='confirmed cases', color=custom_colors[2])
        ax.set_title(f'Weekly evolution of COVID-19 tests, positive tests, and confirmed cases for the area {zipcode}')
        ax.set_xlabel('Periode')
        ax.set_ylabel('Numbre')
        ax.legend()
        ax.grid(True)
    elif period == "MONTHLY":
        zone_data_monthly = zone_data.set_index('Start date').resample('M').sum()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(zone_data_monthly.index, zone_data_monthly['test count'], label='test count', color=custom_colors[0])
        ax.plot(zone_data_monthly.index, zone_data_monthly['positive tests'], label='positive tests', color=custom_colors[1])
        ax.plot(zone_data_monthly.index, zone_data_monthly['case count'], label='confirmed cases', color=custom_colors[2])
        ax.set_title(f'Monthly evolution of COVID-19 tests, positive tests, and confirmed cases for the area {zipcode}')
        ax.set_xlabel('Periode')
        ax.set_ylabel('Numbre')
        ax.legend()
        ax.grid(True)
    else:
        zone_data_yearly = zone_data.set_index('Start date').resample('Y').sum()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(zone_data_yearly.index, zone_data_yearly['test count'], label='test count', color=custom_colors[0])
        ax.plot(zone_data_yearly.index, zone_data_yearly['positive tests'], label='positive tests', color=custom_colors[1])
        ax.plot(zone_data_yearly.index, zone_data_yearly['case count'], label='confirmed cases', color=custom_colors[2])
        ax.set_title(f'Yearly evolution of COVID-19 tests, positive tests, and confirmed cases for the area {zipcode}')
        ax.set_xlabel('Periode')
        ax.set_ylabel('Numbre')
        ax.legend()
        ax.grid(True)
        # Save the plot to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        plt.savefig(temp_file, format='png')
        temp_file_path = temp_file.name
        plt.close(fig)
    return temp_file_path

def graphe_2_3(data,pre):
    data = opendatafile(data,pre)
    data['Start date'] = pd.to_datetime(data['Start date'])
    data['year'] = data['Start date'].dt.year
    grouped_data = data.groupby(['zcta', 'year'])['case count'].sum().unstack()
    fig, ax = plt.subplots(figsize=(12, 6))
    grouped_data.plot(kind='bar', ax = ax, color=custom_colors)
    plt.title('Distribution of positive COVID-19 cases by area and by year')
    plt.xlabel('ZIPCODE')
    plt.legend(["2019", "2020", "2021"])
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        plt.savefig(temp_file, format='png')
        temp_file_path = temp_file.name
        plt.close(fig)
    return temp_file_path

def graphe_2_4(data,pre):
    data = opendatafile(data,pre)
    grouped_data = data.groupby(['population'])['test count'].sum()
    fig, ax = plt.subplots(figsize=(12, 6))
    grouped_data.plot(kind='line', ax = ax, color=custom_colors[0])
    plt.title('Relation between the population and the number of tests conducted')
    plt.xlabel('Population')
    plt.ylabel('Number of tests')
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        plt.savefig(temp_file, format='png')
        temp_file_path = temp_file.name
        plt.close(fig)
    return temp_file_path


def graphe_2_5(data,pre):
    data = opendatafile(data,pre)
    grouped_data = data.groupby('zcta')['positive tests'].sum()
    top_5_zones = grouped_data.sort_values(ascending=False).head(5)
    fig, ax = plt.subplots(figsize=(12, 6))
    top_5_zones.plot(kind='bar', ax = ax, color=custom_colors[0])
    plt.title('Top 5 most affected areas by COVID-19')
    plt.xlabel('ZIPCODE')
    plt.ylabel('Number of positive tests')
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        plt.savefig(temp_file, format='png')
        temp_file_path = temp_file.name
        plt.close(fig)
    return temp_file_path
def graphe_2_6(data,pre,period):
    data = opendatafile(data,pre)
    data_filtered = data[data['time_period'] == period]
    data_filtered['test_case_ratio'] = data_filtered['test count'] / data_filtered['case count']
    data_filtered['test_positive_ratio'] = data_filtered['test count'] / data_filtered['positive tests']
    data_filtered['case_positive_ratio'] = data_filtered['case count'] / data_filtered['positive tests']
    
    grouped_data = data_filtered.groupby(['zcta'])[['test_case_ratio', 'test_positive_ratio', 'case_positive_ratio']].sum()
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    
    grouped_data['test_case_ratio'].plot(kind='bar', ax=axes[0], color= custom_colors[0])
    axes[0].set_title('Test count / Case count  Ratio for the period ' + str(period))
    axes[0].set_xlabel('Zone (ZIPCODE)')
    axes[0].set_ylabel('Test count / Case count  Ratio')
    
    grouped_data['test_positive_ratio'].plot(kind='bar', ax=axes[1], color= custom_colors[1])
    axes[1].set_title('Test count / Positive tests Ratio for the period ' + str(period))
    axes[1].set_xlabel('Zone (ZIPCODE)')
    axes[1].set_ylabel('Test count / Positive tests')
    
    grouped_data['case_positive_ratio'].plot(kind='bar', ax=axes[2], color= custom_colors[2])
    axes[2].set_title('Case count / Positive tests Ratio for the period ' + str(period))
    axes[2].set_xlabel('Zone (ZIPCODE)')
    axes[2].set_ylabel('Case count / Positive tests Ratio')
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        plt.savefig(temp_file, format='png')
        temp_file_path = temp_file.name
        plt.close(fig)
    return temp_file_path
def box_plot(data,pre,area,attribute):
    data = opendatafile(data,pre)
    data = data[data['zcta'] == int(area)]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x=data[attribute], ax=ax, color=custom_colors[0])
    ax.set_title(f'Boxplot of {attribute}')
    ax.set_xlabel(attribute)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        plt.savefig(temp_file, format='png')
        temp_file_path = temp_file.name
        plt.close(fig)
    return temp_file_path
def box_plot1(data,pre,attribute):
    data = opendatafile(data,pre)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x=data[attribute], ax=ax, color=custom_colors[0])
    ax.set_title(f'Boxplot of {attribute}')
    ax.set_xlabel(attribute)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        plt.savefig(temp_file, format='png')
        temp_file_path = temp_file.name
        plt.close(fig)
    return temp_file_path
def scatter_plot(data,pre,attribute1,attribute2):
    data = opendatafile(data,pre)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(data=data, x=attribute1, y=attribute2, ax=ax, color=custom_colors[0])
    ax.set_title(f'Scatter plot of {attribute1} and {attribute2}')
    ax.set_xlabel(attribute1)
    ax.set_ylabel(attribute2)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        plt.savefig(temp_file, format='png')
        temp_file_path = temp_file.name
        plt.close(fig)
    return temp_file_path
def histogram(data,pre,attribute):
    data = opendatafile(data,pre)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data=data, x=attribute, ax=ax, color=custom_colors[0])
    ax.set_title(f'Histogram of {attribute}')
    ax.set_xlabel(attribute)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        plt.savefig(temp_file, format='png')
        temp_file_path = temp_file.name
        plt.close(fig)
    return temp_file_path
def get_periode():
    data = pd.read_csv("./Data/Dataset2_final.csv")
    list = data['time_period'].unique().tolist()
    return sorted(list)
def get_attribut_data2():
    data = pd.read_csv("./Data/Dataset2_final.csv")
    attributes = data.select_dtypes(include='number').columns
    attributes = attributes.drop(['zcta','time_period','population'])
    return attributes.tolist()
def get_attribut_data1():
    data = pd.read_csv("./Data/Dataset1_final.csv")
    attributes = data.select_dtypes(include='number').columns
    attributes = attributes.drop(['Fertility'])
    return attributes.tolist()
def get_attribut_data3():
    data = pd.read_csv("./Data/Dataset3_final.csv")
    attributes = data.select_dtypes(include='number').columns
    return attributes.tolist()

def show_assoc_rules(min_conf,min_sup,min_cor,mesure):
    data = pd.read_csv("./Data/Dataset3_final.csv")
    df = data[['Soil', 'Crop', 'Fertilizer', 'Rainfall']]
    df = pd.DataFrame({'transaction': [set(row) for _, row in df.iterrows()]})
    data = []
    for index, row in df.iterrows():
        transaction = row['transaction']
        data.append(set(transaction))  
    association_rules, _, _ = appriori.regles_d_association(data, min_sup, min_conf, min_cor, mesure) 
    rule_data = []
    for rule in association_rules:
        A, B, confidence, sup = rule
        rule_data.append({
            'Rule': f"{list(A)} => {list(B)}",
            'Confidence': confidence,
            'Support': sup
        })

    df_rules = pd.DataFrame(rule_data)

    return df_rules

def recommendation(soil,crop,rinfall,min_conf,min_sup,min_cor,mesure):
    input_data = [soil,crop,rinfall]
    data = pd.read_csv("./Data/Dataset3_final.csv")
    df = data[['Soil', 'Crop', 'Fertilizer', 'Rainfall']]
    df = pd.DataFrame({'transaction': [set(row) for _, row in df.iterrows()]})
    data = []
    for index, row in df.iterrows():
        transaction = row['transaction']
        data.append(set(transaction))  
    association_rules, _, _ = appriori.regles_d_association(data, min_sup, min_conf, min_cor, mesure)
    matching_sets_B = []
    for rule in association_rules:
        A, B, _, _ = rule
        if set(input_data) == A:
            matching_sets_B.append(list(B))
    if matching_sets_B:
        return "Recommanded "+str(matching_sets_B)
    else:
        return "SORRY we can't help with that! No matching rule found for your input data :( "