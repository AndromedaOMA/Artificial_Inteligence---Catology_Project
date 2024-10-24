import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import pandas as pd
import xlrd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


def detect_errors_in_data_set(d):
    result = dict()

    missed_values = d.isnull().sum()
    if missed_values.any():
        result['missed_values'] = missed_values
    else:
        result['missed_values'] = 'no missed values!'

    duplicated_data = d[data.duplicated()]
    if not duplicated_data.empty:
        result['duplicated_data'] = duplicated_data
    else:
        result['duplicated_data'] = 'no duplicated data!'
    # print(data['Logement'])
    return result

def compute_no_of_instances(d):
    breed_set = set(d['Logement'])
    print(f'Breed set: {breed_set}')
    result = dict()
    for breed in breed_set:
        result[breed] = d['Logement'].value_counts()[breed]
    return result


def compute_frequency_of_values(d):
    attribute_set = set(d)
    # print(attribute_set)

    result = dict()

    for a in attribute_set:
        result[a] = d[a].value_counts()

    return result


def plot_distribution_with_seaborn(d):
    numerical_columns = d.select_dtypes(include=['float64', 'int64']).columns

    # Apply the default theme
    sns.set_theme()

    # Load an example dataset
    # dots = sns.load_dataset("dots")

    # Create a visualization
    for column in numerical_columns:
        # sns.relplot(
        #     data=dots, kind="line",
        #     x=d[column], y="time", col="align",
        #     hue="choice", size="coherence", style="choice",
        #     facet_kws=dict(sharex=False),
        # )
        sns.histplot(d[column], kde=True)

        plt.title(f'Distribution of {column}', fontsize=15)
        plt.xlabel(column, fontsize=10)
        plt.ylabel('Frequency', fontsize=10)

        plt.show()


# Handle missing data
# data.fillna(-1, inplace=True)
def digit_convertor(d):
    # pd.get_dummies(d).to_excel("encoded_data_set_cat.xlsx")
    label = LabelEncoder()
    for column in d.select_dtypes(include=['object']).columns:
        d[column] = label.fit_transform(data[column])
    pd.get_dummies(d).to_excel("labelencoder_data_set_cat.xlsx")

data = pd.read_excel("labelencoder_data_set_cat.xlsx")
print(f"Error detections: {detect_errors_in_data_set(data)}")
print(f"No. of instances per class: {compute_no_of_instances(data)}")
print(f"Attributes and their value frequency: {compute_frequency_of_values(data)}")
digit_convertor(data)
plot_distribution_with_seaborn(data)
