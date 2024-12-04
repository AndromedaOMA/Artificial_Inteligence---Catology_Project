import numpy as np
import openpyxl
import pandas
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import pandas as pd
import xlrd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


class PrepareDataSet:
    def __init__(self, file_path='data_set_cat.xlsx'):
        self.file_path = file_path
        self.data = pandas.read_excel(self.file_path, 'Data')

    def detect_errors_in_data_set(self):
        result = dict()

        missed_values = self.data.isnull().sum()
        if missed_values.any():
            result['missed_values'] = missed_values
        else:
            result['missed_values'] = 'no missed values!'

        duplicated_data = self.data[self.data.duplicated()]
        if not duplicated_data.empty:
            result['duplicated_data'] = duplicated_data
        else:
            result['duplicated_data'] = 'no duplicated data!'
        # print(data['Logement'])
        return result

    def compute_no_of_instances(self, atr='Logement'):
        breed_set = set(self.data[atr])
        print(f'Breed set: {breed_set}')
        result = dict()
        for breed in breed_set:
            result[breed] = self.data[atr].value_counts()[breed]
        return result

    def compute_frequency_of_values(self):
        attribute_set = set(self.data)
        # print(attribute_set)
        result = dict()
        for a in attribute_set:
            result[a] = self.data[a].value_counts()
        return result

    def plot_distribution_with_seaborn(self):
        numerical_columns = self.data.select_dtypes(include=['float64', 'int64']).columns

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
            sns.histplot(self.data[column], kde=True)

            plt.title(f'Distribution of {column}', fontsize=15)
            plt.xlabel(column, fontsize=10)
            plt.ylabel('Frequency', fontsize=10)

            plt.show()

    # Handle missing data
    def digit_convertor(self):
        # pd.get_dummies(d).to_excel("encoded_data_set_cat.xlsx")
        label = LabelEncoder()
        for column in self.data.select_dtypes(include=['object']).columns:
            self.data[column] = label.fit_transform(self.data[column])
        pd.get_dummies(self.data).to_excel("labelencoder_data_set_cat.xlsx")


p = PrepareDataSet()
print(f"Error detections: {p.detect_errors_in_data_set()}")
print(f"No. of instances per class: {p.compute_no_of_instances()}")
print(f"Attributes and their value frequency: {p.compute_frequency_of_values()}")
# p.digit_convertor()
# p.plot_distribution_with_seaborn()


# # Load the workbook and select the active sheet
# file_path = 'labelencoder_data_set_cat.xlsx'
# workbook = openpyxl.load_workbook(file_path)
# sheet = workbook.active
#
# # Iterate through each column
# for col in sheet.iter_cols():
#     # Extract the values from the column (skip None values)
#     values = [cell.value for cell in col if cell.value is not None]
#
#     for cell in col:
#         if cell.value is not None:
#             cell.value = ((cell.value + 1) - np.min(values)) / (np.max(values) - np.min(values))
#
# workbook.save('modified_file.xlsx')
