import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


class PrepareDataSet:
    def __init__(self, file_path='./data_sets/data_set_cat.xlsx'):
        self.file_path = file_path
        self.data = pd.read_excel(self.file_path, 'Data')
        self.dict_of_values_to_compare = {
            'Sexe': ['M', 'F'],
            'Race': ['BEN', 'SBI', 'BRI', 'CHA', 'EUR', 'MCO', 'PER', 'RAG', 'SPH', 'ORI', 'TUV', 'Autre', 'NSP']
        }

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

    def compute_no_of_instances(self, atr='Race'):
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

    def correct_data_set(self):
        column_names = {col_index: self.data.columns[col_index] for col_index in range(2, 28)}
        rows_to_drop = []

        for column_index in range(2, 28):
            current_col_name = column_names[column_index]
            for row_index in range(0, len(self.data)):
                value = self.data.iloc[row_index, column_index]
                match current_col_name:
                    case ('Ext' | 'Obs' | 'PredOiseau' | 'PredMamm'):
                        self.data.iloc[row_index, column_index] = int(value) + 1
                    case ('Sexe' | 'Race'):
                        if str(value) not in self.dict_of_values_to_compare[current_col_name]:
                            rows_to_drop.append(row_index)

        # Eliminăm toate rândurile nevalide și resetăm indecșii
        self.data.drop(index=rows_to_drop, inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        # Scrierea setului de date corectat într-un fișier Excel
        self.data.to_excel('./data_sets/corrected_data_set_cat.xlsx', index=False)

    # One_Hot_Encoding
    def digit_convertor(self):
        updated_data = pd.read_excel('./data_sets/corrected_data_set_cat.xlsx')
        updated_data.drop('Horodateur', axis=1)
        label = LabelEncoder()
        for column in updated_data.select_dtypes(include=['object']).columns:
            self.data[column] = label.fit_transform(self.data[column])

        ohe = pd.get_dummies(self.data, drop_first=True)
        for column_index in range(len(ohe)):
            for row_index in range(1, len(ohe)):
                value = int(ohe.iloc[row_index, column_index])
                ohe.iloc[row_index, column_index] = value + 1

        ohe.to_excel("./data_sets/encoded_data_set_cat.xlsx", index=False)

    # One_Hot_Encoding
    def ohe_plus_one(self):
        updated_data = pd.read_excel('./data_sets/old_labelencoder_data_set_cat.xlsx')

        for column_index in range(0, 26):
            for row_index in range(0, len(updated_data)):
                value = int(updated_data.iloc[row_index, column_index])
                updated_data.iloc[row_index, column_index] = value + 1

        updated_data.to_excel("./data_sets/encoded_data_set_cat.xlsx", index=False)

    # # One_Hot_Encoding
    # def digit_convertor(self):
    #     updated_data = pd.read_excel('./data_sets/corrected_data_set_cat.xlsx')
    #     label = LabelEncoder()
    #
    #     columns_to_encode = ['Sexe', 'Age', 'Race', 'Nombre', 'Logement', 'Zone', 'Abondance']
    #
    #     for col in columns_to_encode:
    #         if col in updated_data.columns:
    #             updated_data[col] = label.fit_transform(updated_data[col])
    #
    #     ohe = pd.get_dummies(updated_data, drop_first=True)
    #     for column_index in range(len(ohe)):
    #         for row_index in range(1, len(ohe)):
    #             value = ohe.iloc[row_index, column_index]
    #             ohe.iloc[row_index, column_index] = int(value) + 1
    #
    #     ohe.to_excel("./data_sets/encoded_data_set_cat.xlsx", index=False)
    #
    #     updated_data.to_excel("./data_sets/encoded_data_set_cat.xlsx", index=False)