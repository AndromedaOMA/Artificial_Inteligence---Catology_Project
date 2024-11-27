# import numpy as np
# from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
# import pandas as pd
# import xlrd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import LabelEncoder

#
# def detect_errors_in_data_set(d):
#     result = dict()
#
#     missed_values = d.isnull().sum()
#     if missed_values.any():
#         result['missed_values'] = missed_values
#     else:
#         result['missed_values'] = 'no missed values!'
#
#     duplicated_data = d[data.duplicated()]
#     if not duplicated_data.empty:
#         result['duplicated_data'] = duplicated_data
#     else:
#         result['duplicated_data'] = 'no duplicated data!'
#     # print(data['Logement'])
#     return result
#
# def compute_no_of_instances(d):
#     breed_set = set(d['Logement'])
#     print(f'Breed set: {breed_set}')
#     result = dict()
#     for breed in breed_set:
#         result[breed] = d['Logement'].value_counts()[breed]
#     return result
#
#
# def compute_frequency_of_values(d):
#     attribute_set = set(d)
#     # print(attribute_set)
#
#     result = dict()
#
#     for a in attribute_set:
#         result[a] = d[a].value_counts()
#
#     return result
#
#
# def plot_distribution_with_seaborn(d):
#     numerical_columns = d.select_dtypes(include=['float64', 'int64']).columns
#
#     # Apply the default theme
#     sns.set_theme()
#
#     # Load an example dataset
#     # dots = sns.load_dataset("dots")
#
#     # Create a visualization
#     for column in numerical_columns:
#         # sns.relplot(
#         #     data=dots, kind="line",
#         #     x=d[column], y="time", col="align",
#         #     hue="choice", size="coherence", style="choice",
#         #     facet_kws=dict(sharex=False),
#         # )
#         sns.histplot(d[column], kde=True)
#
#         plt.title(f'Distribution of {column}', fontsize=15)
#         plt.xlabel(column, fontsize=10)
#         plt.ylabel('Frequency', fontsize=10)
#
#         plt.show()
#
#
# # Handle missing data
# # data.fillna(-1, inplace=True)
# def digit_convertor(d):
#     # pd.get_dummies(d).to_excel("encoded_data_set_cat.xlsx")
#     label = LabelEncoder()
#     for column in d.select_dtypes(include=['object']).columns:
#         d[column] = label.fit_transform(data[column])
#     pd.get_dummies(d).to_excel("labelencoder_data_set_cat.xlsx")
#
# data = pd.read_excel("labelencoder_data_set_cat.xlsx")
# print(f"Error detections: {detect_errors_in_data_set(data)}")
# print(f"No. of instances per class: {compute_no_of_instances(data)}")
# print(f"Attributes and their value frequency: {compute_frequency_of_values(data)}")
# digit_convertor(data)
# plot_distribution_with_seaborn(data)


import numpy as np
import time
import pandas as p

# import numpy as np
# import openpyxl
#
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


def prepare_data_set():
    dataset = p.read_excel('labelencoder_data_set_cat.xlsx')
    # print(f"{dataset.dtypes}")
    dataset = dataset.apply(p.to_numeric, errors='coerce')
    print(dataset.head())

    # debugging
    # u_values = {col: dataset[col].unique() for col in dataset.columns}
    # u_values.pop('Plus')
    # for col, val in u_values.items():
    #     print(f"'{col}': {val}")

    for col in dataset:
        dataset[col] = (((dataset[col]+1) - (dataset[col].min()+1)) / ((dataset[col].max()+1) - (dataset[col].min()+1))).round(2)

    x = dataset.drop(['Race', 'Plus'], axis=1).values
    y = dataset['Race'].values

    # makes sure that the randomizers work the same all the time!
    np.random.seed(42)

    indicies = np.arange(x.shape[0])
    np.random.shuffle(indicies)

    train_indicies = indicies[:int(0.8*len(indicies))]
    test_indicies = indicies[int(0.8*len(indicies)):]
    # train_x, train_y, test_x, test_y
    return x[train_indicies], y[train_indicies], x[test_indicies], y[test_indicies]


train_x, train_y, test_x, test_y = prepare_data_set()


print(f"Here we got the #row  {len(train_x)} and #col: {len(train_x[0])} of training data.\n")
print(f"Here we got the length of the first training data: {len(train_x[0])};\n"
      f" The first training data: {train_x[0]};\n"
      f" The first training label: {train_y[0]};")
print(f"{train_x.shape}")


def batches_generator(train_data, train_labels, no_of_batches):
    """YIELD (continuous "return") the current batches of 100 elements each"""

    # makes sure that the randomizers work the same all the time!
    # np.random.seed(42)

    indices = np.arange(len(train_data))
    np.random.shuffle(indices)

    for i in range(0, len(train_data), no_of_batches):
        batch_indices = indices[i:i + no_of_batches]
        yield train_data[batch_indices], train_labels[batch_indices]


def sigmoid(x, backpropagation=False):
    # Clip values to prevent overflow
    x = np.clip(x, -500, 500)  # Clip x to avoid large values

    s = 1 / (1 + np.exp(-x))
    if backpropagation:
        return s * (1 - s)
    return s


def relu(x, backpropagation=False):
    if backpropagation:
        return (x > 0).astype(float)
    return np.maximum(0, x)


def softmax(x, backpropagation=False):
    # # Clip values to prevent overflow
    x = np.clip(x, -500, 500)  # Clip x to avoid large values

    exp = np.exp(x - np.max(x))
    s = exp / np.sum(exp, axis=0, keepdims=True)
    if backpropagation:
        return s * (1 - s)
    return s


class NN:
    def __init__(self, sizes=None, epochs=50, batches=1000, learning_rate=0.001, dropout_rate=0.001):
        if sizes is None:
            sizes = [25, 100, 13]
        self.sizes = sizes
        self.epochs = epochs
        self.batches = batches
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        in_layer = self.sizes[0]
        h_layer = self.sizes[1]
        out_layer = self.sizes[2]

        """configurare Xavier pentru model ce utilizează funcții simetrice
           (tocmai pentru evitarea blocării rețelei neuronale)"""
        self.params = {
            'W1': np.random.randn(h_layer, in_layer) * np.sqrt(2 / (h_layer + in_layer)),  #200x25
            'W2': np.random.randn(out_layer, h_layer) * np.sqrt(2 / (out_layer + h_layer))  #13x200
        }

    def forward_prop(self, x_train, train=True):
    # def forward_prop(self, x_train):
        params = self.params

        params['A0'] = x_train  #27x100
        params['Z1'] = np.dot(params['W1'], params['A0'].T)  #100x100
        params['A1'] = relu(params['Z1'])  #100x100

        if train:
            dropout_mask = np.random.rand(*params['A1'].shape) < (1 - self.dropout_rate)
            params['A1'] *= dropout_mask
            params['A1'] /= (1 - self.dropout_rate)

        params['Z2'] = np.dot(params['W2'], params['A1'])  #10x100
        params['A2'] = softmax(params['Z2'])  #10x100

        return params['A2']

    def backward_prop(self, y_train, output):
        params = self.params

        # err = output - y_train.T
        err = (output - y_train.T) * softmax(params['Z2'], backpropagation=True)
        # params['W2'] -= self.learning_rate * np.outer(err, params['A1'])
        params['W2'] -= self.learning_rate * np.dot(err, params['A1'].T) / len(y_train)

        err = np.dot(params['W2'].T, err) * relu(params['Z1'], backpropagation=True)
        # params['W1'] -= self.learning_rate * np.outer(err, params['A0'])
        params['W1'] -= self.learning_rate * np.dot(err, params['A0']) / len(y_train)

    def compute_acc(self, test_data, test_labels):
        predictions = []
        # for data_batch, label_batch in batches_generator(test_data, test_labels, self.batches):
        for i in range(len(test_data)):
            output = self.forward_prop(test_data[i], train=False)
            # output = self.forward_prop(test_data[i])
            predict = np.argmax(output)
            predictions.append(predict == np.argmax(test_labels[i]))
        return np.mean(predictions)

    def train(self, train_list, train_labels, test_list, test_labels):
        start_time = time.time()
        for i in range(self.epochs):
            for data_batch, label_batch in batches_generator(train_list, train_labels, self.batches):
            # for j in range(len(train_list)):
                output = self.forward_prop(data_batch, train=True)
                self.backward_prop(label_batch, output)

            accuracy = self.compute_acc(test_list, test_labels)
            print(f'Epoch: {i + 1}, Time Spent: {time.time() - start_time}s, Accuracy: {accuracy * 100}%')


if __name__ == "__main__":
    nn = NN()
    nn.train(train_x, train_y, test_x, test_y)
