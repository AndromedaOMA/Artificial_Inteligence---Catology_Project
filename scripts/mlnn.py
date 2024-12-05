import numpy as np
import time
import pandas as p


def prepare_data_set():
    # dataset = p.read_excel('./data_sets/old_labelencoder_data_set_cat.xlsx')
    dataset = p.read_excel('./data_sets/corrected_data_set_cat.xlsx')
    # print(f"{dataset.dtypes}")
    dataset = dataset.apply(p.to_numeric, errors='coerce')
    print(dataset.head())

    # debugging
    # u_values = {col: dataset[col].unique() for col in dataset.columns}
    # u_values.pop('Plus')
    # for col, val in u_values.items():
    #     print(f"'{col}': {val}")

    for col in dataset:
        dataset[col] = (((dataset[col]+1) - (dataset[col].min()+1)) /
                        ((dataset[col].max()+1) - (dataset[col].min()+1))).round(2)

    x = dataset.drop(['Race', 'Plus', 'Row.names', 'Horodateur'], axis=1).values
    y = dataset['Race'].values

    # makes sure that the randomizers work the same all the time!
    # np.random.seed(42)

    indexes = np.arange(x.shape[0])
    np.random.shuffle(indexes)

    train_indexes = indexes[:int(0.8*len(indexes))]
    test_indexes = indexes[int(0.8*len(indexes)):]
    # train_x, train_y, test_x, test_y
    return x[train_indexes], y[train_indexes], x[test_indexes], y[test_indexes]


train_x, train_y, test_x, test_y = prepare_data_set()


print(f"\nHere we got the #row: {len(train_x)} and #col: {len(train_x[0])} of training data.\n")
print(f"Here we got the length of the first training data: {len(train_x[0])};\n\n"
      f" The head of training data: \n{train_x[:10]};\n\n"
      f" The first training data:\n {train_x[0]};\n\n"
      f" The first training label:\n {train_y[0]};\n\n"
      f" And the shape of training data:\n {train_x.shape}\n")


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


class MLNN:
    def __init__(self, tr_x=train_x, tr_y=train_y, te_x=test_x, te_y=test_y,
                 sizes=None, epochs=50, batches=100, learning_rate=0.001, dropout_rate=0.25):
        self.train_x = tr_x
        self.train_y = tr_y
        self.test_x = te_x
        self.test_y = te_y
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
            'W1': np.random.randn(h_layer, in_layer) * np.sqrt(2 / (h_layer + in_layer)),  #100x25
            'W2': np.random.randn(out_layer, h_layer) * np.sqrt(2 / (out_layer + h_layer))  #13x100
        }

    def forward_prop(self, x_train, train=True):
    # def forward_prop(self, x_train):
        params = self.params

        params['A0'] = x_train  #25x100
        params['Z1'] = np.dot(params['W1'], params['A0'].T)  #100x100
        params['A1'] = relu(params['Z1'])  #100x100

        if train:
            dropout_mask = np.random.rand(*params['A1'].shape) < (1 - self.dropout_rate)
            params['A1'] *= dropout_mask
            params['A1'] /= (1 - self.dropout_rate)

        params['Z2'] = np.dot(params['W2'], params['A1'])  #13x100
        params['A2'] = softmax(params['Z2'])  #13x100

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

    def compute_acc(self, data, labels):
        correct_predictions = 0
        for i in range(len(data)):
            output = self.forward_prop(data[i], train=False)
            if np.argmax(output) == np.argmax(labels[i]):
                correct_predictions += 1
        return correct_predictions / len(data)

    def train(self):
        start_time = time.time()
        for i in range(self.epochs):
            for data_batch, label_batch in batches_generator(self.train_x, self.train_y, self.batches):
                output = self.forward_prop(data_batch, train=True)
                self.backward_prop(label_batch, output)

            test_accuracy = self.compute_acc(self.test_x, self.test_y)
            print(f'Epoch: {i + 1}, '
                  f'Time Spent: {np.around(time.time() - start_time, 2)}s, '
                  f'Test accuracy: {np.around(test_accuracy * 100, 2)}%')
