import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from random import seed
from random import randrange
from random import random
from scipy import exp
from sklearn.model_selection import KFold


def load_csv(filename):
    if filename == 'iris.csv':
        dataset = pd.read_csv(filename, names=['sepal length', 'sepal width', 'petal length',
                                               'petal width', 'class'])
        for index, row in dataset.iterrows():
            if row['class'] == 'Iris-setosa':
                dataset.set_value(index, 'class', 0)
            elif row['class'] == 'Iris-versicolor':
                dataset.set_value(index, 'class', 1)
            else:
                dataset.set_value(index, 'class', 2)
    else:
        dataset = pd.read_csv(filename, names=['x1', 'x2', 'y'])
    return dataset


def minmax(dataset):
    minmax = list()
    stats = [[dataset[column].min(), dataset[column].max()]
             for column in dataset]

    return stats


def normalize(dataset, minmax):
    for index, row in dataset.iterrows():
        for i in range(len(row) - 1):
            dataset.set_value(
                index, i, (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0]))
    dataset['class'] = dataset['class'].apply(np.int)


def cross_validation_split(dataset, n_folds):
    kf = KFold(n_splits=5, shuffle=True, random_state=2)
    return kf


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def plot_synthetic(test_set, predicted):
    colormap = np.array(['r', 'k'])

    plt.scatter(test_set[:, 0], test_set[:, 1], c=colormap[predicted])
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    red_patch = mpatches.Patch(color='red', label='False')
    black_patch = mpatches.Patch(color='black', Label='True')
    plt.legend(handles=[black_patch, red_patch])
    plt.xticks(test_set[:, 0])
    plt.yticks(test_set[:, 1])
    plt.title("Logical Or")
    plt.show()


def plot_features(test_set, predicted):
    colormap = np.array(['r', 'k', 'b'])
    fig, axes = plt.subplots(2, 2)

    axes[0, 0].scatter(test_set[:, 0], test_set[:, 4], c=colormap[predicted])
    axes[0, 0].set_xlabel("Sepal Length")
    axes[0, 0].set_ylabel("Actual")
    axes[0, 0].set_yticks(test_set[:, 4])

    axes[0, 1].scatter(test_set[:, 1], test_set[:, 4], c=colormap[predicted])
    axes[0, 1].set_xlabel("Sepal Width")
    axes[0, 1].set_ylabel("Actual")
    axes[0, 1].set_yticks(test_set[:, 4])

    axes[1, 0].scatter(test_set[:, 2], test_set[:, 4], c=colormap[predicted])
    axes[1, 0].set_xlabel("Petal Length")
    axes[1, 0].set_ylabel("Actual")
    axes[1, 0].set_yticks(test_set[:, 4])

    axes[1, 1].scatter(test_set[:, 3], test_set[:, 4], c=colormap[predicted])
    axes[1, 1].set_xlabel("Petal Width")
    axes[1, 1].set_ylabel("Actual")
    axes[1, 1].set_yticks(test_set[:, 4])

    red_patch = mpatches.Patch(color='red', label='Iris setosa')
    black_patch = mpatches.Patch(color='black', Label='Iris versicolor')
    blue_patch = mpatches.Patch(color='blue', Label='Iris verginica')

    axes[0, 0].legend(handles=[red_patch, black_patch, blue_patch])
    axes[0, 1].legend(handles=[red_patch, black_patch, blue_patch])
    axes[1, 0].legend(handles=[red_patch, black_patch, blue_patch])
    axes[1, 1].legend(handles=[red_patch, black_patch, blue_patch])

    fig.suptitle("Flower Features vs Class")
    plt.show()


def classify(dataset, n_folds, l_rate, n_epoch, n_hidden):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()

    for train_index, test_index in folds.split(dataset):

        train_set = np.asarray([[row['sepal length'], row['sepal width'], row['petal length'], row[
            'petal width'], row['class']] for index, row in dataset.iloc[train_index].iterrows()])
        test_set = np.asarray([[row['sepal length'], row['sepal width'], row['petal length'], row[
            'petal width'], row['class']] for index, row in dataset.iloc[test_index].iterrows()])

        predicted = back_propagation(
            train_set, test_set, l_rate, n_epoch, n_hidden)

        plot_features(test_set, predicted)

        actual = [row[-1] for row in test_set]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)

    return scores


def classify_synthetic(dataset, n_folds, l_rate, n_epoch, n_hidden):
    scores = list()
    folds = cross_validation_split(dataset, n_folds)
    for train_index, test_index in folds.split(dataset):

        train_set = np.asarray([[row['x1'], row['x2'], row['y']]
                                for index, row in dataset.iloc[train_index].iterrows()])
        test_set = np.asarray([[row['x1'], row['x2'], row['y']]
                               for index, row in dataset.iloc[test_index].iterrows()])

        predicted = back_propagation(
            train_set, test_set, l_rate, n_epoch, n_hidden)

        plot_synthetic(test_set, predicted)

        actual = [row[-1] for row in test_set]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)

    return scores


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def transfer_derivative(output):
    return output * (1.0 - output)


def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[: -1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        for row in train:
            index = row[-1].astype(int)
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[index] = 1
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)


def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]}
                    for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]}
                    for i in range(n_outputs)]
    network.append(output_layer)
    return network


def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return(predictions)

if __name__ == "__main__":
    seed(12)

    filename = 'or.csv'
    dataset = load_csv(filename)

    n_folds = 5
    l_rate = 0.3
    n_epoch = 500
    n_hidden = 3

    scores = classify_synthetic(
        dataset, n_folds, l_rate, n_epoch, n_hidden)

    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

    n_hidden = 5

    filename = 'iris.csv'
    dataset = load_csv(filename)

    minmax = minmax(dataset)
    normalize(dataset, minmax)

    scores = classify(
        dataset, n_folds, l_rate, n_epoch, n_hidden)

    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
