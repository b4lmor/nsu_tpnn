from math import log2

import numpy as np
import pandas as pd


def calculate_correlation_matrix(data):
    def mean(x):
        return sum(x) / len(x)

    def covariance(x, y):
        n = len(x)
        return sum((x[i] - mean(x)) * (y[i] - mean(y)) for i in range(n)) / (n - 1)

    def standard_deviation(x):
        n = len(x)
        return (sum((x[i] - mean(x)) ** 2 for i in range(n)) / (n - 1)) ** 0.5

    def pearson_correlation(x, y):
        cov = covariance(x, y)
        std_x = standard_deviation(x)
        std_y = standard_deviation(y)

        if std_x == 0 or std_y == 0:
            return 0

        return cov / (std_x * std_y)

    numeric_columns = data.columns
    n_features = len(numeric_columns)

    correlation_matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(n_features):
            correlation_matrix[i, j] = pearson_correlation(data[numeric_columns[i]], data[numeric_columns[j]])

    return correlation_matrix, numeric_columns


def select_features_with_gain_ratio(data, target):
    def entropy(target_col):
        elements, counts = np.unique(target_col, return_counts=True)

        if len(elements) == 1:
            return 0

        return -np.sum([(counts[i] / np.sum(counts)) * log2(counts[i] / np.sum(counts)) for i in range(len(elements))])

    def information_gain(data, feature, target):
        total_entropy = entropy(data[target])

        values, counts = np.unique(data[feature], return_counts=True)

        weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(data[data[feature] == values[i]][target])
                                   for i in range(len(values))])

        return total_entropy - weighted_entropy

    def split_information(data, feature):
        values, counts = np.unique(data[feature], return_counts=True)

        if len(values) == 1:
            return 0

        return -np.sum([(counts[i] / np.sum(counts)) * log2(counts[i] / np.sum(counts)) for i in range(len(values))])

    def gain_ratio(data, feature, target):
        ig = information_gain(data, feature, target)
        si = split_information(data, feature)

        return ig / si if si != 0 else 0

    for column in data.columns:
        if data[column].dtype == 'float64' or data[column].dtype == 'int64':
            if len(np.unique(data[column])) > 2:
                data[column] = pd.cut(data[column], bins=2, labels=False)

    features = data.columns.drop(target)
    gain_ratios = {feature: gain_ratio(data, feature, target) for feature in features}

    sorted_features = sorted(gain_ratios.items(), key=lambda x: x[1], reverse=True)

    return sorted_features
