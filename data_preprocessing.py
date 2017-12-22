#!/bin/env/python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(train_path, test_path):
    """
    Load the training and testing data into Pandas DataFrame.

    Args:
        train_path: String path to training dataset.
        test_path: String path to testing dataset.
    Returns:
        train_data: Training data formatted into Pandas DataFrame.
        test_data: Testing data formatted into Pandas DataFrame.
    """
    col_names = ["age", "workclass", "fnlwgt", "education", "education-num",
                 "marital-status", "occupation", "relationship", "race",
                 "sex", "capital-gain", "capital-loss", "hours-per-week",
                 "native-country", "income"]
    train_data = pd.read_csv(train_path, header=None, names=col_names)
    test_data = pd.read_csv(test_path, header=None, names=col_names)

    print("Training Data Loaded.")
    print("Total training instances:", len(train_data))
    print("Total testing instances:", len(test_data), "\n")

    return train_data, test_data


def clean_data(train_data, test_data):
    """
    Clean the training and testing data by removing invalid data points.

    Args:
        train_data: Train data as Pandas DataFrame.
        test_data: Test data as Pandas DataFrame.
    Returns:
        train_clean: Cleaned testing data in Pandas DataFrame.
        test_clean: Cleaned testing data in Pandas DataFrame.
    """
    # Replace all " ?" with NaN and then drop rows where NaN appears
    train_clean = train_data.replace(' ?', np.nan).dropna()
    test_clean = test_data.replace(' ?', np.nan).dropna()

    print("Number or training instances removed:", len(train_data) - len(train_clean))
    print("Number or testing instances removed:", len(test_data) - len(test_clean))
    print("Total training instances:", len(train_clean))
    print("Total testing instances:", len(test_clean), "\n")

    return train_clean, test_clean


def standardize_data(train_data, test_data):
    """
    Standardize the train and test data to have 0 mean and unit variance.

    Args:
        train_data: Train data as Pandas DataFrame.
        test_data: Test data as Pandas DataFrame.
    Returns:
        train_data: Standardized train data as Pandas DataFrame.
        test_data: Standardized test data as Pandas DataFrame.
    """
    # Fit scaler on train data only. Transform training and testing set
    numerical_col = ["age", "fnlwgt", "education-num", "capital-gain",
                     "capital-loss", "hours-per-week"]
    scaler = StandardScaler()
    train_data[numerical_col] = scaler.fit_transform(train_data[numerical_col])
    test_data[numerical_col] = scaler.transform(test_data[numerical_col])

    return train_data, test_data


def split_data(train_data, test_data):
    """
    Split data into training/testing features and training/testing labels.

    Args:
        train_data: Train dataset as Pandas DataFrame.
        test_data: Test dataset as Pandas DataFrame.
    Returns:
        X_train: Train features as Pandas DataFrame.
        y_train: Train labels as Pandas Series.
        X_test: Test features as Pandas DataFrame.
        y_test: Test labels as Pandas Series.
    """
    y_train = train_data["income"]
    X_train = train_data.drop("income", axis=1)

    y_test = test_data['income']
    X_test = test_data.drop("income", axis=1)

    return X_train, y_train, X_test, y_test


def ohe_data(X_train, y_train, X_test, y_test):
    """
    One hot encode categorical data.

    Args:
        X_train: Train features as Pandas DataFrame.
        y_train: Train labels as Pandas Series.
        X_test: Test features as Pandas DataFrame.
        y_test: Test labels as Pandas Series.
    Returns:
        X_train_ohe: One-hot encoded training features as Pandas DataFrame.
        y_train_ohe: One-hot encoded training labels as Pandas Series.
        X_test_ohe: One-hot encoded testing features as Pandas DataFrame.
        y_test_ohe: One-hot encoded testing labels as Pandas Series.
    """
    data = pd.concat([X_train, X_test])
    data_ohe = pd.get_dummies(data)
    X_train_ohe = data_ohe[:len(X_train)]
    X_test_ohe = data_ohe[len(X_train):]

    y_train_ohe = y_train.replace([' <=50K', ' >50K'], [0, 1])
    y_test_ohe = y_test.replace([' <=50K', ' >50K'], [0, 1])

    return X_train_ohe, y_train_ohe, X_test_ohe, y_test_ohe
