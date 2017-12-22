#!/bin/env/python3

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
import argparse
import os
from data_preprocessing import *


def preprocess_data():
    """
    Preprocess the data.

    Args:
        None
    Returns:
        X_train: Training features as Pandas DataFrame.
        y_train: Training labels as Pandas Series.
        X_test: Testing features as Pandas DataFrame.
        y_test: Testing labels as Pandas Series.
    """
    path_to_train = "data" + os.sep + "train_data.txt"
    path_to_test = "data" + os.sep + "test_data.txt"

    # Load the data
    print("Loading data...")
    train_data, test_data = load_data(path_to_train, path_to_test)
    # Clean the data
    print("Cleaning data...")
    train_clean, test_clean = clean_data(train_data, test_data)
    # Standardize the data
    print("Standardizing the data...")
    train_data, test_data = standardize_data(train_data, test_data)
    # Split data into features and labels
    X_train, y_train, X_test, y_test = split_data(train_data, test_data)
    # One-hot encode the data
    X_train, y_train, X_test, y_test = ohe_data(X_train, y_train, X_test, y_test)

    return X_train, y_train, X_test, y_test


def train_and_validate(algorithm):
    """
    Train and validate machine learning model using algorithm and data provided.

    Args:
        algorithm: String defining ML algorithm to use (naive_bayes,
                   decision_tree, knn, or svm).
    Returns:
        Prints out classification accuracy.
    """
    X_train, y_train, X_test, y_test = preprocess_data()
    print("\nData sucessfully loaded.")
    input("Press ENTER to continue...")

    clf = None
    parameters = {}
    if algorithm == "naive_bayes":
        clf = GaussianNB()
    elif algorithm == "decision_tree":
        clf = DecisionTreeClassifier()
        parameters = {"criterion": ("gini", "entropy"),
                      "max_depth": (None, 2, 3),
                      "min_samples_split": (2, 3, 4)}
    elif algorithm == "knn":
        clf = KNeighborsClassifier()
        parameters = {"n_neighbors": (3, 5, 6, 8, 10, 15),
                      "weights": ("uniform", "distance")}
    elif algorithm == "svm":
        clf = SVC()
        parameters = {"C": (0.1, 1, 5, 10),
                      "kernel": ("rbf", "linear"),
                      "gamma": (0.1, 0.5, 1)}
    else:
        print("Error: Model not found.")
        return

    clf_gs = GridSearchCV(clf, parameters, verbose=1)
    print("Training model...")
    clf_gs.fit(X_train, y_train)
    y_pred = clf_gs.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", acc * 100.0)


if __name__ == "__main__":
    """
    Run main program.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--clf", help="Define the type of classifier to use.")
    args = parser.parse_args()
    if args.clf == "naive_bayes":
        train_and_validate("naive_bayes")
    elif args.clf == "decision_tree":
        train_and_validate("decision_tree")
    elif args.clf == "knn":
        train_and_validate("knn")
    elif args.clf == "svm":
        train_and_validate("svm")
    else:
        print("No classifier provided. Using naive_bayes as default.")
        train_and_validate("naive_bayes")
