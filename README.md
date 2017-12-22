# ML-Census-Income
Using machine learning algorithms on the Census Income dataset

## Requirements
* Python 3.5 and up
* Sci-kit Learn 0.17.1 and up
* Numpy 1.13.1 and up
* Pandas 0.20.3 and up

## File Descriptions
**data_preprocessing.py**: Load, normalize, and process the train/test data.

**train_and_test.py**: Train a machine learning algorith (either naive-bayes, knn, decision tree, or svm) on the census income data to predict whether income exceeds $50k per year.


## Data
The data used is from the UCI Machine Learning Repository and can be found [here](https://archive.ics.uci.edu/ml/datasets/adult). For the program to run properly make sure the train and test data are titled "train_data.txt" and "test_data.txt" respectively and are both in a folder titled "data" that is in the same directory as both python files.

## Usage
From terminal/command prompt navigate to the directory where *data_preprocessing.py*, *train_test.py*, and *data* are located and run one of the following commands.

`python -c naive_bayes`

`python -c decision_tree`

`python -c knn`

`python -c svm`

Note: If no classifier is provided then it defaults to naive bayes.
