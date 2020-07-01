
"""
## Version history:
2019:
   Original script by Jodie Ashford [ashfojsm], Aston University
2020, January:
	Revised, and updated by Jodie Ashford [ashfojsm], Aston University
	(ashfojsm@aston.ac.uk)
"""

import pandas as pd
import Feature_selection as fs
from sklearn import svm
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib



def run_svm(training_path, test_size, clf_output_file):
    """
    Builds and saves a trained support vector machine classifier.
    :param training_path: String
        File path for the training matrix.
    :param test_size: float
        Proportion of data to use for testing.
    :param clf_output_file: String
        Name of file to save the classifier to.
    """

    # Load dataset into pandas.DataFrame
    data = pd.read_csv(training_path)

    # Feature selection
    # Note: feature selection is based on the entire dataset
    selected_features = fs.feature_selection(training_path)
    # Create new dataset containing only selected features
    feature_names_plus_label = selected_features.copy()
    feature_names_plus_label.append("Label")
    selected_data = data[feature_names_plus_label].copy()

    # Split dataset into train and test (x is data, y is labels)
    x_train, x_test, y_train, y_test = split_dataset(selected_data, test_size)

    # Train Support Vector Machine classifier
    clf = svm.SVC()
    clf = clf.fit(x_train, y_train)
    # Save the model
    joblib.dump(clf, clf_output_file)

    # Predict on the testing data
    y_predict = clf.predict(x_test)
    print("Accuracy of SVM classifier = " + str(accuracy_score(y_test, y_predict)))

    return None

def run_knn(training_path, test_size, clf_output_file):
    """
    Builds and saves a trained K nearest neighbour classifier.
    :param training_path: String
        File path for the training matrix.
    :param test_size: float
        Proportion of data to use for testing.
    :param clf_output_file: String
        Name of file to save the classifier to.
    """

    # Load dataset into pandas.DataFrame
    data = pd.read_csv(training_path)

    # Feature selection
    # Note: feature selection is based on the entire dataset
    selected_features = fs.feature_selection(training_path)
    # Create new dataset containing only selected features
    feature_names_plus_label = selected_features.copy()
    feature_names_plus_label.append("Label")
    selected_data = data[feature_names_plus_label].copy()

    # Split dataset into train and test (x is data, y is labels)
    x_train, x_test, y_train, y_test = split_dataset(selected_data, test_size)

    # Train K Nearest Neighbour classifier
    clf = NearestCentroid()
    clf = clf.fit(x_train, y_train)
    # Save the model
    joblib.dump(clf, clf_output_file)

    # Predict on the testing data
    y_predict = clf.predict(x_test)
    print("Accuracy of KNN classifier = " + str(accuracy_score(y_test, y_predict)))

    return None

def run_random_forest(training_path, test_size, clf_output_file):
    """
    Builds and saves a trained random forest classifier.
    :param training_path: String
        File path for the training matrix.
    :param test_size: float
        Proportion of data to use for testing.
    :param clf_output_file: String
        Name of file to save the classifier to.
    """

    # Load dataset into pandas.DataFrame
    data = pd.read_csv(training_path)

    # Feature selection
    # Note: feature selection is based on the entire dataset
    selected_features = fs.feature_selection(training_path)
    # Create new dataset containing only selected features
    feature_names_plus_label = selected_features.copy()
    feature_names_plus_label.append("Label")
    selected_data = data[feature_names_plus_label].copy()

    # Split dataset into train and test (x is data, y is labels)
    x_train, x_test, y_train, y_test = split_dataset(selected_data, test_size)

    # Train Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(x_train, y_train)
    # Save the model
    joblib.dump(clf, clf_output_file)

    # Predict on the testing data
    y_predict = clf.predict(x_test)
    print("Accuracy of Random Forest classifier = " + str(accuracy_score(y_test, y_predict)))

    return None

def run_ada_boost(training_path, test_size, clf_output_file):
    """
    Builds and saves a trained AdaBoost classifier.
    :param training_path: String
        File path for the training matrix.
    :param test_size: float
        Proportion of data to use for testing.
    :param clf_output_file: String
        Name of file to save the classifier to.
    """

    # Load dataset into pandas.DataFrame
    data = pd.read_csv(training_path)

    # Feature selection
    # Note: feature selection is based on the entire dataset
    selected_features = fs.feature_selection(training_path)
    # Create new dataset containing only selected features
    feature_names_plus_label = selected_features.copy()
    feature_names_plus_label.append("Label")
    selected_data = data[feature_names_plus_label].copy()

    # Split dataset into train and test (x is data, y is labels)
    x_train, x_test, y_train, y_test = split_dataset(selected_data, test_size)

    # Train Random Forest classifier
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf = clf.fit(x_train, y_train)
    # Save the model
    joblib.dump(clf, clf_output_file)

    # Predict on the testing data
    y_predict = clf.predict(x_test)
    print("Accuracy of Ada Boost classifier = " + str(accuracy_score(y_test, y_predict)))

    return None

def run_mlp(training_path, test_size, clf_output_file):
    """
    Builds and saves a trained multi layer perceptron classifier.
    :param training_path: String
        File path for the training matrix.
    :param test_size: float
        Proportion of data to use for testing.
    :param clf_output_file: String
        Name of file to save the classifier to.
    """

    # Load dataset into pandas.DataFrame
    data = pd.read_csv(training_path)

    # Feature selection
    # Note: feature selection is based on the entire dataset
    selected_features = fs.feature_selection(training_path)
    # Create new dataset containing only selected features
    feature_names_plus_label = selected_features.copy()
    feature_names_plus_label.append("Label")
    selected_data = data[feature_names_plus_label].copy()

    # Split dataset into train and test (x is data, y is labels)
    x_train, x_test, y_train, y_test = split_dataset(selected_data, test_size)

    # Train Multi Layer Perceptron classifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf = clf.fit(x_train, y_train)
    # Save the model
    joblib.dump(clf, clf_output_file)

    # Predict on the testing data
    y_predict = clf.predict(x_test)
    print("Accuracy of Multilayer Perceptron classifier = " + str(accuracy_score(y_test, y_predict)))

    return None


def split_dataset(dataset, test_size):
    """
    Split the data into train and test sets.
    :param dataset: pandas.DataFrame
        The dataset to split.
    :param test_size: float
        Proportion of data to use for testing.
    :return: Training and test sets.
    """
    x = dataset.drop('Label', axis=1)
    y = dataset['Label']
    # split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    return x_train, x_test, y_train, y_test


