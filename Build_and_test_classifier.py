
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib


def build_classifier(training_path, test_size, clf_output_file):
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
    selected_features = fs.run_select_k_best(training_path)
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
    print("Accuracy of classifier = " + str(accuracy_score(y_test, y_predict)))

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


