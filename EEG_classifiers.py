
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
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
import csv
import time



def run_svm(clf_output_file, x_train, x_test, y_train, y_test):
    """
    Builds and saves a trained support vector machine classifier.
    :param training_path: String
        File path for the training matrix.
    :param test_size: float
        Proportion of data to use for testing.
    :param clf_output_file: String
        Name of file to save the classifier to.
    """

    # Train Support Vector Machine classifier
    start = time.time() # Performance
    clf = svm.SVC()
    clf = clf.fit(x_train, y_train)
    # Save the model
    joblib.dump(clf, clf_output_file)
    end = time.time() # Performance
    print('SVM train & save model in: ' + str(end - start)) # Performance

    # Predict on the testing data
    y_predict = clf.predict(x_test)

    # Performance measurements
    svm_acc = accuracy_score(y_test, y_predict)
    svm_mcc = matthews_corrcoef(y_test, y_predict)
    svm_auc = roc_auc_score(y_test, y_predict)
    
    print("SVM classifier:")
    print("acc: " + str(svm_acc))
    print("mcc: " + str(svm_mcc))
    print("auc: " + str(svm_auc))

    return svm_acc, svm_mcc, svm_auc


def run_knn(clf_output_file, x_train, x_test, y_train, y_test):
    """
    Builds and saves a trained K nearest neighbour classifier.
    :param training_path: String
        File path for the training matrix.
    :param test_size: float
        Proportion of data to use for testing.
    :param clf_output_file: String
        Name of file to save the classifier to.
    """

    # Train K Nearest Neighbour classifier
    start = time.time() # Performance
    clf = NearestCentroid()
    clf = clf.fit(x_train, y_train)
    # Save the model
    joblib.dump(clf, clf_output_file)
    end = time.time() # Performance
    print('KNN train & save model in: ' + str(end - start)) # Performance

    # Predict on the testing data
    y_predict = clf.predict(x_test)

    # Performance measurements
    knn_acc = accuracy_score(y_test, y_predict)
    knn_mcc = matthews_corrcoef(y_test, y_predict)
    knn_auc = roc_auc_score(y_test, y_predict)
    
    print("KNN classifier:")
    print("acc: " + str(knn_acc))
    print("mcc: " + str(knn_mcc))
    print("auc: " + str(knn_auc))

    return knn_acc, knn_mcc, knn_auc


def run_random_forest(clf_output_file, x_train, x_test, y_train, y_test):
    """
    Builds and saves a trained random forest classifier.
    :param training_path: String
        File path for the training matrix.
    :param test_size: float
        Proportion of data to use for testing.
    :param clf_output_file: String
        Name of file to save the classifier to.
    """

    # Train Random Forest classifier
    start = time.time() # Performance
    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(x_train, y_train)
    # Save the model
    joblib.dump(clf, clf_output_file)
    end = time.time() # Performance
    print('Random Forest train & save model in: ' + str(end - start)) # Performance

    # Predict on the testing data
    y_predict = clf.predict(x_test)

    # Performance measurements
    randf_acc = accuracy_score(y_test, y_predict)
    randf_mcc = matthews_corrcoef(y_test, y_predict)
    randf_auc = roc_auc_score(y_test, y_predict)
    
    print("Random Forest classifier:")
    print("acc: " + str(randf_acc))
    print("mcc: " + str(randf_mcc))
    print("auc: " + str(randf_auc))

    return randf_acc, randf_mcc, randf_auc


def run_ada_boost(clf_output_file, x_train, x_test, y_train, y_test):
    """
    Builds and saves a trained AdaBoost classifier.
    :param training_path: String
        File path for the training matrix.
    :param test_size: float
        Proportion of data to use for testing.
    :param clf_output_file: String
        Name of file to save the classifier to.
    """

    # Train Random Forest classifier
    start = time.time() # Performance
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf = clf.fit(x_train, y_train)
    # Save the model
    joblib.dump(clf, clf_output_file)
    end = time.time() # Performance
    print('ADA B train & save model in: ' + str(end - start)) # Performance

    # Predict on the testing data
    y_predict = clf.predict(x_test)

    # Performance measurements
    adab_acc = accuracy_score(y_test, y_predict)
    adab_mcc = matthews_corrcoef(y_test, y_predict)
    adab_auc = roc_auc_score(y_test, y_predict)
    
    print("ADA B classifier:")
    print("acc: " + str(adab_acc))
    print("mcc: " + str(adab_mcc))
    print("auc: " + str(adab_auc))

    return adab_acc, adab_mcc, adab_auc


def run_mlp(clf_output_file, x_train, x_test, y_train, y_test):
    """
    Builds and saves a trained multi layer perceptron classifier.
    :param training_path: String
        File path for the training matrix.
    :param test_size: float
        Proportion of data to use for testing.
    :param clf_output_file: String
        Name of file to save the classifier to.
    """

    # Train Multi Layer Perceptron classifier
    start = time.time() # Performance
    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf = MLPClassifier()
    clf = clf.fit(x_train, y_train)
    # Save the model
    joblib.dump(clf, clf_output_file)
    end = time.time() # Performance
    print('MLP train & save model in: ' + str(end - start)) # Performance

    # Predict on the testing data
    y_predict = clf.predict(x_test)

    # Performance measurements
    mlp_acc = accuracy_score(y_test, y_predict)
    mlp_mcc = matthews_corrcoef(y_test, y_predict)
    mlp_auc = roc_auc_score(y_test, y_predict)
    
    print("MLP classifier:")
    print("acc: " + str(mlp_acc))
    print("mcc: " + str(mlp_mcc))
    print("auc: " + str(mlp_auc))

    return mlp_acc, mlp_mcc, mlp_auc


def split_dataset(dataset, test_size, random_seed, shuffle):
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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_seed, shuffle=shuffle)

    return x_train, x_test, y_train, y_test

def load_data_one_file_split(training_path, test_size, random_seed, shuffle):
    # Load dataset into pandas.DataFrame
    data = pd.read_csv(training_path)

    # Feature selection
    # Note: feature selection is based on the entire dataset
    selected_features = fs.run_select_percentile(training_path)
    # Create new dataset containing only selected features
    feature_names_plus_label = selected_features.copy()
    feature_names_plus_label.append("Label")
    selected_data = data[feature_names_plus_label].copy()


    # debug - to check what fatures got actually promoted
    #with open('promoted_features.csv', 'w', newline='') as data_file:
    #    writer = csv.writer(data_file)
    #    writer.writerow(selected_features)



    # Split dataset into train and test (x is data, y is labels)
    x_train, x_test, y_train, y_test = split_dataset(selected_data, test_size, random_seed, shuffle)


    # Confirming MLP bug in acc
    #print(type(y_test.values))
    #print(y_test.values)

    #with open('aaa.csv', 'w', newline='') as data_file:
    #    writer = csv.writer(data_file)
    #    writer.writerow(y_test.values)

    return x_train, x_test, y_train, y_test

def load_data_two_files(training_path, test_path):
    # Load dataset into pandas.DataFrame
    train_data = pd.read_csv(training_path)
    test_data = pd.read_csv(test_path)

    # Feature selection
    # Note: feature selection is based on the entire dataset
    selected_features = fs.run_select_percentile(training_path)
    # Create new train dataset containing only selected features
    feature_names_plus_label = selected_features.copy()
    feature_names_plus_label.append("Label")
    selected_train_data = train_data[feature_names_plus_label].copy()

    # Persist selected features for future statistical analysis
    with open(training_path + 'promoted_features.csv', 'w', newline='') as data_file:
        writer = csv.writer(data_file)
        writer.writerow(feature_names_plus_label)

    x_train = selected_train_data.iloc[:, 0:-2]
    y_train = selected_train_data.iloc[:, -1]

    # Create new test dataset containing only selected features
    selected_test_data = test_data[feature_names_plus_label].copy()
    x_test = selected_test_data.iloc[:, 0:-2]
    y_test = selected_test_data.iloc[:, -1]

    return x_train, x_test, y_train, y_test
