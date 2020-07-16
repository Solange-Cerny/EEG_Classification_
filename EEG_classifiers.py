
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



def run_svm(training_path, test_size, random_seed, shuffle, clf_output_file):
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
    start = time.time() # Performance
    data = pd.read_csv(training_path)
    end = time.time() # Performance
    print('SVM file loaded in: ' + str(end - start)) # Performance

    # Feature selection
    # Note: feature selection is based on the entire dataset
    start = time.time() # Performance
    selected_features = fs.run_select_percentile(training_path)
    # Create new dataset containing only selected features
    feature_names_plus_label = selected_features.copy()
    feature_names_plus_label.append("Label")
    selected_data = data[feature_names_plus_label].copy()
    end = time.time() # Performance
    print('SVM feature selection in: ' + str(end - start)) # Performance


    # debug - to check what fatures got actually promoted
    #with open('promoted_features.csv', 'w', newline='') as data_file:
    #    writer = csv.writer(data_file)
    #    writer.writerow(selected_features)



    # Split dataset into train and test (x is data, y is labels)
    start = time.time() # Performance
    x_train, x_test, y_train, y_test = split_dataset(selected_data, test_size, random_seed, shuffle)
    end = time.time() # Performance
    print('SVM train/test split in: ' + str(end - start)) # Performance

    # Train Support Vector Machine classifier
    start = time.time() # Performance
    clf = svm.SVC()
    clf = clf.fit(x_train, y_train)
    # Save the model
    joblib.dump(clf, clf_output_file)
    end = time.time() # Performance
    print('SVM train & save model in: ' + str(end - start)) # Performance

    # Predict on the testing data
    start = time.time() # Performance
    y_predict = clf.predict(x_test)
    end = time.time() # Performance
    print('SVM predict in: ' + str(end - start)) # Performance

    start = time.time() # Performance
    svm_acc = accuracy_score(y_test, y_predict)
    svm_mcc = matthews_corrcoef(y_test, y_predict)
    svm_auc = roc_auc_score(y_test, y_predict)
    end = time.time() # Performance
    print('SVM calculate metrics: ' + str(end - start)) # Performance
    
    print("SVM classifier:")
    print("acc: " + str(svm_acc))
    print("mcc: " + str(svm_mcc))
    print("auc: " + str(svm_auc))

    return svm_acc, svm_mcc, svm_auc


def run_knn(training_path, test_size, random_seed, shuffle, clf_output_file):
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
    start = time.time() # Performance
    data = pd.read_csv(training_path)
    end = time.time() # Performance
    print('KNN file loaded in: ' + str(end - start)) # Performance

    # Feature selection
    # Note: feature selection is based on the entire dataset
    start = time.time() # Performance
    selected_features = fs.run_select_percentile(training_path)
    # Create new dataset containing only selected features
    feature_names_plus_label = selected_features.copy()
    feature_names_plus_label.append("Label")
    selected_data = data[feature_names_plus_label].copy()
    end = time.time() # Performance
    print('KNN feature selection in: ' + str(end - start)) # Performance

    # Split dataset into train and test (x is data, y is labels)
    start = time.time() # Performance
    x_train, x_test, y_train, y_test = split_dataset(selected_data, test_size, random_seed, shuffle)
    end = time.time() # Performance
    print('KNN train/test split in: ' + str(end - start)) # Performance

    # Train K Nearest Neighbour classifier
    start = time.time() # Performance
    clf = NearestCentroid()
    clf = clf.fit(x_train, y_train)
    # Save the model
    joblib.dump(clf, clf_output_file)
    end = time.time() # Performance
    print('KNN train & save model in: ' + str(end - start)) # Performance

    # Predict on the testing data
    start = time.time() # Performance
    y_predict = clf.predict(x_test)
    end = time.time() # Performance
    print('KNN predict in: ' + str(end - start)) # Performance

    start = time.time() # Performance
    knn_acc = accuracy_score(y_test, y_predict)
    knn_mcc = matthews_corrcoef(y_test, y_predict)
    knn_auc = roc_auc_score(y_test, y_predict)
    end = time.time() # Performance
    print('KNN calculate metrics: ' + str(end - start)) # Performance
    
    print("KNN classifier:")
    print("acc: " + str(knn_acc))
    print("mcc: " + str(knn_mcc))
    print("auc: " + str(knn_auc))

    return knn_acc, knn_mcc, knn_auc


def run_random_forest(training_path, test_size, random_seed, shuffle, clf_output_file):
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
    start = time.time() # Performance
    data = pd.read_csv(training_path)
    end = time.time() # Performance
    print('Random Forest file loaded in: ' + str(end - start)) # Performance

    # Feature selection
    start = time.time() # Performance
    # Note: feature selection is based on the entire dataset
    selected_features = fs.run_select_percentile(training_path)
    # Create new dataset containing only selected features
    feature_names_plus_label = selected_features.copy()
    feature_names_plus_label.append("Label")
    selected_data = data[feature_names_plus_label].copy()
    end = time.time() # Performance
    print('Random Forest feature selection in: ' + str(end - start)) # Performance

    # Split dataset into train and test (x is data, y is labels)
    start = time.time() # Performance
    x_train, x_test, y_train, y_test = split_dataset(selected_data, test_size, random_seed, shuffle)
    end = time.time() # Performance
    print('Random Forest train/test split in: ' + str(end - start)) # Performance

    # Train Random Forest classifier
    start = time.time() # Performance
    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(x_train, y_train)
    # Save the model
    joblib.dump(clf, clf_output_file)
    end = time.time() # Performance
    print('Random Forest train & save model in: ' + str(end - start)) # Performance

    # Predict on the testing data
    start = time.time() # Performance
    y_predict = clf.predict(x_test)
    end = time.time() # Performance
    print('Random Forest predict in: ' + str(end - start)) # Performance

    start = time.time() # Performance
    randf_acc = accuracy_score(y_test, y_predict)
    randf_mcc = matthews_corrcoef(y_test, y_predict)
    randf_auc = roc_auc_score(y_test, y_predict)
    end = time.time() # Performance
    print('Random Forest calculate metrics: ' + str(end - start)) # Performance
    
    print("Random Forest classifier:")
    print("acc: " + str(randf_acc))
    print("mcc: " + str(randf_mcc))
    print("auc: " + str(randf_auc))

    return randf_acc, randf_mcc, randf_auc


def run_ada_boost(training_path, test_size, random_seed, shuffle, clf_output_file):
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
    start = time.time() # Performance
    data = pd.read_csv(training_path)
    end = time.time() # Performance
    print('ADA B file loaded in: ' + str(end - start)) # Performance

    # Feature selection
    # Note: feature selection is based on the entire dataset
    start = time.time() # Performance
    selected_features = fs.run_select_percentile(training_path)
    # Create new dataset containing only selected features
    feature_names_plus_label = selected_features.copy()
    feature_names_plus_label.append("Label")
    selected_data = data[feature_names_plus_label].copy()
    end = time.time() # Performance
    print('ADA B feature selection in: ' + str(end - start)) # Performance

    # Split dataset into train and test (x is data, y is labels)
    start = time.time() # Performance
    x_train, x_test, y_train, y_test = split_dataset(selected_data, test_size, random_seed, shuffle)
    end = time.time() # Performance
    print('ADA B train/test split in: ' + str(end - start)) # Performance

    # Train Random Forest classifier
    start = time.time() # Performance
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf = clf.fit(x_train, y_train)
    # Save the model
    joblib.dump(clf, clf_output_file)
    end = time.time() # Performance
    print('ADA B train & save model in: ' + str(end - start)) # Performance

    # Predict on the testing data
    start = time.time() # Performance
    y_predict = clf.predict(x_test)
    end = time.time() # Performance
    print('ADA B predict in: ' + str(end - start)) # Performance

    start = time.time() # Performance
    adab_acc = accuracy_score(y_test, y_predict)
    adab_mcc = matthews_corrcoef(y_test, y_predict)
    adab_auc = roc_auc_score(y_test, y_predict)
    end = time.time() # Performance
    print('ADA B calculate metrics: ' + str(end - start)) # Performance
    
    print("ADA B classifier:")
    print("acc: " + str(adab_acc))
    print("mcc: " + str(adab_mcc))
    print("auc: " + str(adab_auc))

    return adab_acc, adab_mcc, adab_auc


def run_mlp(training_path, test_size, random_seed, shuffle, clf_output_file):
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
    start = time.time() # Performance
    data = pd.read_csv(training_path)
    end = time.time() # Performance
    print('MLP file loaded in: ' + str(end - start)) # Performance

    # Feature selection
    # Note: feature selection is based on the entire dataset
    start = time.time() # Performance
    selected_features = fs.run_select_percentile(training_path)
    # Create new dataset containing only selected features
    feature_names_plus_label = selected_features.copy()
    feature_names_plus_label.append("Label")
    selected_data = data[feature_names_plus_label].copy()
    end = time.time() # Performance
    print('MLP feature selection in: ' + str(end - start)) # Performance

    # Split dataset into train and test (x is data, y is labels)
    start = time.time() # Performance
    x_train, x_test, y_train, y_test = split_dataset(selected_data, test_size, random_seed, shuffle)
    end = time.time() # Performance
    print('MLP train/test split in: ' + str(end - start)) # Performance

    # Train Multi Layer Perceptron classifier
    start = time.time() # Performance
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf = clf.fit(x_train, y_train)
    # Save the model
    joblib.dump(clf, clf_output_file)
    end = time.time() # Performance
    print('MLP train & save model in: ' + str(end - start)) # Performance

    # Predict on the testing data
    start = time.time() # Performance
    y_predict = clf.predict(x_test)
    end = time.time() # Performance
    print('MLP predict in: ' + str(end - start)) # Performance

    start = time.time() # Performance
    mlp_acc = accuracy_score(y_test, y_predict)
    mlp_mcc = matthews_corrcoef(y_test, y_predict)
    mlp_auc = roc_auc_score(y_test, y_predict)
    end = time.time() # Performance
    print('MLP calculate metrics: ' + str(end - start)) # Performance
    
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


