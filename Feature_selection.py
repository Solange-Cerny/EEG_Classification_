"""
## Version history:
2019:
   Original script by Jodie Ashford [ashfojsm], Aston University
"""

import pandas as pd
from sklearn import feature_selection as fs


def feature_selection(file_path):
    """
    Returns a list of selected feature names.
    :param file_path: Path for the training matrix of extracted features to select from
    :return: List of selected feature names
    """

    # Setting up dataset
    data = pd.read_csv(file_path)
    x_train = data.drop('Label', axis=1)
    y_train = data['Label']

    # Feature selection - k = number of features to select
    feature_selector = fs.SelectKBest(fs.f_classif, k=256)
    feature_selector.fit_transform(x_train, y_train)
    mask = feature_selector.get_support()

    # List of all feature names
    feature_names = list(data.columns.values)

    # List of selected feature names
    new_feature_names = []
    for feature_is_selected, feature in zip(mask, feature_names):
        if feature_is_selected:
            new_feature_names.append(feature)
    return new_feature_names

