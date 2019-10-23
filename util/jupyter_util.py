
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pickle
import logging

log = logging.getLogger(__name__)

import seaborn as sns
import matplotlib.pyplot as plt




def plot_2d(X_test: pd.DataFrame, y_predict):
    """
    Use PCA to dimensionality reduction then plot win vs loses on a 2d graph
    :param: X_test - test features
    :param: y_predict - predictions from model
    """

    # normalize our data first before using PCA so weights are the same for all variables
    mms = MinMaxScaler()
    normalized_df = X_test.copy()
    for col in normalized_df.columns:
        normalized_col = mms.fit_transform([normalized_df[col].values])
        normalized_df[col] = normalized_col[-1]


    # reduce X to 2D
    X_test_2d = pd.DataFrame(PCA(n_components=2).fit_transform(X_test))

    # let's figure out which ones of these are predicted Wins
    wins = X_test_2d[y_predict == 1]

    # entries that are predicted losses
    losses = X_test_2d[y_predict == 0]

    f, a = plt.subplots(1, 1, figsize=(20,5))
    p = sns.scatterplot(x=0, y=1, data=losses, ax=a, color='r', alpha=0.25)
    p = sns.scatterplot(x=0, y=1, data=wins, ax=a, color='b', alpha=0.25)



def get_data(filename: str, label_col: str, start_year: int, end_year: int, random_state = 1, data_filter = None) -> (pd.DataFrame, pd.DataFrame):
    """
    Gets the data file, filters out unwanted entries and then split into training and test sets

    :param filename: filename to load
    :param label_col: name of label column
    :param start_year: filter out entries before this year
    :param end_year: filter out entries after this year
    :param random_state: seeds the random state for splitting train and test data
    :param data_filter: function to filter out feature columns
    :return: X_train, X_test, y_train, y_test
    """
    log.info(f"loading {filename}")

    features = pd.read_csv(filename)
    features = features[(features.tourney_year >= start_year) & (features.tourney_year <= end_year)]
    labels = features[label_col].copy()
    features = features.drop([label_col], axis=1)
    if data_filter:
        log.info(f'Shape before filtering: {features.shape}')
        features = data_filter(features)
        log.info(f'Shape after filtering: {features.shape}')
    log.info(f'Features shape: {features.shape}')
    return train_test_split(features, labels, random_state=random_state)


def get_tourney_data(filename: str, label_col: str, tourney_id: int, tourney_year: int, feature_filter = None, ohe = True) -> (pd.DataFrame, pd.DataFrame):
    """
    Gets samples for a particular tournament for a particular year.

    We are going to use this to test predictions for matches of a particular tournament

    :param filename: data file to load
    :param label_col: name of label column
    :param tourney_id: name of tournament
    :param tourney_year: year of tournament
    :param feature_filter: function to filter out features
    :param ohe: indicates whether the features are one hot encoded or not. if it is the function will concat the id with tourney_id to figure out how to get the info. Default is True
    :return: features, labels
    """
    features = pd.read_csv(filename)
    features = features[features.tourney_year == tourney_year]
    if ohe:
        features = features[features[f'tourney_id_{tourney_id}'] == 1]
    else:
        features = features[features.tourney_id == tourney_id]
    log.info(features.shape)

    # make a copy of labels before we drop them
    labels = features[label_col]
    features = features.drop([label_col], axis=1)

    # additional feature filters
    if feature_filter:
        features = feature_filter(features)

    return features, labels




def save_model(model, file_template_name: str):
    pickle.dump(model, open(f'../models/{type(model).__name__.lower()}-{file_template_name}', 'wb'))





