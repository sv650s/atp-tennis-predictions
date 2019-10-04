
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



def get_data(filename: str, label_col: str, start_year: int, random_state = 1) -> (pd.DataFrame, pd.DataFrame):
    """
    Gets the data file, filters out unwanted entries
    :param filename: filename to load
    :param lable_col: name of label column
    :param start_year: filter out entries before this year
    :return: X_train, X_test, y_train, y_test
    """
    log.info(f"loading {filename}")

    features = pd.read_csv(filename)
    features = features[features.tourney_year >= start_year]
    labels = features[label_col].copy()
    features = features.drop([label_col], axis=1)
    print(features.shape)
    return train_test_split(features, labels, random_state=random_state)


def save_model(model, file_template_name: str):
    pickle.dump(model, open(f'../models/{type(model).__name__.lower()}-{file_template_name}', 'wb'))





