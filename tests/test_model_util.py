import pytest
import logging
from util.model_util import ModelWrapper
from sklearn.tree import DecisionTreeClassifier
from util import jupyter_util as ju
import os
import pandas as pd
import numpy as np



log = logging.getLogger(__name__)

MODEL_DIR = "../models"
DESCRIPTION = "diff-ohe-history_test-matchup_test"
START_YEAR = 1998
END_YEAR = 2018
MODEL_TEMPLATE_NAME = f'{START_YEAR}-{END_YEAR}-{DESCRIPTION}.pkl'
MODEL_NAME = "modelname"



def test_get_model_file():
    """

    :return:
    """

    model_file_format = MODEL_TEMPLATE_NAME
    ModelWrapper.model_file_format = model_file_format
    model_name = MODEL_NAME
    filename = ModelWrapper._get_model_filename(model_name)
    assert filename == f"{ModelWrapper.MODEL_DIR}/{model_name}-{model_file_format}", "filename mismatch"


def test_get_info_from_model_filename():
    """
    Test to make sure we can parse the model dir and the description correctly
    :return:
    """

    model_dir, description = ModelWrapper._get_info_from_model_filename(f'{MODEL_DIR}/{MODEL_NAME}-{MODEL_TEMPLATE_NAME}')
    assert model_dir == MODEL_DIR, "mismatched model_dir"
    assert description == DESCRIPTION, "mismatched description"


def test_save_load_model(datadir):
    """
    Create a model and train then save the model and load it back to make sure we are getting the same thing
    as before

    :param datadir: directory where files are
    :return:
    """
    # save our files to our test directory
    ModelWrapper.REPORT_FILE = f'{datadir}/summary.csv'
    ModelWrapper.MODEL_DIR = f'{datadir}'

    if os.path.exists(ModelWrapper.REPORT_FILE):
        os.remove(ModelWrapper.REPORT_FILE)

    feature_file = f'{datadir}/features.csv'
    X_train, X_test, y_train, y_test = ju.get_data(feature_file, "p1_winner", 1980, 2019)
    mw = ModelWrapper(DecisionTreeClassifier(random_state=1),
                      description=DESCRIPTION,
                      data_file=feature_file,
                      start_year=START_YEAR,
                      end_year=END_YEAR,
                      X_train=X_train,
                      y_train=y_train,
                      X_test=X_test,
                      y_test=y_test).fit()
    y_predict_dt = mw.predict()
    log.debug(f'y_predict_dt {y_predict_dt}')
    mw.analyze()

    accuracy = mw.accuracy
    roc_auc_score = mw.roc_auc_score

    mw.save()

    # load our model back
    report = pd.read_csv(ModelWrapper.REPORT_FILE)
    assert len(report) == 1, "report should have 1 row"


    loaded_mw = ModelWrapper.get_model_wrapper_from_report(report[report.description == DESCRIPTION])
    loaded_mw.X_train = X_train
    loaded_mw.y_train = y_train
    loaded_mw.X_test = X_test
    loaded_mw.y_test = y_test

    y_predict_loaded = loaded_mw.predict()
    log.debug(f'y_predict_loaded {y_predict_loaded}')

    assert loaded_mw.accuracy == accuracy, "accuracy does not match"
    assert loaded_mw.roc_auc_score == roc_auc_score, "roc/auc score does not match"
    assert (y_predict_dt == y_predict_loaded).all(), "predictions don't match"

