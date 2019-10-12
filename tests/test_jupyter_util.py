import pytest
import logging
import pandas as pd
import util.jupyter_util as ju
import os

log = logging.getLogger(__name__)

DATAFILE = "/tmp/test.csv"
LABEL_COL = "p1_winner"

def setup_function(function):
    """
    setup function for test_get_data
    we are going to create small dataset csv to make the test run fast
    :param function:
    :return:
    """
    if function.__name__ == "test_get_tourney_data":
        d = {LABEL_COL: [0, 1, 1, 0, 1, 0, 1, 1],
             "tourney_year": [1980, 1998, 2000, 2018, 2019, 2019, 2019, 2019],
             "tourney_id": [1, 2, 3, 4, 3, 3, 1, 2]
             }
    else:
        d = {LABEL_COL: [0, 1, 1, 0, 1, 0, 1, 1],
             "tourney_year": [1980, 1998, 2000, 2018, 2019, 2019, 2019, 2019],
             "tourney_id_1": [1, 0, 0, 0, 0, 0, 1, 0],
             "tourney_id_2": [0, 1, 0, 0, 0, 0, 0, 1],
             "tourney_id_3": [0, 0, 1, 0, 1, 1, 0, 0],
             "tourney_id_4": [0, 0, 0, 4, 0, 0, 0, 0]
             }
    df = pd.DataFrame(d)
    df.to_csv(DATAFILE, index=False)

def teardown_function(function):
    """
    delete datafile that we created during setup
    :param function:
    :return:
    """
    os.remove(DATAFILE)


def test_get_data():
    """
    load data from file and test to see if the date range query for our dataset is running correctly
    :return:
    """
    start_year = 1998
    end_year = 2018

    X_train, X_test, y_train, y_test = ju.get_data(DATAFILE, LABEL_COL, start_year, end_year)
    df = X_train.append(X_test, ignore_index=True)
    assert len(df) == 3, "number of rows returned not correct"
    assert df["tourney_year"].min() == start_year, "start date not working"
    assert df["tourney_year"].max() == end_year, "end date not working"


def test_get_tourney_data():
    """
    Test to make sure we are only getting data for particular tournament and year
    :return:
    """
    tourney_id = 3
    tourney_year = 2019
    features, labels = ju.get_tourney_data(DATAFILE, LABEL_COL, tourney_id, tourney_year)
    assert len(features) == 2, "number of rows mismatched"

def test_get_tourney_data_ohe():
    """
    Now test to see if our function behaves properly of tourney_id is OHE'd

    The test is the same as above but the dataset created in setup_function is slightly different
    :return:
    """
    tourney_id = 3
    tourney_year = 2019
    features, labels = ju.get_tourney_data(DATAFILE, LABEL_COL, tourney_id, tourney_year, ohe=True)
    assert len(features) == 2, "number of rows mismatched"
