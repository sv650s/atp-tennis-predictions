import pytest
import logging
import pandas as pd
import util.model_util as mu
import util.jupyter_util as ju
import os
from util.model_util import ColumnFilter

log = logging.getLogger(__name__)

# DATAFILE = "/tmp/test.csv"
LABEL_COL = "p1_winner"
REMOVE_COL = "feature_to_remove"

class TestColumnFilter(ColumnFilter):

    def get_columns(self):
        return ["tourney_year", "tourney_id_1", "tourney_id_2", "tourney_id_3", "tourney_id_4"]

@pytest.fixture
def features_file_ohe(datadir):
    return f'{datadir}/features_ohe.csv'

@pytest.fixture
def features_file(datadir):
    return f'{datadir}/features.csv'

class TestJupyterUtil(object):


    def test_get_tourney_data(self, features_file):
        """
        Test to make sure we are only getting data for particular tournament and year
        :return:
        """
        tourney_id = 3
        tourney_year = 2019
        features, labels = ju.get_tourney_data(features_file, LABEL_COL, tourney_id, tourney_year, ohe = False)
        assert len(features) == 2, "number of rows mismatched"



    def test_get_tourney_data_ohe(self, features_file_ohe):
        """
        Now test to see if our function behaves properly of tourney_id is OHE'd

        The test is the same as above but the dataset created in setup_function is slightly different
        :return:
        """
        tourney_id = 3
        tourney_year = 2019
        features, labels = ju.get_tourney_data(features_file_ohe, LABEL_COL, tourney_id, tourney_year)
        assert len(features) == 2, "number of rows mismatched"
        assert len(features.columns) == 6, "number of columns mismatched"


    def feature_filter(self, data: pd.DataFrame):
        """
        function used to test feature_filter
        :param data:
        :return:
        """
        data = data.drop([REMOVE_COL], axis=1)
        return data


    def test_get_tourney_data_ohe_with_filter(self, features_file_ohe):
        """
        Now test to see if our function behaves properly of tourney_id is OHE'd

        The test is the same as above but the dataset created in setup_function is slightly different
        :return:
        """
        tourney_id = 3
        tourney_year = 2019
        features, labels = ju.get_tourney_data(features_file_ohe, LABEL_COL, tourney_id, tourney_year, column_filters=["tests.test_jupyter_util.TestColumnFilter"])
        assert len(features) == 2, "number of rows mismatched"
        assert len(features.columns) == 5, "number of columns mismatched"
