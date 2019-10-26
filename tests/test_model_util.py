import pytest
import logging
import util.model_util as mu
from util.model_util import ModelWrapper
from util.model_util import BaseDiffFilter, BaseRawFilter, OHEFilter, RAW_COLUMNS, DIFF_COLUMNS
from sklearn.tree import DecisionTreeClassifier
from util import jupyter_util as ju
import os
import re
import pandas as pd
import numpy as np



log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_DIR = "../models"
DESCRIPTION = "diff-ohe-history_test-matchup_test"
START_YEAR = 1980
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
    X_train, X_test, y_train, y_test = ju.get_data(feature_file, "p1_winner", START_YEAR, END_YEAR)
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
    assert os.path.exists(mw.report_file), f"{mw.report_file} does not exist"
    log.info(f"loading report file: {mw.report_file}")
    report = pd.read_csv(mw.report_file)
    assert len(report) == 1, "report should have 1 row"

    log.info(f'report type: {type(report)}')
    log.info(report.head())
    log.info(f'Selected row: {report[report.description == DESCRIPTION]}')
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


def filter_data(data):
    data = data.drop(["tourney_level_label"], axis=1)
    return data



@pytest.fixture()
def data_filter():
    return filter_data

def test_save_load_model_with_filter(datadir, data_filter):
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
    X_train, X_test, y_train, y_test = ju.get_data(feature_file, "p1_winner", START_YEAR, END_YEAR, data_filter= data_filter)
    mw = ModelWrapper(DecisionTreeClassifier(random_state=1),
                      description=DESCRIPTION,
                      data_file=feature_file,
                      start_year=START_YEAR,
                      end_year=END_YEAR,
                      X_train=X_train,
                      y_train=y_train,
                      X_test=X_test,
                      y_test=y_test,
                      data_filter=data_filter).fit()
    y_predict_dt = mw.predict()
    log.debug(f'y_predict_dt {y_predict_dt}')
    mw.analyze()

    accuracy = mw.accuracy
    roc_auc_score = mw.roc_auc_score

    mw.save()

    # load our model back
    report = pd.read_csv(ModelWrapper.REPORT_FILE)
    assert len(report) == 1, "report should have 1 row"


    loaded_mw = ModelWrapper.get_model_wrapper_from_report(report[report.description == DESCRIPTION], load_data = True)
    # loaded_mw.X_train = X_train
    # loaded_mw.y_train = y_train
    # loaded_mw.X_test = X_test
    # loaded_mw.y_test = y_test

    y_predict_loaded = loaded_mw.predict()
    log.debug(f'y_predict_loaded {y_predict_loaded}')

    assert loaded_mw.accuracy == accuracy, "accuracy does not match"
    assert loaded_mw.roc_auc_score == roc_auc_score, "roc/auc score does not match"
    assert (y_predict_dt == y_predict_loaded).all(), "predictions don't match"


class TestColumnFilters(object):

    @pytest.fixture()
    def data_for_filters(self, datadir):
        """
        get our data file
        :param self:
        :param datadir:
        :return:
        """
        return pd.read_csv(f'{datadir}/column_features_data.csv')

    def compare_columns(self, left, right):

        assert len(left)  == len(right), "list length should be the same"

        # now do 2 way element wise comparison
        diff_col = [col for col in left if col not in right]
        assert len(diff_col) == 0, "there are extra columns in the generated list"
        diff_col = [col for col in right if col not in left]
        assert len(diff_col) == 0, "there are missing columns in the generated list"

    def test_base_raw_filter(self, data_for_filters):
        filter = BaseRawFilter(data_for_filters)
        data = filter.get_data()
        diff_col = [col for col in data.columns if col not in RAW_COLUMNS ]
        assert len(diff_col) == 0, "there are extra columns in the generated list"
        diff_col = [col for col in RAW_COLUMNS if col not in data.columns]
        assert len(diff_col) == 0, "there are missing columns in the generated list"

    def test_base_diff_filter(self, data_for_filters):
        filter = BaseDiffFilter(data_for_filters)
        data = filter.get_data()
        diff_col = [col for col in data.columns if col not in DIFF_COLUMNS ]
        assert len(diff_col) == 0, "there are extra columns in the generated list"
        diff_col = [col for col in DIFF_COLUMNS if col not in data.columns]
        assert len(diff_col) == 0, "there are missing columns in the generated list"

    def test_ohe_filter(self, data_for_filters):
        filter = OHEFilter(data_for_filters)
        columns = filter.get_data().columns

        assert len(columns) > 4000, "not enough columns"
        assert len([col for col in columns if re.search(r'(p1|p2)_[\d]+', col)]) > 0, "player id's are missing"
        assert len([col for col in columns if re.search(r'(p1|p2)_ioc_[\w]+', col)]) > 0, "player origins are missing"
        assert len([col for col in columns if re.search(r'(p1|p2)_hand_[\w]+', col)]) > 0, "player hand are missing"
        assert len([col for col in columns if re.search(r'tourney_id_', col)]) > 0, "tourney id's are missing"
        assert len([col for col in columns if re.search(r'best_of_', col)]) > 0, "tourney id's are missing"
        assert len([col for col in columns if re.search(r'surface_', col)]) > 0, "tourney id's are missing"


    def test_stats_diff_filter(self, data_for_filters):
        filter = mu.StatsDiffFilter(data_for_filters)
        data = filter.get_data()

        cols = ['p1_stats_1stin_avg_diff',
                 'p1_stats_1stwon_avg_diff',
                 'p1_stats_2ndwon_avg_diff',
                 'p1_stats_ace_avg_diff',
                 'p1_stats_bpfaced_avg_diff',
                 'p1_stats_bpsaved_avg_diff',
                 'p1_stats_df_avg_diff',
                 'p1_stats_svgms_avg_diff',
                 'p1_stats_svpt_avg_diff']

        self.compare_columns(data.columns, cols)

    def test_stats_raw_filter(self, data_for_filters):

        target = ['p1_stats_1stin_avg',
                             'p1_stats_1stwon_avg',
                             'p1_stats_2ndwon_avg',
                             'p1_stats_ace_avg',
                             'p1_stats_bpfaced_avg',
                             'p1_stats_bpsaved_avg',
                             'p1_stats_df_avg',
                             'p1_stats_svgms_avg',
                             'p1_stats_svpt_avg',
                             'p2_stats_1stin_avg',
                             'p2_stats_1stwon_avg',
                             'p2_stats_2ndwon_avg',
                             'p2_stats_ace_avg',
                             'p2_stats_bpfaced_avg',
                             'p2_stats_bpsaved_avg',
                             'p2_stats_df_avg',
                             'p2_stats_svgms_avg',
                             'p2_stats_svpt_avg']

        filter = mu.StatsRawFilter(data_for_filters)
        cols = filter.get_columns()

        self.compare_columns(cols, target)

    def test_stats5_diff_filter(self, data_for_filters):

        filter = mu.Stats5DiffFilter(data_for_filters)
        cols = filter.get_columns()

        assert len([col for col in cols if re.search("percentage", col)]) == 0, "columns should not contain percentage"
        assert len([col for col in cols if re.search("stats5", col)]) == len(cols), \
            "all columns should have stats5 in the name"

    def test_history5_diff_filter(self, data_for_filters):

        filter = mu.History5PercentageDiffFilter(data_for_filters)
        cols = filter.get_columns()

        assert len(cols) == 3, " there should be 3 columns"
        assert len([col for col in cols if not re.search(r"percentage.+diff", col)]) == 0, "columns should contain percentage"
        assert len([col for col in cols if re.search("history5", col)]) == len(cols), \
            "all columns should have stats5 in the name"

    def test_matchup5_diff_filter(self, data_for_filters):

        filter = mu.Matchup5PercentageDiffFilter(data_for_filters)
        cols = filter.get_columns()

        assert len(cols) == 3, " there should be 3 columns"
        assert len([col for col in cols if not re.search(r"percentage.+diff", col)]) == 0, "columns should contain percentage"
        assert len([col for col in cols if re.search("matchup5", col)]) == len(cols), \
            "all columns should have stats5 in the name"


    def test_default_column_filter(self, data_for_filters):
        default_filter = mu.DefaultColumnFilter(data_for_filters)
        default_cols = default_filter.get_columns()

        diff_filter = mu.BaseDiffFilter(data_for_filters)
        diff_cols = diff_filter.get_columns()

        ohe_filter = mu.OHEFilter(data_for_filters)
        ohe_cols = ohe_filter.get_columns()

        assert len(default_cols) == len(diff_cols) + len(ohe_cols), "default columns should be sum of diff and ohe"

    def test_base_rank_diff_filter(self, data_for_filters):
        filter = mu.BaseRankDiffFilter(data_for_filters)
        cols = filter.get_columns()

        assert len(cols) == 1, "should return only 1 column"
        assert "seed_diff" in cols, "should have seed_diff"


class TestWeightCalculators(object):

    def test_interval_based_weight_calculator(self):
        year_df = pd.DataFrame({"tourney_year": [1998, 2005, 2004, 2013, 2018] })
        weight_calculator = mu.IntervalBasedWeightCalculator(year_df, "tourney_year", 1)
        weights = weight_calculator.get_weights()

        assert weights[0] == 1, f"weight for {year_df.tourney_year[0]} is incorrect"
        assert weights[1] == 8, f"weight for {year_df.tourney_year[1]} is incorrect"
        assert weights[2] == 7, f"weight for {year_df.tourney_year[2]} is incorrect"
        assert weights[3] == 16, f"weight for {year_df.tourney_year[3]} is incorrect"
        assert weights[4] == 21, f"weight for {year_df.tourney_year[4]} is incorrect"

    def test_bin_based_weight_calculator(self):
        year_df = pd.DataFrame({"tourney_year": [2014, 2016, 2015, 2017, 2018] })
        weight_calculator = mu.BinBasedWeightCalculator(year_df, "tourney_year", 5)
        weights = weight_calculator.get_weights()

        assert weights[0] == 1, f"weight for {year_df.tourney_year[0]} is incorrect"
        assert weights[1] == 3, f"weight for {year_df.tourney_year[1]} is incorrect"
        assert weights[2] == 2, f"weight for {year_df.tourney_year[2]} is incorrect"
        assert weights[3] == 4, f"weight for {year_df.tourney_year[3]} is incorrect"
        assert weights[4] == 5, f"weight for {year_df.tourney_year[4]} is incorrect"
