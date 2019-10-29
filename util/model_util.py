import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import logging
import pickle
import seaborn as sns
import json
import os
import re
import json
from datetime import datetime
import util.class_util as cu
import dill
import numpy as np


log = logging.getLogger(__name__)

REPORT_FILE = '../reports/summary.csv'
MODEL_DIR = '../models'
LABEL_COL = 'p1_winner'
RSTATE = 1
N_JOBS = 4
MAX_ITER = 100


DIFF_COLUMNS = ["draw_size", "round_label", "tourney_level_label", "tourney_month", "tourney_year",
                  "age_diff", "ht_diff", "seed_diff", "rank_diff",
                  ]
RAW_COLUMNS = ["draw_size", "round_label", "tourney_level_label", "tourney_month", "tourney_year", "best_of",
                   "p1", "p1_age", "p1_ht", "p1_rank", "p1_seed", "p1_hand_label",
                   "p2", "p2_age", "p2_ht", "p2_rank", "p2_seed", "p2_hand_label",
                ]

STATS_RAW_COLUMNS = ['p1_stats_1stin_avg',
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


class WeightCalculator(object):

    def __init__(self, features, column):
        self.features = features
        self.column = column

    def calculate_bins(self) -> int:
        raise Exception("Not yet implemented")

    def get_weights(self) -> pd.DataFrame:
        bins = max(1, self.calculate_bins())
        weights = pd.cut(self.features[self.column], bins, labels=np.arange(1, bins+1)).tolist()
        return weights



class BinBasedWeightCalculator(WeightCalculator):
    """
    Splits data into equally sized bins and assign weights based on tourney_year
    The later the year, the higher the weight
    """

    def __init__(self, features, column, bins):
        super().__init__(features, column)
        self.bins = bins

    def calculate_bins(self):
        self.bins = max(1, self.bins)
        log.info(f'bins: {self.bins}')
        return self.bins

class IntervalBasedWeightCalculator(WeightCalculator):
    """
    Splits data into equally sized bins and assign weights based on interval specified
    Larger the column value, the greater the weight will be
    """

    def __init__(self, features, column, years):
        super().__init__(features, column)
        self.years = years

    def calculate_bins(self) -> pd.DataFrame:
        """
        Number of years to create bins from
        :param years:
        :return:
        """
        bins, _ = divmod(self.features[self.column].max() - self.features[self.column].min() + 1, self.years)
        bins = max(1, bins)
        return bins


class ColumnFilter(object):
    """
    Base class for column filtering

    The approach we are taking is each filter will get a subset of feature columns. You can then use pd.concat to put together
    all the columns you need to create the dataset
    """
    def __init__(self, data: pd.DataFrame):
        """
        default constructor
        :param data: feature DF that you want to filter
        """
        self.data = data

    def get_columns(self):
        raise Exception("Not yet implemented")

    def get_data(self):
        filtered_data = self.data[self.get_columns()]
        return filtered_data

class OHEFilter(ColumnFilter):
    """
    gets all one hot encoded columns
    """
    def get_columns(self):
        return [col for col in self.data.columns if
                re.search(r'(p1|p2)_[\d]+', col)
                or re.search(r'(p1|p2)_ioc_.+', col)
                or re.search(r'tourney_id_.+', col)
                or re.search(r'(p1|p2)_hand_[\w]{1}', col)
                or re.search(r'best_of_', col)
                or re.search(r'surface_[\w]+', col)
                ]


class BaseRawFilter(ColumnFilter):
    """
    Filters out base columns for our data
    """
    def get_columns(self):
        return RAW_COLUMNS

class BaseRawNoPlayerFilter(ColumnFilter):
    """
    Filters out base columns for our data
    """
    def get_columns(self):
        return [ col for col in RAW_COLUMNS if not re.search(r"(p1|p2)", col) ]


class BaseDiffFilter(ColumnFilter):
    def get_columns(self):
        return DIFF_COLUMNS




class BaseRankDiffFilter(ColumnFilter):
    """
    returns only seed_diff as a feature - will be using this to establish our baseline for our models
    """
    def get_columns(self):
        return ["seed_diff"]

class DefaultColumnFilter(ColumnFilter):
    """
    Base data filter - our base columns are basic player diff columns + OHE columns
    """

    def get_columns(self):
        diff_filter = BaseDiffFilter(self.data)
        diff_col = diff_filter.get_columns()

        ohe_filter = OHEFilter(self.data)
        ohe_col = ohe_filter.get_columns()

        return diff_col + ohe_col


class StatsRawFilter(ColumnFilter):

    def get_columns(self):
        return [col for col in self.data.columns if  re.search(r"stats_.+avg$", col) and not re.search("percentage", col)]


class StatsDiffFilter(ColumnFilter):
    """
    all time player average stats
    """
    def get_columns(self):
        return [col for col in self.data.columns if  re.search(r"stats_.+diff", col) and not re.search("percentage", col)]

# class Stats5DiffFilter(ColumnFilter):
#     def get_columns(self):


class Stats5DiffFilter(ColumnFilter):
    """
    gets stats5 diff columns
    """
    def get_columns(self):
        return [col for col in self.data.columns if re.search("stats5.*diff", col) and not re.search("percentage", col)]


class History5PercentageDiffFilter(ColumnFilter):
    """
    get history percentage_diff columns for last 5 matches
    """
    def get_columns(self):
        return [col for col in self.data.columns if re.search(r"history5.+percent.+diff", col)]

class History5DiffFilter(ColumnFilter):
    """
    get history percentage_diff columns for last 5 matches
    """
    def get_columns(self):
        return [col for col in self.data.columns if re.search(r"history5.+diff", col)]


class Matchup5PercentageDiffFilter(ColumnFilter):
    """
    get matchup percentage_diff columns for last 5 matches
    """
    def get_columns(self):
        return [col for col in self.data.columns if re.search(r"matchup5.+percent.+diff", col)]



class ModelWrapper(object):

    # default values
    REPORT_FILE = REPORT_FILE
    MODEL_DIR = MODEL_DIR

    # keys
    MODEL_NAME = "model_name"
    MODEL_FILE = "model_file"
    DESCRIPTION = "description"
    DATA_FILE = "data_file"
    START_YEAR = "start_year"
    END_YEAR = "end_year"
    ACCURACY = "accuracy"
    ROC_AUC_SCORE = "roc_auc_score"
    CONFUSION_MATRIX = "confusion_matrix"
    CLASSIFICATION_REPORT = "classification_report"
    FIT_TIME_MIN = "fit_time_min"
    COLUMN_FILTERS = "column_filters"
    PREDICT_TIME_MIN = "predict_time_min"
    TOTAL_TIME_MIN = "total_time_min"

    @staticmethod
    def get_data(filename: str, label_col: str, start_year: int, end_year: int, random_state=1,
                 column_filters=None) -> (pd.DataFrame, pd.DataFrame):
        """
        Gets the data file, filters out unwanted entries and then split into training and test sets

        :param filename: filename to load
        :param label_col: name of label column
        :param start_year: filter out entries before this year
        :param end_year: filter out entries after this year
        :param random_state: seeds the random state for splitting train and test data
        :param data_filters: function to filter out feature columns
        :return: X_train, X_test, y_train, y_test
        """
        if column_filters is not None:
            assert isinstance(column_filters, list), "data_filter must be a list if specified"
            assert len(column_filters) > 0, "data_filter length must be > 0"

        log.info(f"loading {filename}")

        features = pd.read_csv(filename)
        features = features[(features.tourney_year >= start_year) & (features.tourney_year <= end_year)]
        labels = features[label_col].copy()
        features = features.drop([label_col], axis=1)

        # filter out data if specified
        if column_filters is not None:
            log.info(f'Shape before filtering: {features.shape}')
            filter_instances = []
            for filter in column_filters:
                log.info(f'Adding filter: {filter}')
                klass = cu.get_class(filter)
                instance = klass(features)
                filter_instances.append(instance)
            features = pd.concat([instance.get_data() for instance in filter_instances], axis=1)
            log.debug(f'columns: {features.columns}')
            log.info(f'Shape after filtering: {features.shape}')

        log.info(f'Final Features shape: {features.shape}')
        log.debug(f'Final Features columns: {features.columns}')
        return train_test_split(features, labels, random_state=random_state)

    @classmethod
    def get_model_wrapper_from_report(cls, data: pd.DataFrame, load_data: bool = False):
        """
        Gets a Model Wrapper object along with the original model object using a row from the report dataframe

        Report dataframe has the following columns:
            model
            description
            data_file
            start_year
            accuracy
            confusion_matrix
            classification_report
            model_file

        confusion matrix and classification reports will be converted back into their original dictionary representation

        :param data: a row in the report data frame with information to reconstruct the model
        :param load_data: tells us whether to load the original data as we create the model. default False
        :param data_filter: function used to filter columns when we load data. This will only be used if load_data is set to True
        :return: ModelWrapper object
        """
        log.info(type(data))
        log.debug(data)
        assert len(data) == 1, f"data must of length 1 - got {len(data)}"

        # get these from the file name
        model_name = data[ModelWrapper.MODEL_NAME].values[0]
        start_year = int(data[ModelWrapper.START_YEAR].values[0])
        end_year = int(data[ModelWrapper.END_YEAR].values[0])
        description = data[ModelWrapper.DESCRIPTION].values[0]
        data_file = data[ModelWrapper.DATA_FILE].values[0]
        accuracy = float(data[ModelWrapper.ACCURACY].values[0])
        roc_auc_score = float(data[ModelWrapper.ROC_AUC_SCORE].values[0])
        confusion_matrix_str = data[ModelWrapper.CONFUSION_MATRIX].values[0]
        classification_report_str = data[ModelWrapper.CLASSIFICATION_REPORT].values[0]
        model_file = data[ModelWrapper.MODEL_FILE].values[0]
        predict_time_min = int(data[ModelWrapper.PREDICT_TIME_MIN].values[0])
        fit_time_min = int(data[ModelWrapper.FIT_TIME_MIN].values[0])
        if ModelWrapper.COLUMN_FILTERS not in data.columns or data[ModelWrapper.COLUMN_FILTERS].isna().values[0]:
            column_filters = None
        else:
            column_filters = data[ModelWrapper.COLUMN_FILTERS].values[0]


        log.debug(f'model_file {model_file}')


        model_dir, model_file_template = ModelWrapper._get_info_from_model_filename(model_file)
        log.debug(model_dir, model_name, start_year, end_year, model_file_template)

        log.info(f'Loading model from file: {model_file}')
        with open(model_file, 'rb') as file:
            model_bin = pickle.load(file)
        mw = ModelWrapper(model_bin, description, data_file, start_year, end_year, model_name = model_name)

        # load json fields
        mw.cm = json.loads(confusion_matrix_str)
        mw.cr = json.loads(classification_report_str)
        if column_filters is not None:
            mw.column_filters = json.loads(column_filters)
        else:
            mw.column_filters = None

        # set other variables
        mw.accuracy = accuracy
        mw.roc_auc_score = roc_auc_score
        mw.fit_time_min = fit_time_min
        mw.predict_time_min = predict_time_min


        if load_data:
            mw.X_train, mw.X_test, mw.y_train, mw.y_test = ModelWrapper.get_data(data_file, LABEL_COL, start_year, end_year, column_filters=mw.column_filters)

        return mw


    @staticmethod
    def _get_model_filename(model_name: str):
        """
        Creates model file name based on model_name and model_file_format
        :param model_name: name of current model
        :return: fully qualitifed file path of model to save
        """
        return f'{ModelWrapper.MODEL_DIR}/{model_name.lower()}-{ModelWrapper.model_file_format}'

    @staticmethod
    def _get_info_from_model_filename(model_fullpath: str) -> (str, str, str, str, str):
        """
        reverses _get_model_filename - takes in a fully qualified path for the model file
        and return the model_name and model_file_format as a tuple

        :param model_fullpath: fully qualitifed path of model file
        :return: model_dir and template of model
        """
        matches = re.search(r'(.*)/([a-zA-Z]+)-([\d]+)-([\d]+)-([\-\d\w]+)\.pkl', model_fullpath)
        model_dir = matches[1]
        # model_name = matches[2]
        # start_year = int(matches[3])
        # end_year = int(matches[4])
        model_template = matches[5]
        # return model_dir, model_name, start_year, end_year, description
        return model_dir, model_template


    def __init__(self, model, description, data_file, start_year, end_year,
                 X_train = None, y_train = None, X_test = None, y_test = None, model_name = None,
                 model_file_format = None, model_dir = None, report_file = None, column_filters = None):
        # TODO: remove x_test, y_test, etc
        """
        Creates a model wrapper object
        :param model: model binary file
        :param description: description of mode - this will be used for labels for graphs, etc
        :param data_file: which data file was used to train the model
        :param start_year: start of year data was used to train model - inclusive ie, >=
        :param end_year: end of year data was used to train model - inclusive ie, <=
        :param model_name: name of model - ie, KNeighborClassifer
        :param model_file_format: file format used to pick the binary model
        :param model_dir: directory to save model file
        :param report_file: directory + filename of where to write the report
        :param column_filters: list of filter classnames to be loaded -ie,
        """
        self.description = description
        self.data_file = data_file
        self.start_year = start_year
        self.end_year = end_year
        self.model_file_format = model_file_format

        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.column_filters = column_filters

        if not model_dir:
            self.model_dir = ModelWrapper.MODEL_DIR

        if not report_file:
            self.report_file = ModelWrapper.REPORT_FILE

        if not self.model_file_format:
            self.model_file_format = f'{self.start_year}-{self.end_year}-{description}.pkl'

        #  confusion matrix
        self.cm = None
        # classification report
        self.cr = None
        # accuracy score
        self.accuracy = None
        self.roc_auc_score = None
        self.fit_time_min = 0
        self.predict_time_min = 0
        self.sample_weights = None

        if model_name:
            self.model_name = model_name
        else:
            self.model_name = type(self.model).__name__
        self.model_file = f'{ModelWrapper.MODEL_DIR}/{self.model_name.lower()}-{self.model_file_format}'


    def fit(self, X_train = None, y_train = None, sample_weights = None) -> pd.DataFrame:
        start_time = datetime.now()
        if X_train is not None and y_train is not None:
            self.X_train = X_train
            self.y_train = y_train
        self.sample_weights = sample_weights
        if self.sample_weights is not None:
            self.model = self.model.fit(self.X_train, self.y_train, self.sample_weights)
        else:
            self.model = self.model.fit(self.X_train, self.y_train)
        end_time = datetime.now()
        self.fit_time_min = divmod((end_time - start_time).total_seconds(), 60)[0]
        return self

    def predict(self, X = None):
        if X is not None:
            self.X_test = X
        start_time = datetime.now()
        self.y_predict = self.model.predict(self.X_test)
        end_time = datetime.now()
        self.predict_time_min = divmod((end_time - start_time).total_seconds(), 60)[0]
        return self.y_predict

    def analyze(self, y_test = None):
        # TODO: make y_test required after refactor

        if y_test is not None:
            self.y_test = y_test

        self.accuracy = accuracy_score(self.y_test, self.y_predict)
        print(f'Model Score: {accuracy_score(self.y_test, self.y_predict)}\n')

        self.roc_auc_score = roc_auc_score(self.y_test, self.y_predict)
        print(f'ROC/AUC Score: {self.roc_auc_score}')

        cr_str = classification_report(self.y_test, self.y_predict, target_names=['Loss', 'Win'])
        self.cr = classification_report(self.y_test, self.y_predict, target_names=['Loss', 'Win'], output_dict=True)
        print(cr_str)

        self.cm = confusion_matrix(self.y_test, self.y_predict)
        cm = pd.DataFrame(self.cm, index=['Loss', 'Win'], columns=['Loss', 'Win'])
        print(cm)

        sns.heatmap(cm, annot=True, vmin=0, vmax=len(self.y_predict))

    def save(self):
        log.info(f'Saving model file: {self.model_file}')
        pickle.dump(self.model, open(self.model_file, 'wb'))

        d = {
            ModelWrapper.MODEL_NAME: [self.model_name, ],
            ModelWrapper.DESCRIPTION: [self.description, ],
            ModelWrapper.DATA_FILE: [self.data_file, ],
            ModelWrapper.START_YEAR: [self.start_year, ],
            ModelWrapper.END_YEAR: [self.end_year, ],
            ModelWrapper.ACCURACY: [self.accuracy, ],
            ModelWrapper.ROC_AUC_SCORE: [self.roc_auc_score, ],
            ModelWrapper.CONFUSION_MATRIX: [json.dumps(pd.DataFrame(self.cm).to_dict()), ],
            ModelWrapper.CLASSIFICATION_REPORT: [json.dumps(self.cr), ],
            ModelWrapper.MODEL_FILE: [self.model_file, ],
            ModelWrapper.PREDICT_TIME_MIN: [self.predict_time_min, ],
            ModelWrapper.FIT_TIME_MIN: [self.fit_time_min, ],
            ModelWrapper.TOTAL_TIME_MIN: [self.fit_time_min + self.predict_time_min, ],
        }
        if self.column_filters is not None:
            d[ModelWrapper.COLUMN_FILTERS] =  [json.dumps(self.column_filters), ]

        if os.path.exists(self.report_file):
            log.info(f'Reading report: {self.report_file}')
            report = pd.read_csv(self.report_file)
        else:
            report = pd.DataFrame(columns=list(d.keys()))

        report = report.append(pd.DataFrame(d), ignore_index=True, sort=False)
        log.debug(f'report dataframe:\n{report}')
        print(f'Saving report: {self.report_file}')
        report.to_csv(self.report_file, index=False)
        # report.to_csv(ModelWrapper.report_file, index=False, sep="|")
