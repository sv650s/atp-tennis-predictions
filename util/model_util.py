import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score
import logging
import pickle
import seaborn as sns
import json
import os
import re
import json
from datetime import datetime


log = logging.getLogger(__name__)
REPORT_FILE = '../reports/summary.csv'
MODEL_DIR = '../models'


class ModelWrapper(object):

    # default values
    report_file = REPORT_FILE
    model_dir = MODEL_DIR

    # keys
    MODEL_NAME = "model_name"
    MODEL_FILE = "model_file"
    DESCRIPTION = "description"
    DATA_FILE = "data_file"
    START_YEAR = "start_year"
    END_YEAR = "end_year"
    ACCURACY = "accuracy"
    CONFUSION_MATRIX = "confusion_matrix"
    CLASSIFICATION_REPORT = "classification_report"
    FIT_TIME_MIN = "fit_time_min"
    PREDICT_TIME_MIN = "predict_time_min"
    TOTAL_TIME_MIN = "total_time_min"

    @staticmethod
    def get_model_wrapper_from_report(data: pd.DataFrame):
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

        :param data: a row in the report data frame
        :return: ModelWrapper object
        """
        log.info(type(data))
        log.info(data)
        assert len(data) == 1, f"data must of length 1 - got {len(data)}"

        # get these from the file name
        model_name = data[ModelWrapper.MODEL_NAME].values[0]
        start_year = int(data[ModelWrapper.START_YEAR].values[0])
        end_year = int(data[ModelWrapper.END_YEAR].values[0])
        description = data[ModelWrapper.DESCRIPTION].values[0]
        data_file = data[ModelWrapper.DATA_FILE].values[0]
        accuracy = float(data[ModelWrapper.ACCURACY].values[0])
        confusion_matrix_str = data[ModelWrapper.CONFUSION_MATRIX].values[0]
        classification_report_str = data[ModelWrapper.CLASSIFICATION_REPORT].values[0]
        model_file = data[ModelWrapper.MODEL_FILE].values[0]
        predict_time_min = int(data[ModelWrapper.PREDICT_TIME_MIN].values[0])
        fit_time_min = int(data[ModelWrapper.FIT_TIME_MIN].values[0])

        log.debug(f'model_file {model_file}')


        model_dir, model_file_template = ModelWrapper._get_info_from_model_filename(model_file)
        log.debug(model_dir, model_name, start_year, end_year, model_file_template)

        with open(model_file, 'rb') as file:
            model_bin = pickle.load(file)
        mw = ModelWrapper(model_bin, description, data_file, start_year, end_year, model_name = model_name)

        # load confusion matrix and classification_report
        mw.cm = json.loads(confusion_matrix_str)
        mw.cr = json.loads(classification_report_str)

        # set other variables
        mw.accuracy = accuracy
        mw.fit_time_min = fit_time_min
        mw.predict_time_min = predict_time_min

        return mw


    @staticmethod
    def _get_model_filename(model_name: str):
        """
        Creates model file name based on model_name and model_file_format
        :param model_name: name of current model
        :return: fully qualitifed file path of model to save
        """
        return f'{ModelWrapper.model_dir}/{model_name.lower()}-{ModelWrapper.model_file_format}'

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
                model_file_format = None, model_dir = MODEL_DIR, report_file = REPORT_FILE):
        """
        Creates a model wrapper object
        :param model: model binary file
        :param description: description of mode - this will be used for labels for graphs, etc
        :param data_file: which data file was used to train the model
        :param start_year: start of year data was used to train model - inclusive ie, >=
        :param end_year: end of year data was used to train model - inclusive ie, <=
        :param X_train: training features
        :param y_train: training labels
        :param X_test: test features
        :param y_test: test labels
        :param model_name: name of model - ie, KNeighborClassifer
        :param model_file_format: file format used to pick the binary model
        :param model_dir: directory to save model file
        :param report_file: directory + filename of where to write the report
        """
        self.description = description
        self.data_file = data_file
        self.start_year = start_year
        self.end_year = end_year
        self.model_file_format = model_file_format
        self.model_dir = model_dir
        self.report_file = report_file

        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        if not self.model_file_format:
            self.model_file_format = f'{self.start_year}-{self.end_year}-{description}.pkl'

        # predictions
        self.y_predict = None
        #  confusion matrix
        self.cm = None
        # classification report
        self.cr = None
        # accuracy score
        self.accuracy = None
        self.fit_time_min = None
        self.predict_time_min = None

        if model_name:
            self.model_name = model_name
        else:
            self.model_name = type(self.model).__name__
        self.model_file = f'../models/{self.model_name.lower()}-{self.model_file_format}'


    def fit(self, X_train = None, y_train = None) -> pd.DataFrame:
        start_time = datetime.now()
        if X_train and y_train:
            self.X_train = X_train
            self.y_train = y_train
        self.model = self.model.fit(self.X_train, self.y_train)
        end_time = datetime.now()
        self.fit_time_min = divmod((end_time - start_time).total_seconds(), 60)[0]
        return self

    def predict(self):
        start_time = datetime.now()
        self.y_predict = self.model.predict(self.X_test)
        end_time = datetime.now()
        self.predict_time_min = divmod((end_time - start_time).total_seconds(), 60)[0]
        return self.y_predict

    def analyze(self):
        self.accuracy = accuracy_score(self.y_test, self.y_predict)

        print(f'Model Score: {accuracy_score(self.y_test, self.y_predict)}\n')

        cr_str = classification_report(self.y_test, self.y_predict, target_names=['Loss', 'Win'])
        self.cr = classification_report(self.y_test, self.y_predict, target_names=['Loss', 'Win'], output_dict=True)
        print(cr_str)

        self.cm = confusion_matrix(self.y_test, self.y_predict)
        cm = pd.DataFrame(self.cm, index=['Loss', 'Win'], columns=['Loss', 'Win'])
        print(cm)

        sns.heatmap(cm, annot=True, vmin=0, vmax=len(self.y_predict))

    def save(self):
        pickle.dump(self.model, open(self.model_file, 'wb'))

        d = {
            ModelWrapper.MODEL_NAME: [self.model_name, ],
            ModelWrapper.DESCRIPTION: [self.description, ],
            ModelWrapper.DATA_FILE: [self.data_file, ],
            ModelWrapper.START_YEAR: [self.start_year, ],
            ModelWrapper.END_YEAR: [self.end_year, ],
            ModelWrapper.ACCURACY: [self.accuracy, ],
            ModelWrapper.CONFUSION_MATRIX: [json.dumps(pd.DataFrame(self.cm).to_dict()), ],
            ModelWrapper.CLASSIFICATION_REPORT: [json.dumps(self.cr), ],
            ModelWrapper.MODEL_FILE: [self.model_file, ],
            ModelWrapper.PREDICT_TIME_MIN: [self.predict_time_min, ],
            ModelWrapper.FIT_TIME_MIN: [self.fit_time_min, ],
            ModelWrapper.TOTAL_TIME_MIN: [self.fit_time_min + self.predict_time_min, ],

        }

        if os.path.exists(self.report_file):
            log.info(f'Reading report: {self.report_file}')
            report = pd.read_csv(self.report_file)
        else:
            report = pd.DataFrame(columns=list(d.keys()))

        report = report.append(pd.DataFrame(d), ignore_index=True)
        log.debug(f'report dataframe:\n{report}')
        log.info(f'Saving report: {self.report_file}')
        report.to_csv(self.report_file, index=False)
        # report.to_csv(ModelWrapper.report_file, index=False, sep="|")
