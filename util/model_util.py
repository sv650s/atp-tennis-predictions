import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score
import logging
import pickle
import seaborn as sns
import json
import os
import re
import json


log = logging.getLogger(__name__)


class ModelWrapper(object):


    # keys
    MODEL = "model"
    MODEL_FILE = "model_file"
    DESCRIPTION = "description"
    DATA_FILE = "data_file"
    START_YEAR = "start_year"
    ACCURACY = "accuracy"
    CONFUSION_MATRIX = "confusion_matrix"
    CLASSIFICATION_REPORT = "classification_report"

    # class variable for all models in the notebook
    # you need to set this fore calling constructor
    model_dir = "../models"
    model_file_format = None
    report_file = None
    description = None
    data_file = None
    start_year = None

    @staticmethod
    def init(description, data_file, start_year, model_file_format, model_dir = "../models", report_file = '../reports/summary.csv'):
        ModelWrapper.description = description
        ModelWrapper.data_file = data_file
        ModelWrapper.start_year = int(start_year)
        ModelWrapper.model_file_format = model_file_format
        ModelWrapper.report_file = report_file
        ModelWrapper.model_dir = model_dir

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

        model = data[ModelWrapper.MODEL].values[0]
        description = data[ModelWrapper.DESCRIPTION].values[0]
        data_file = data[ModelWrapper.DATA_FILE].values[0]
        start_year = int(data[ModelWrapper.START_YEAR].values[0])
        accuracy = float(data[ModelWrapper.ACCURACY].values[0])
        confusion_matrix_str = data[ModelWrapper.CONFUSION_MATRIX].values[0]
        classification_report_str = data[ModelWrapper.CLASSIFICATION_REPORT].values[0]
        model_file = data[ModelWrapper.MODEL_FILE].values[0]

        log.debug(f'model_file {model_file}')


        model_dir, model_name, start_year, end_year, model_file_template = ModelWrapper._get_info_from_model_filename(model_file)
        log.debug(model_dir, model_name, start_year, end_year, model_file_template)

        ModelWrapper.init(description, data_file, start_year, model_file_template)

        with open(model_file, 'rb') as file:
            model_bin = pickle.load(file)
        mw = ModelWrapper(model_bin, model_name = model_name)

        # load confusion matrix and classification_report
        mw.cm = json.loads(confusion_matrix_str)
        mw.cr = json.loads(classification_report_str)

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
        :return:
        """
        matches = re.search(r'(.*)/([a-zA-Z]+)-([\d]+)-([\d]+)-([\-\d\w]+)\.pkl', model_fullpath)
        model_dir = matches[1]
        model_name = matches[2]
        start_year = int(matches[3])
        end_year = int(matches[4])
        description = matches[5]
        return model_dir, model_name, start_year, end_year, description


    def __init__(self, model, X_train = None, y_train = None, X_test = None, y_test = None, model_name = None):
        # TODO: move this to class level
        if ModelWrapper.model_file_format and \
                ModelWrapper.report_file and \
                ModelWrapper.description and \
                ModelWrapper.data_file:
            self.model = model
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test

            self.y_predict = None
            #  confusion matrix
            self.cm = None
            # classification report
            self.cr = None
            # accuracy score
            self.accuracy = None

            if model_name:
                self.model_name = model_name
            else:
                self.model_name = type(self.model).__name__
            self.model_file = f'../models/{self.model_name.lower()}-{ModelWrapper.model_file_format}'
        else:
            raise Exception("file_format and report_file needs to be initialized before using")


    def fit(self, X_train = None, y_train = None) -> pd.DataFrame:
        if X_train and y_train:
            self.X_train = X_train
            self.y_train = y_train
        self.model = self.model.fit(self.X_train, self.y_train)
        return self

    def predict(self):
        self.y_predict = self.model.predict(self.X_test)
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
            ModelWrapper.MODEL: [self.model_name, ],
            ModelWrapper.DESCRIPTION: [ModelWrapper.description, ],
            ModelWrapper.DATA_FILE: [ModelWrapper.data_file, ],
            ModelWrapper.START_YEAR: [ModelWrapper.start_year, ],
            ModelWrapper.ACCURACY: [self.accuracy, ],
            ModelWrapper.CONFUSION_MATRIX: [json.dumps(pd.DataFrame(self.cm).to_dict()), ],
            ModelWrapper.CLASSIFICATION_REPORT: [json.dumps(self.cr), ],
            ModelWrapper.MODEL_FILE: [self.model_file, ]
        }

        if os.path.exists(ModelWrapper.report_file):
            log.info(f'Reading report: {ModelWrapper.report_file}')
            report = pd.read_csv(ModelWrapper.report_file)
        else:
            report = pd.DataFrame(columns=list(d.keys()))

        report = report.append(pd.DataFrame(d), ignore_index=True)
        log.debug(f'report dataframe:\n{report}')
        log.info(f'Saving report: {ModelWrapper.report_file}')
        report.to_csv(ModelWrapper.report_file, index=False)
        # report.to_csv(ModelWrapper.report_file, index=False, sep="|")
