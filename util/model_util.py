import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score
import logging
import pickle
import seaborn as sns
import json
import os


log = logging.getLogger(__name__)


class ModelWrapper(object):

    # class variable for all models in the notebook
    # you need to set this fore calling constructor
    model_file_format = None
    report_file = None
    description = None
    data_file = None
    start_year = None

    @staticmethod
    def init(description, data_file, start_year, model_file_format, report_file = '../reports/summary.csv'):
        ModelWrapper.description = description
        ModelWrapper.data_file = data_file
        ModelWrapper.start_year = start_year
        ModelWrapper.model_file_format = model_file_format
        ModelWrapper.report_file = report_file


    def __init__(self, model, X_train, y_train, X_test, y_test, model_name = None):
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


    def fit(self) -> pd.DataFrame:
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
            "model": [self.model_name, ],
            "description": [ModelWrapper.description, ],
            "data_file": [ModelWrapper.data_file, ],
            "start_year": [ModelWrapper.start_year, ],
            "accuracy": [self.accuracy, ],
            "confusion_matrix": [json.dumps(pd.DataFrame(self.cm).to_dict()), ],
            "classification_report": [json.dumps(self.cr), ],
            "model_file": [self.model_file, ]
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
