import pytest
import logging
from util.model_util import ModelWrapper

log = logging.getLogger(__name__)

MODEL_DIR = "../models"
DESCRIPTION = "diff-ohe-history-matchup"
START_YEAR = 1998
END_YEAR = 2019
MODEL_TEMPLATE_NAME = f'{START_YEAR}-{END_YEAR}-{DESCRIPTION}.pkl'
MODEL_NAME = "ModelName"




def test_get_model_file():

    model_file_format = MODEL_TEMPLATE_NAME
    ModelWrapper.model_file_format = model_file_format
    model_name = MODEL_NAME
    filename = ModelWrapper._get_model_filename(model_name)
    assert filename == f"{ModelWrapper.model_dir}/{model_name}-{model_file_format}", "filename mismatch"


def test_get_info_from_model_filename():

    model_dir, model_name, start_year, end_year, description = ModelWrapper._get_info_from_model_filename(f'{MODEL_DIR}/{MODEL_TEMPLATE_NAME}')
    assert model_dir == MODEL_DIR, "mismatched model_dir"
    assert model_name == MODEL_NAME, "mismatched model_name"
    assert start_year == START_YEAR, "mismatched start_year"
    assert end_year == START_YEAR, "mismatched end_year"
    assert description == DESCRIPTION, "mismatched description"
