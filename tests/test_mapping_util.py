"""
pytest to test Mapper object
"""
import pytest
import pandas as pd
import logging
from util.mapping_util import Mapper


log = logging.getLogger(__name__)



def test_get_player_info_by_id(shared_datadir):
    """
    test lookup player by player_id
    :param shared_datadir:
    :return:
    """
    Mapper.PLAYERS_FILE = f'{shared_datadir}/players.csv'
    info = Mapper.get_player_info_by_id(player_id = "102778")
    assert len(info) == 1, "returned number of rows mismatched"
    assert info.name.values[0] == "abdul hamid makhkamov", "returned player name mismatch"


def test_get_player_info_by_ohe_col(shared_datadir):
    """
    test lookup of a person from players.csv using ohe_column parameter
    :param shared_datadir:
    :return:
    """
    Mapper.PLAYERS_FILE = f'{shared_datadir}/players.csv'
    info = Mapper.get_player_info_by_id(ohe_column = "p1_102778")
    assert len(info) == 1, "returned number of rows mismatched"
    assert info.name.values[0] == "abdul hamid makhkamov", "returned player name mismatch"

    info = Mapper.get_player_info_by_id(ohe_column = "p2_102778")
    assert len(info) == 1, "returned number of rows mismatched"
    assert info.name.values[0] == "abdul hamid makhkamov", "returned player name mismatch"

