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
    info = Mapper.get_player_info_by_id(player_ids="102778")
    assert len(info) == 1, "returned number of rows mismatched"
    assert info.name.values[0] == "abdul hamid makhkamov", "returned player name mismatch"

def test_get_player_info_by_id_list(shared_datadir):
    """
    test lookup player by player_id
    :param shared_datadir:
    :return:
    """
    Mapper.PLAYERS_FILE = f'{shared_datadir}/players.csv'
    info = Mapper.get_player_info_by_id(player_ids=["102778"])
    assert len(info) == 1, "returned number of rows mismatched"
    assert info.name.values[0] == "abdul hamid makhkamov", "returned player name mismatch"

    info = Mapper.get_player_info_by_id(player_ids=["102778", "106361"])
    assert len(info) == 2, "returned number of rows mismatched"
    assert info.name.values[0] == "abdul hamid makhkamov", "returned player name mismatch"

def test_get_player_info_by_ohe_col(shared_datadir):
    """
    test lookup of a person from players.csv using ohe_column parameter
    :param shared_datadir:
    :return:
    """
    Mapper.PLAYERS_FILE = f'{shared_datadir}/players.csv'
    info = Mapper.get_player_info_by_id(ohe_columns="p1_102778")
    assert len(info) == 1, "returned number of rows mismatched"
    assert info.name.values[0] == "abdul hamid makhkamov", "returned player name mismatch"

    info = Mapper.get_player_info_by_id(ohe_columns="p2_102778")
    assert len(info) == 1, "returned number of rows mismatched"
    assert info.name.values[0] == "abdul hamid makhkamov", "returned player name mismatch"

def test_get_player_info_by_ohe_col_list(shared_datadir):
    """
    test lookup of a person from players.csv using ohe_column parameter
    :param shared_datadir:
    :return:
    """
    Mapper.PLAYERS_FILE = f'{shared_datadir}/players.csv'
    info = Mapper.get_player_info_by_id(ohe_columns=["p1_102778"])
    assert len(info) == 1, "returned number of rows mismatched"
    assert info.name.values[0] == "abdul hamid makhkamov", "returned player name mismatch"


    info = Mapper.get_player_info_by_id(ohe_columns=["p2_102778", "p1_106361"])
    assert len(info) == 2, "returned number of rows mismatched"
    assert info.name.values[0] == "abdul hamid makhkamov", "returned player name mismatch"


def test_tourney_id_reverse_map(shared_datadir):
    """
    test to see if our reverse mapping is correct
    :return:
    """
    Mapper.TOURNEY_MAP_FILE = f'{shared_datadir}/tid_map.json'
    reverse_map = Mapper.get_tourney_id_reverse_map()
    print(reverse_map)
    assert reverse_map["163"] == "6116", "should get 6116"

def test_get_tourney_info_by_label(shared_datadir):
    """
    test to see if we can get tourney info by id properly
    :param shared_datadir:
    :return:
    """
    Mapper.TOURNEY_FILE = f'{shared_datadir}/preprocessed.csv'
    Mapper.TOURNEY_MAP_FILE = f'{shared_datadir}/tid_map.json'
    info = Mapper.get_tourney_info_by_label("163")
    assert len(info) == 1, "returned number of rows mismatched"
    assert info.name.values[0] == "atlanta", "returned tournament name mismatch"

    # test if we pass in list
    info = Mapper.get_tourney_info_by_label(["163"])
    assert len(info) == 1, "returned number of rows mismatched"
    assert info.name.values[0] == "atlanta", "returned tournament name mismatch"

def test_get_tourney_label(shared_datadir):
    """
    test to make sure we are getting the right label from tournament id
    """
    Mapper.TOURNEY_MAP_FILE = f'{shared_datadir}/tid_map.json'
    assert Mapper.get_tourney_label("0315") == "4", "returned wrong tourney label"
