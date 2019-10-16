"""
Since we one hot encoded categories in our features, use this utility class to map back to values

This is implemented as a singleton
"""
import json
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import logging
import pickle
import re
import pandas as pd


log = logging.getLogger(__name__)



class Mapper(object):

    PLAYERS_FILE = '../datasets/players.csv'

    instance = None
    players_df = None
    player_mapper = None
    tourney_mapper = None

    @staticmethod
    def getInstance():
        if Mapper.instance is None:
            Mapper.instance = Mapper()
        return  Mapper.instance

    @staticmethod
    def _get_players_df():
        if Mapper.players_df is None:
            players_df = pd.read_csv(Mapper.PLAYERS_FILE)
            Mapper.players_df = players_df.astype({"id": str})
            log.debug(Mapper.players_df.head(1))
            log.debug(Mapper.players_df.info())
        return Mapper.players_df


    @staticmethod
    def get_player_info_by_id(ohe_columns = None, player_ids = None):
        """
        Returns all player information based on column name or just the player id
        :param ohe_columns: ohe column name - ie, p1_1234 or p2_1234
        :param player_ids: numeric player id - ie, 12234
        :return:
        """
        assert ohe_columns is not None or player_ids is not None, "Cannot pass both ohe_column and player_id"

        if ohe_columns:
            if isinstance(ohe_columns, list):
                player_ids = [col.split("_")[1] for col in ohe_columns]
            else:
                player_ids = [ohe_columns.split("_")[1]]
        log.info(f'player_id: {player_ids}')

        if not isinstance(player_ids, list):
            player_ids = [player_ids]

        players_df = Mapper._get_players_df()
        player_info = players_df[players_df.id.isin(player_ids)]
        log.debug(f'player info: {player_info}')
        return player_info




