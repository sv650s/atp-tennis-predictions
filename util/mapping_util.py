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
    TOURNEY_FILE = '../datasets/atp_matches_1985-2019_preprocessed.csv'
    TOURNEY_MAP_FILE = '../models/tid_map.json'

    instance = None
    players_df = None
    tourney_df = None
    # maps actual tourney ID to label
    tourney_id_map = None
    # maps label to actual tourney ID
    tourney_id_reverse_map = None

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
    def _get_tourney_df():
        if Mapper.tourney_df is None:
            tourney_df = pd.read_csv(Mapper.TOURNEY_FILE)
            tourney_df["year"] = tourney_df.tourney_id.apply(lambda x: x.split("-")[0])
            tourney_df["tourney_id"] = tourney_df.tourney_id.apply(lambda x: x.split("-")[1])
            # filter out columns
            tourney_df = tourney_df[["tourney_id", "year", "tourney_name", "surface", "draw_size", "tourney_level"]]
            Mapper.tourney_df = tourney_df.drop_duplicates(subset=["tourney_id", "tourney_name"], keep='last')
            Mapper.tourney_df = Mapper.tourney_df.rename({"tourney_id":"id", "tourney_name":"name", "tourney_level": "level"}, axis=1)
        return Mapper.tourney_df

    @staticmethod
    def get_tourney_id_reverse_map():
        if Mapper.tourney_id_reverse_map is None:
            tourney_id_map = Mapper.get_tourney_id_map()
            Mapper.tourney_id_reverse_map = { str(value) : key for key, value in tourney_id_map.items() }
        return Mapper.tourney_id_reverse_map

    @staticmethod
    def get_tourney_id_map():
        if Mapper.tourney_id_map is None:
            with open(Mapper.TOURNEY_MAP_FILE, 'r') as file:
                Mapper.tourney_id_map = json.load(file)
        return Mapper.tourney_id_map


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
                if "_" in ohe_columns:
                    player_ids = [col.split("_")[1] for col in ohe_columns]
                else:
                    player_ids = [col for col in ohe_columns]
            else:
                player_ids = [ohe_columns.split("_")[1]]
        log.info(f'player_id: {player_ids}')

        if not isinstance(player_ids, list):
            player_ids = [player_ids]

        players_df = Mapper._get_players_df()
        player_info = players_df[players_df.id.isin(player_ids)]
        log.debug(f'player info: {player_info}')
        return player_info

    @staticmethod
    def get_tourney_info_by_label(tourney_ids = None):
        """
        Get tournament name(s) by their ID's
        :param tourney_ids:
        :return:
        """

        tourney_id_reverse_map = Mapper.get_tourney_id_reverse_map()

        # convert input to list
        if isinstance(tourney_ids, list):
            mapped_tourney_ids = [tourney_id_reverse_map[id] for id in tourney_ids]
        else:
            mapped_tourney_ids = [tourney_id_reverse_map[tourney_ids]]

        print(f'mapped_tourney_ids: {mapped_tourney_ids}')

        tourney_df = Mapper._get_tourney_df()
        tourney_info = tourney_df[tourney_df.id.isin(mapped_tourney_ids)]
        return tourney_info

    @staticmethod
    def get_tourney_label(tourney_ids):
        """
        gets the tourney label from a tourney_id

        this supports passing in a single value or a list

        :param tourney_ids:  this can be a list or a string
        :return: string tourney label
        """
        tourney_id_map = Mapper.get_tourney_id_map()

        if not isinstance(tourney_ids, list):
            tourney_labels = str(tourney_id_map[tourney_ids])
        else:
            tourney_labels = [str(tourney_id_map[id]) for id in tourney_ids]

        return tourney_labels





