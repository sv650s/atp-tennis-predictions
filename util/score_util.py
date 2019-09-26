import logging
import pandas as pd
import re

log = logging.getLogger(__name__)


def process_scores(scores: str):
    """
    Use this to parse out differnt parts of a match score.
    Will be using this for feature Engineering as input to predict match results

    :param scores: - string with scores - ie, 6-7 7-6(7) 6-3
    :return:
        winner_sets_won
        winner_games_won
        loser_sets_won
        loser_games_won
    """

    scores = re.sub(r"[^0-9-\ ]","", scores).strip()
    log.debug(f'scores {scores}')
    set_score = [0, 0]
    game_score = [0, 0]
    sets = scores.split()
    for set in sets:
        p1 = int(set.split("-")[0])
        p2 = int(set.split("-")[1])
        game_score[0] += p1
        game_score[1] += p2
        if p1 > p2:
            set_score[0] += 1
        else:
            set_score[1] += 1

    return set_score[0], game_score[0], set_score[1], game_score[1]


def seed_all_players(matches: pd.DataFrame, tid: str) -> pd.DataFrame:
    """
    Seed all players in a tournament according to the rank of the player

    :param matches: dataframe with matches to update
    :param tid: year + tournament id - ie, 1998-345
    :return: dataframe with seeds updated for the tournament
    """
    tourney_matches = matches[matches.tourney_id == tid]
    winners = pd.DataFrame(tourney_matches[["winner_id", "winner_rank"]]).rename({"winner_id": "player_id", "winner_rank": "rank"}, axis=1)
    losers = pd.DataFrame(tourney_matches[["loser_id", "loser_rank"]]).rename({"loser_id": "player_id", "loser_rank": "rank"}, axis=1)

    # let's seed all players in the tournament
    players = winners.append(losers, ignore_index=True).drop_duplicates().sort_values("rank")
    counter = 1
    for index, player in players.iterrows():
        players.loc[index, "seed"] = counter
        counter += 1

    log.debug(f'seeding players {players}')

    for index, match in tourney_matches.iterrows():
        if pd.isnull(match.winner_seed):
            seed = players[players.player_id == match.winner_id]["seed"].values[0]
    #         print(f'index {index} winner_id {match.winner_id} winner seed {seed}')
            matches.loc[index, 'winner_seed'] = players[players.player_id == match.winner_id]["seed"].values[0]
        if pd.isnull(match.loser_seed):
            seed = players[players.player_id == match.loser_id]["seed"].values[0]
    #         print(f'index {index} loser_id {match.loser_id} loser seed {seed}')
            matches.loc[index, 'loser_seed'] = seed


