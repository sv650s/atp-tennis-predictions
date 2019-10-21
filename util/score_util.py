import logging
import pandas as pd
import re
import numpy as np

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

    log.debug(f'scores {scores}')
    set_score = [0, 0]
    game_score = [0, 0]

    # we don't have some of the scores
    if scores is not np.nan and scores is not None:
        scores = clean_score(scores)
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

    set_diff = set_score[0] - set_score[1]
    games_diff = game_score[0] - game_score[1]

    return set_score[0], game_score[0], set_score[1], game_score[1], set_diff, games_diff

def breakup_match_score(scores: str):
    """
    Breaks up match score into how many games winner and loser won per set

    Since matches can have up to 5 sets, this function will return 10 numbers with 0's if not played

    :param scores:
    :return: winner game(wg) 1, loser game(lg) 1, wg 2, lg 2, wg 3, lg 3, wg 4, lg 4, wg 5, lg 5,
                s1_diff, s2_diff, s3_diff, s4_diff, s5_diff, sets
    """
    # print(scores)
    # print(type(scores))
    # print(scores is None or scores is np.nan)

    # winner games
    wg = [0] * 5
    # loser games
    lg = [0] * 5
    sets = []

    # we dont' have scores for some matches
    if scores is not np.nan and scores is not None:
        sets = clean_score(scores).split()
        for idx in np.arange(0, len(sets)):
            set_scores = sets[idx].split("-")
            wg[idx] = int(set_scores[0])
            lg[idx] = int(set_scores[1])

    return wg[0], lg[0], wg[1], lg[1], wg[2], lg[2], wg[3], lg[3], wg[4], lg[4], \
           wg[0] - lg[0], wg[1] - lg[1], wg[2] - lg[2], wg[3] - lg[3], wg[4] - lg[4], len(sets)


def clean_score(scores: str):
    """
    Since we are not going to be predictiong down to the points of a match (missing for normal sets.
    We want to clean up match score by removing tiebreak points information. Normally this is indicated by ()

    :param scoresreturn:
    """
    scores = re.sub(r'\(\d+\)', '', scores)
    return re.sub(r"[^0-9-\ ]", "", scores).strip()


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


