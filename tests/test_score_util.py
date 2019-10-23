import pytest
import util.score_util as su
import logging
import pandas as pd


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture(scope='class')
def seed_df():
    d = {"tourney_id": [1, 1],
         "winner_id": [5, 10],
         "winner_rank": [5, 10],
         "winner_seed": [None, None],
         "loser_id": [3, 7],
         "loser_rank": [3, 7],
         "loser_seed": [None, None]
         }
    return pd.DataFrame(d)


def test_process_scores():
    score = "6-7 4-6 6-3 6-2 6-0"
    ws, wg, ls, lg, sd, gd = su.process_scores(score)
    assert ws == 3, "winning set should be 3"
    assert ls == 2, "loser set should be 2"
    assert wg == 28, "winning games should be 28"
    assert lg == 18, "winning games should be 18"
    assert sd == 1, "set diff should be 1"
    assert gd == 10, "game diff should be 10"


def test_seed_all_players(seed_df):
    print(seed_df.head())
    su.seed_all_players(seed_df, 1)

    # ranks are 3, 5, 7, 10 - so the corresponding seeds should be 1, 2, 3, 4
    assert seed_df.iloc[0].winner_seed == 2, "winner seed should be 2"
    assert seed_df.iloc[0].loser_seed == 1, "winner seed should be 1"
    assert seed_df.iloc[1].winner_seed == 4, "winner seed should be 4"
    assert seed_df.iloc[1].loser_seed == 3, "winner seed should be 3"


def test_clean_score():
    score = "6-7(3) 6-3"
    assert su.clean_score(score) == "6-7 6-3", "did not remove tiebreak from first set"

    score = "6-3 6-7(3)"
    assert su.clean_score(score) == "6-3 6-7", "did not remove tiebreak from 2nd set"

    score = "6-7(3) 6-7(3)"
    assert su.clean_score(score) == "6-7 6-7", "did not remove tiebreak from both set"

    score = "b6-7(3) 6-7(3)a "
    assert su.clean_score(score) == "6-7 6-7", "did not remove special characters from score"

def test_breakup_match_scores():

    score = "7-6(3) 6-3"
    wg1, lg1, wg2, lg2, wg3, lg3, wg4, lg4, wg5, lg5, s1d, s2d, s3d, s4d, s5d, sets = su.breakup_match_score(score)
    assert wg1 == 7
    assert lg1 == 6
    assert wg2 == 6
    assert lg2 == 3
    assert wg3 == 0
    assert lg3 == 0
    assert wg4 == 0
    assert lg4 == 0
    assert wg5 == 0
    assert lg5 == 0
    assert s1d == 1, "set 1 diff wrong"
    assert s2d == 3, "set 2 diff wrong"
    assert s3d == 0, "set 3 diff wrong"
    assert s4d == 0, "set 4 diff wrong"
    assert s5d == 0, "set 5 diff wrong"
    assert sets == 2, "number of sets wrong"

    score = "7-5 6-3 2-6 6-7(3) 7-6(5)"
    wg1, lg1, wg2, lg2, wg3, lg3, wg4, lg4, wg5, lg5, s1d, s2d, s3d, s4d, s5d, sets = su.breakup_match_score(score)
    assert wg1 == 7
    assert lg1 == 5
    assert wg2 == 6
    assert lg2 == 3
    assert wg3 == 2
    assert lg3 == 6
    assert wg4 == 6
    assert lg4 == 7
    assert wg5 == 7
    assert lg5 == 6
    assert s1d == 2, "set 1 diff wrong"
    assert s2d == 3, "set 2 diff wrong"
    assert s3d == -4, "set 3 diff wrong"
    assert s4d == -1, "set 4 diff wrong"
    assert s5d == 1, "set 5 diff wrong"
    assert sets == 5, "number of sets wrong"

