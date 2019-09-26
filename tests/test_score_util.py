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
    ws, wg, ls, lg = su.process_scores(score)
    assert ws == 3, "winning set should be 3"
    assert ls == 2, "loser set should be 2"
    assert wg == 28, "winning games should be 28"
    assert lg == 18, "winning games should be 18"


def test_seed_all_players(seed_df):
    print(seed_df.head())
    su.seed_all_players(seed_df, 1)

    # ranks are 3, 5, 7, 10 - so the corresponding seeds should be 1, 2, 3, 4
    assert seed_df.iloc[0].winner_seed == 2, "winner seed should be 2"
    assert seed_df.iloc[0].loser_seed == 1, "winner seed should be 1"
    assert seed_df.iloc[1].winner_seed == 4, "winner seed should be 4"
    assert seed_df.iloc[1].loser_seed == 3, "winner seed should be 3"


