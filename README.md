# ATP Tennis Predictions

For this project, I will be using various datasets from ATP tennis to see if we can accurately predict match results for professional tennis matches

Raw data for my dataset can be downloaded here. Looks like data has been updated since I first started this project with more data from 2019: https://github.com/JeffSackmann/tennis_atp

As a final test for this project, I will use the best ML model to make predictions on a particular tournmanet and look at our prediction accuracy


## Summary of Classification

* Decision Tree seems to work the best when doing classification (win/loss)
* Adding match history data for players as well as matchup history leading up to a match did slightly improve performance but only marginally
* For our final prediction (Sydney Open 2019), we accurately predicted 54 out of 55 matches correctly, giving us a 85% accuracy score

# Requirements

* Python 3.7
* Ananconda

# Anaconda Environment Setup

```
conda env create --file environment.yml
```

# Project Structure

| Directory | Description |
| ---------- | ---------- |
| notebooks | jupyter notebooks |
| reports | summary.csv captures results from our various models and datasets |
| tests | pytests |
| util | utility functions |
| models | pickled ML models after each run as well as encoders (models not checked in because files were too big for git |
| datasets | data files (not all datasets are checked in because of size) |


