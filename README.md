# ATP Tennis Predictions

For this project, I will use match data from ATP tennis to see if we can accurately predict winners for professional tennis matches

Raw data for my dataset can be downloaded here: com/JeffSackmann/tennis_atp (looks like there has been updates since I dowloaded this)

As a final test for this project, I will use the best model to make predictions on a particular tournmanet and look at our prediction accuracy


## Summary

* Gradient Boosting gave us the best results for this problem 
* While I did try some neural network architectures, they never exceeded accuracy for Gradient Boosting
* Adding match history data for players as well as matchup data leading up to a match did slightly improve performance but only marginally
* Even though our training data only got 69% accuracy, when we used new data that the model has not seen before (Sydney Open 2019), we accurately predicted 22 out of 27 matches correctly, giving us a 85% accuracy score

For details and final analysis, please refer to this [notebook]()

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


