#!/bin/bash
#
# some of the notebooks have test mode so I can test them to make sure they run
# use this script to create test
#
#

in_file="atp_matches_1985-2019_preprocessed.csv"
out_file="test-preprocessed.csv"

if [ $# -lt 1 ]; then
    echo "missing parameter: lines"
    exit 1
fi

lines=$1

# take the header from the file
head -1 $in_file > $out_file
# take the last lines from the file
tail -${lines} $in_file >> $out_file


# run jupyter notebooks to create other test files
(
    export IPYNB_DEBUG="True"
    cd ../notebooks
    jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=1200 --execute 3.0.0-classification_feature_engineering.ipynb

    jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=1200 --execute 3.1.0-classification_feature_engineering-history-matchup-stats.ipynb

    jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=1200 --execute 3.0.1-classification_feature_engineering-symmetric.ipynb

    rm *.nbconvert.ipynb

    unset IPYNB_DEBUG
)
