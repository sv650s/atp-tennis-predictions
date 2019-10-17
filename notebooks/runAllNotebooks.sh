#!/bin/bash
#
# use nbconvert to run all notebooks again
#

for notebook in `ls [7]*.ipynb`; do
    echo ""
    echo "`date` Running $notebook"
    jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=1200 --execute $notebook 2>&1
    echo "`date` Finished running $notebook"
    echo ""
done

