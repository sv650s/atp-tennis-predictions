#!/bin/bash
#
# use nbconvert to run all notebooks via the command line
# if you use -d, this will use the test dataset that should run a lot faster. You can use this to verify that
# all notebooks have the right syntax
#
# running in debug mode will not overwrite existing notebooks since we don't pass the --inplace paramter in
#

usage() {
  echo "`$0`: [-d run all notebooks in debug]"
}

DEBUG="false"

while getopts "d" arg; do
  case $arg in
    h) echo "usage" & exit 0 ;;
    d) DEBUG="true" ;;
    ?) usage && exit 1 ;;
  esac
done


log_file="$0.log"

# delete existing log file
if [ -f $log_file ]; then
  rm $log_file
fi

if [ $DEBUG == "true" ]; then
  export IPYNB_DEBUG="True"
  echo "Running in DEBUG mode" | tee -a $log_file
  rm ../reports/summary-test.csv
fi


# notebooks starting with 4 are our regular ML classification notebooks
for notebook in `ls [4]*.ipynb | grep -v nbconvert`; do
    echo "" | tee -a $log_file
    echo "`date` Running $notebook" | tee -a $log_file
    if [ $DEBUG == "true" ]; then
      # don't run inplace
      jupyter nbconvert --to notebook --allow-errors --ExecutePreprocessor.timeout=1200 --execute $notebook 2>&1 | tee -a $log_file
    else
      jupyter nbconvert --to notebook --allow-errors --inplace --ExecutePreprocessor.timeout=1200 --execute $notebook 2>&1 | tee -a $log_file
    fi
    echo "`date` Finished running $notebook" | tee -a $log_file
    echo "" | tee -a $log_file
done

unset IPYNB_DEBUG

