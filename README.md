# ATP Tennis Predictions

For this project, I will be using various datasets from ATP tennis to see if we can accurately predict results for professional tennis tournaments

I looked at a number of tennis datasets and recently found this one: https://github.com/JeffSackmann/tennis_atp

The data is generally pretty clean and more current than any of the datasets that I found on Kaggle.

# Requirements

* Python 3.7
* Ananconda
* PySpark - https://www.sicara.ai/blog/2017-05-02-get-started-pyspark-jupyter-notebook-3-minutes

# Anaconda Environment Setup

```
conda env create --file environment.yml
```
# PySpark Setup Using Brew

NOTE: PySpark 2.4.4 only supports Java 8. You will need to install cask to help you install this JDK

```
brew tap caskroom/cask
brew cask install homebrew/cask-versions/adoptopenjdk8
brew install apache-spark
```


## To Run PySpark in Jupyter Notebooks

Add the following to your ~/.bash_profile:
```
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS='lab'
```
Source your ~/.bash_profile to pick up the new variables
```
source ~/.bash_profile
```

Run the following command in terminal:
```
pyspark
```
