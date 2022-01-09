# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
We would like to identify and predict credit card customers that are most likely to churn. The main aim of this project is to build a customer churn prediction model and implement all of the clean code principles based on the provided code in `churn_notebook.ipynb`. 

The dataset for this project was pulled from [Kaggle](https://www.kaggle.com/sakshigoyal7/credit-card-customers/code). 


## Running Files
First, we create virtual environment (`venv`) and activate the newly created virtual environment. 


#### 1. create and set virtual environment (venv)
```bash
git clone <github HTTPS filepath>
virtualenv venv
source venv/bin/activate
```
### 2. install requirements.txt
Then install all dependencies of this file. 

```bash 
pip install -r requirements.txt
```
### 3. for testing the code on `churn_library.py`

```bash 
pip install pytest
pytest churn_script_logging_and_tests.py
```
