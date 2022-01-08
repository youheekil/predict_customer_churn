CATEGORY_LST = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
RESPONSE = 'Churn'
DATA_PATH = "./data/bank_data.csv"
EDA_FILEPATH = "./images/eda/"
RESULTS_FILEPATH = "./images/results/"
CLA_REPORT_IMAGE_FILEPATH = "./images/results"
PARAM_GRID =  {
        'n_estimators': [200, 500], 
        'max_features': ['auto', 'sqrt'], 
        'max_depth': [4, 5, 100], 
        'criterion': ['gini', 'entropy']
    }
