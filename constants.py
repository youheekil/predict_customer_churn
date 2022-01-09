CATEGORY_LST = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
EDA_COL_NAMES = ["Heatmap", "Total_Trans", "Churn", "Customer_Age", "Marital_Status"]
RESPONSE = 'Churn'
DATA_PATH = "./data/bank_data.csv"
EDA_FILEPATH = "./images/eda/"
RESULTS_FILEPATH = "./images/results/"
MODEL_FILEPATH = "./models/"
CLA_REPORT_IMAGE_FILEPATH = "./images/results"
PARAM_GRID =  {
        'n_estimators': [200, 500], 
        'max_features': ['auto', 'sqrt'], 
        'max_depth': [4, 5, 100], 
        'criterion': ['gini', 'entropy']
    }
