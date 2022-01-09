import logging
import joblib
import pytest
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering, perform_feature_engineering, train_models


DATA_PATH = "./data/bank_data.csv"
CATEGORY_LST = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
EDA_COL_NAMES = ["Heatmap", "Total_Trans", "Churn", "Customer_Age", "Marital_Status"]
RESPONSE = 'Churn'
EDA_FILEPATH = "./images/eda/"
RESULTS_FILEPATH = "./images/results/"
MODEL_FILEPATH = "./models/"
FORMAT = "%(asctime)s | %(name)s - %(levelname)s - %(message)s"
LOG_FILEPATH = "./logs/churn_library.log/"
logging.basicConfig(
    filename=LOG_FILEPATH,
    level=logging.INFO,
    filemode='w',
    format=FORMAT)


@pytest.fixture
def raw_data():
    
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err
    return df


@pytest.fixture()
def encoded_data():

    try:
        _encoded_data = encoder_helper(import_data(DATA_PATH), 
                                                    CATEGORY_LST, 
                                                    RESPONSE)
        logging.info("Testing test_encoder_helper: SUCCESS")
    except KeyError as err:
        logging.error("Testing test_encoder_helper: The file doesn't appear to have rows and columns")
    return _encoded_data


@pytest.fixture()
def train_test_data(encoded_data):
    try:
        _X_train, _X_test, _y_train, _y_test = perform_feature_engineering(encoded_data, RESPONSE)
    except BaseException:
        logging.error(
            "Testing test_perform_feature_engineering helper: the splitted data mismatch")
        raise
    return _X_train, _X_test, _y_train, _y_test
    
    
def test_import(raw_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    df = raw_data
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return df


def test_eda(raw_data):
    '''
    test perform eda function - test creation of images related eda
    '''
    perform_eda(raw_data)
    for col in EDA_COL_NAMES:
            try:
                 img_file_pth = f"{EDA_FILEPATH}{col}.png"
                 with open(img_file_pth, 'r'):
                     logging.info("Testing perform_eda: SUCCESS")
            except FileNotFoundError as err:
                logging.error("Testing perform_eda: generated images missing")
                raise err


def test_encoder_helper(encoded_data):
    '''
    test encoder helper
    '''   
    try:
        assert encoded_data.shape[0] > 0
        assert encoded_data.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing test_encoder_helper: The dataframe has an issue with containig either rows or columns")
        raise err
    
    try:
        assert isinstance(
            encoded_data[f"{CATEGORY_LST[0]}_Churn"].mean(), float)
        assert isinstance(
            encoded_data[f"{CATEGORY_LST[1]}_Churn"].mean(), float)
    except AssertionError as err:
        logging.error(
            "Testing test_encoder_helper: The values of the encoded variable are not float")
        raise err


def test_perform_feature_engineering(encoded_data):
    '''
    test perform_feature_engineering
    '''
    try:
        _X_train, _X_test, _y_train, _y_test = perform_feature_engineering(encoded_data, RESPONSE)
    except BaseException:
        logging.error(
            "Testing test_perform_feature_engineering helper: the splitted data mismatch")
        raise 
    
    try:
        assert len(_X_train) == len(_y_train)
        assert len(_X_test) == len(_y_test)
        logging.info("Testing test_perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing test_perform_feature_engineering helper: the number of each train and test data values mismatch ")
        raise err


def test_train_models(train_test_data):
    '''
    test train_models
    '''
    train_models(train_test_data[0], train_test_data[1], train_test_data[2], train_test_data[3])
    try:
        joblib.load(f'{MODEL_FILEPATH}rfc_model.pkl')
        joblib.load(f'{MODEL_FILEPATH}logistic_model.pkl')
        logging.info("Testing testing_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: The models were not saved properly")
        raise err
    for img_name in [
        "logistic_results",
        "rf_results",
        "feature_importances_plot", 
        "roc_plot"]:
        try:
            img_result_path = f"{RESULTS_FILEPATH}{img_name}.png"
            with open(img_result_path, 'r'):
                logging.info("Testing testing_models : SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing testing_models : there is an issue with generating report images")
            raise err
