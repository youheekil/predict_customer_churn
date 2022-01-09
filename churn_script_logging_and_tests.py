import logging
import pytest
from churn_library import import_data, perform_eda
import constants


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data(constants.DATA_PATH)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return df


def test_eda(perform_eda):
    '''
    test perform eda function - test creation of images related eda
    '''
    perform_eda(data_frame=import_data(constants.DATA_PATH))
    for col in constants.EDA_COL_NAMES:
            try:
                 img_file_pth = f"{constants.EDA_FILEPATH}{col}.png"
                 with open(img_file_pth, 'r'):
                     logging.info("Testing perform_eda: SUCCESS")
            except FileNotFoundError as err:
                logging.error("Testing perform_eda: generated images missing")
                raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        encoded_data = encoder_helper(import_data(constants.DATA_PATH), 
                                                    constants.CATGORY_LST, 
                                                    constants.RESULTS)
        logging.info("Testing test_encoder_helper: SUCCESS")
    except KeyError as err:
        logging.error("Testing test_encoder_helper: The file doesn't appear to have rows and columns")
    
    try:
        assert encoded_data.shape[0] > 0
        assert encoded_data.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing test_encoder_helper: The dataframe has an issue with containig either rows or columns")
        raise err
    
    try:
        assert isinstance(
            encoded_data[f"{constants.CATEGORY_LST[0]}_Churn"], float)
        assert isinstance(
            encoded_data[f"{constants.CATEGORY_LST[1]}_Churn"], float)
    except AssertionError as err:
        logging.error(
            "Testing test_encoder_helper: The values of the encoded variable are not float")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    pass
# TODO: CHECK 4 DATA - # COLUMNS - feature sequences fixtures - returns 4 series containing features sequences
# TODO: len(x_train) == len(y_train)
# TODO: len(x_test) = len(y_test)
# 



def test_train_models(train_models):
    '''
    test train_models
    '''
    pass
# TODO: check if models can be loaded it well 
# TODO: check number of images stored in the folder 

if __name__ == '__main__':
    test_import(import_data)
    test_eda(perform_eda)
