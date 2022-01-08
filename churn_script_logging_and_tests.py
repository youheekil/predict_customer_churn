import os
import logging
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering, train_models
import constants


logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
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
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda):
	'''
	test perform eda function
	'''
	try:
		perform_eda(df = )
		logging.info("Testing perform_eda: SUCCESS")


def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	'''
	try:
		encoded_data = encoder_helper(df = import_data(constants.DATA_PATH), category_lst = constants.CATEGORY_LST, response=constants.RESPONSE)
		logging.info("Testing test_encoder_helper: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing test_encoder_helper: The file wasn't found")
		raise err
	
	try: 
		assert isinstance(encoded_data[f"{constants.CATEGORY_LST[0]}_Churn"], float)
		assert isinstance(encoded_data[f"{constants.CATEGORY_LST[1]}_Churn"], float)
		logging.info("Testing test_encoder_helper: SUCCESS")
	except AssertionError as err:
		logging.error("Testing test_encoder_helper: The values of the encoded variable are not float")
		raise err


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''
	pass


def test_train_models(train_models):
	'''
	test train_models
	'''
	pass


if __name__ == "__main__":
	test_import(import_data)
	test_encoder_helper(encoder_helper)








