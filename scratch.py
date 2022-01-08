
if __name__ == "__main__":
    data_df = import_data(constants.DATA_PATH)
    perform_eda(data_df)
    encoded_data_df = encoder_helper(data_df, constants.CATEGORY_LST, constants.RESPONSE)
    x_train_, x_test_, y_train_, y_test_ = perform_feature_engineering(encoded_data_df, constants.RESPONSE)
    train_models(x_train_, x_test_, y_train_, y_test_)
