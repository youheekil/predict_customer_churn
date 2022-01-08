
if __name__ == "__main__":
    data = import_data(constants.PATH)
    perform_eda(df=data)
    new_data = encoder_helper(
        data,
        category_lst=constants.CATEGORY_LST,
        response=constants.RESPONSE)
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df=new_data, response=constants.RESPONSE)
    y_train_preds_lr, y_test_preds_lr, y_train_preds_rf, y_test_preds_rf = train_models(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    classification_report_image(y_train=y_train,
                                y_test=y_test,
                                y_train_preds_lr=y_train_preds_lr,
                                y_train_preds_rf=y_train_preds_rf,
                                y_test_preds_lr=y_test_preds_lr,
                                y_test_preds_rf=y_test_preds_rf)
    feature_importance_plot(
        model=joblib.load('./models/rfc_model.pkl'),
        X_data=X_test,
        output_pth=constants.RESULTS_FILEPATH)