"""
Project Predict Customer Churn

Author: Youhee
Date: Dec 2021
"""

# import libraries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import classification_report
import constants


def import_data(data_path):
    '''
    returns dataframe for the csv found at pth
    input:
            data_path: a path to the csv
    output:
            imported_data: imported data in pandas dataframe
    '''
    imported_data = pd.read_csv(rf"{data_path}")
    imported_data['Churn'] = imported_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return imported_data


def perform_eda(data_frame):
    '''
    perform eda on df and save figures to images folder
    input:
            data_frame: pandas dataframe
    output:
            None
    '''
    sns.set_style('whitegrid')
    eda_column_names = constants.EDA_COL_NAMES
    for column_name in eda_column_names:
        plt.figure(figsize=(20, 10))
        if column_name == "Heatmap":
            sns.heatmap(
                data_frame.corr(),
                annot=False,
                cmap='Dark2_r',
                linewidths=2)
        elif column_name == "Total_Trans_Ct":
            sns.histplot(data_frame['Total_Trans_Ct'])
        if column_name == "Churn":
            data_frame.Churn.hist()
        elif column_name == "Customer_Age":
            data_frame.Customer_Age.hist()
        elif column_name == "Marital_Status":
            data_frame['Marital_Status'].value_counts(
                "normalize").plot(kind="bar")
        elif column_name == "Total_Trans":
            sns.displot(data_frame['Total_Trans_Ct'])
        plt.savefig(f"{constants.EDA_FILEPATH}{column_name}.png")
        plt.close()


def encoder_helper(data_frame, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook
    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
    output:
            df: pandas dataframe with new columns for
    '''
    for col in category_lst:
        lst = []
        group = data_frame.groupby(col).mean()[response]
        for val in data_frame[col]:
            lst.append(group.loc[val])
        col_name = f"{col}_Churn"
        data_frame[col_name] = lst
    return data_frame


def perform_feature_engineering(data_frame, response):
    '''
    input:
              data_frame: pandas dataframe
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    X = pd.DataFrame()
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    X[keep_cols] = data_frame[keep_cols]
    y = data_frame[response]
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_data):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    y_train = y_data[0]
    y_test = y_data[1]
    y_train_preds_lr = y_data[2]
    y_train_preds_rf = y_data[3]
    y_test_preds_lr = y_data[4]
    y_test_preds_rf = y_data[5]
    model_type = ['Logistic Regression', 'Random Forest']
    for model in model_type:
        plt.rc('figure', figsize=(5, 5))
        if model == 'Logistic Regression':
            plt.text(0.01, 1.25, str('Logistic Regression Train'),
                     {'fontsize': 10}, fontproperties='monospace')
            plt.text(
                0.01, 0.05, str(
                    classification_report(
                        y_train, y_train_preds_lr)), {
                    'fontsize': 10}, fontproperties='monospace')
            plt.text(0.01, 0.6, str('Logistic Regression Test'), {
                     'fontsize': 10}, fontproperties='monospace')
            plt.text(
                0.01, 0.7, str(
                    classification_report(
                        y_test, y_test_preds_lr)), {
                    'fontsize': 10}, fontproperties='monospace')
            plt.axis('off')
            plt.savefig(f"{constants.RESULTS_FILEPATH}logistic_results.png")
        elif model == 'Random Forest':
            plt.text(0.01, 1.25, str('Random Forest Train'), {
                     'fontsize': 10}, fontproperties='monospace')
            plt.text(
                0.01, 0.05, str(
                    classification_report(
                        y_test, y_test_preds_rf)), {
                    'fontsize': 10}, fontproperties='monospace')
            plt.text(0.01, 0.6, str('Random Forest Test'), {
                     'fontsize': 10}, fontproperties='monospace')
            plt.text(
                0.01, 0.7, str(
                    classification_report(
                        y_train, y_train_preds_rf)), {
                    'fontsize': 10}, fontproperties='monospace')
            plt.axis('off')
            plt.savefig(f"{constants.RESULTS_FILEPATH}rf_results.png")

def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure
    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(25, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth + 'feature_importances_plot.png')
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    - random forest classifier & logistic regression
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              none
    '''
    # random forest classifier & logistic regression
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()
    # cross validation
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=constants.PARAM_GRID, cv=5)
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    # plot
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp.plot(ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(constants.RESULTS_FILEPATH + 'roc_plot.png')
    plt.close()
    response_data = [
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf]
    classification_report_image(y_data=response_data)
    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_test,
        constants.RESULTS_FILEPATH)
    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == "__main__":
    data_df = import_data(data_path=constants.DATA_PATH)
    perform_eda(data_df)
    encoded_data_df = encoder_helper(
        data_df, constants.CATEGORY_LST, constants.RESPONSE)
    x_train_, x_test_, y_train_, y_test_ = perform_feature_engineering(
        encoded_data_df, constants.RESPONSE)
    train_models(x_train_, x_test_, y_train_, y_test_)
