import sqlite3
from sqlite_ml.sqml import SQML
import pytermgui as ptg
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report

def multiple_linear_regression(dependent_variable_name, independent_variable_names, dataframe, show_dependency=False):
    dependent_variable_name = dependent_variable_name
    test = [dependent_variable_name] + independent_variable_names
    df_reduced = dataframe[[dependent_variable_name] + independent_variable_names]
    # creating feature variables
    Y = dataframe[dependent_variable_name]
    X = dataframe[independent_variable_names]
    print(X)
    print(Y)
    if show_dependency:
        correlations = df_reduced.corr()
        sns.heatmap(correlations, annot=True).set(title='Heat map of Pearson Correlations');
        plt.show()
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.25,
                                                        shuffle="shuffle")
    # creating a regression model
    model = LinearRegression()
    # fitting the model
    model.fit(X_train, y_train)
    model.predict(X_test)
    score = model.score(X_test, y_test)
    """
    print(model.coef_)
    # making predictions
    predictions = model.predict(X_test)

    # model evaluation
    print(
        'mean_squared_error : ', mean_squared_error(y_test, predictions))
    print(
        'mean_absolute_error : ', mean_absolute_error(y_test, predictions))
    """


if __name__ == '__main__':
    conn = sqlite3.connect("/Users/flash/Desktop/real_result_database.dat")
    df = pd.read_sql_query("SELECT * FROM results", conn)
    indep = ['WRbwmaxMiB', 'WRbwminMiB', 'WRbwmeanMiB', 'WRiopsmaxOPS', 'WRiopsminOPS', 'WRiopsmeanOPS', 'RDbwmaxMiB', 'RDbwminMiB', 'RDbwmeanMiB', 'RDiopsmaxOPS', 'RDiopsminOPS', 'RDiopsmeanOPS']
    dp = 'RDbwmaxMiB'
    multiple_linear_regression(dependent_variable_name=dp, independent_variable_names=indep, dataframe=df, show_dependency=True)

