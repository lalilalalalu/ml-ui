from __future__ import (print_function,
                        unicode_literals,
                        division)
import sqlite3

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report

from enum import Enum
from textual.app import App, ComposeResult
from textual.widgets import Footer, Label, Tabs, Input, Select, SelectionList, Button, RadioButton, RadioSet, Static
from textual.containers import ScrollableContainer
from textual import on

Steps = [
    "Training",
    "Predication",
]

LINES = """linear_regression
sgd
ridge
ridge_cv
elastic_net
elastic_net_cv
lasso
lasso_cv
decision_tree
ada_boost
bagging
random_forest
gradient_boosting
knn
mlp
svr""".splitlines()

LINES2 = """TEST
sgd
ridge
ridge_cv
elastic_net
elastic_net_cv
lasso
lasso_cv
decision_tree
ada_boost
bagging
random_forest
gradient_boosting
knn
mlp
svr""".splitlines()


def linear_regression(dependent_variable_names, independent_variable_name, dataframe, show_dependency=False):
    # creating feature variables
    dependent_variable_name = dependent_variable_names[0]
    independent_variable_name = independent_variable_name[0]
    print("target name - ", dependent_variable_name)
    print("feature name - ", independent_variable_name)
    print("--------------------------------------------------------------------------------------------------------\n")
    y = dataframe[dependent_variable_name]
    X = dataframe[[independent_variable_name]]
    print(X)
    print(y)
    if True:
        sns.regplot(x=independent_variable_name, y=dependent_variable_name, data=dataframe).set(
            title=f'Regression plot of {independent_variable_name} and {dependent_variable_name}')
        plt.show()
        plt.savefig("/home/zhazhu/repos/JUBE/test.png")
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=101)
    # creating a regression model
    model = LinearRegression()
    # fitting the model
    model.fit(X_train, y_train)
    print(model.coef_)
    # making predictions
    predictions = model.predict(X_test)
    # model evaluation
    print(
        'mean_squared_error : ', mean_squared_error(y_test, predictions))
    print(
        'mean_absolute_error : ', mean_absolute_error(y_test, predictions))


def multiple_linear_regression(dependent_variable_names, independent_variable_names, dataframe, show_dependency=False):
    dependent_variable_name = dependent_variable_names[0]
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
                                                        test_size=0.5,
                                                        random_state=101)
    # creating a regression model
    model = LinearRegression()
    # fitting the model
    model.fit(X_train, y_train)
    print(model.coef_)
    # making predictions
    predictions = model.predict(X_test)
    print(model.predict())
    # model evaluation
    print(
        'mean_squared_error : ', mean_squared_error(y_test, predictions))
    print(
        'mean_absolute_error : ', mean_absolute_error(y_test, predictions))


def binary_logistic_regression(dependent_variable_name, independent_variable_names, dataframe, show_dependency=False):
    df_reduced = dataframe[[dependent_variable_name] + independent_variable_names]
    # creating feature variables
    y = dataframe[dependent_variable_name]
    X = dataframe[independent_variable_names]
    print(X)
    print(y)
    if show_dependency:
        correlations = df_reduced.corr()
        sns.heatmap(correlations, annot=True).set(title='Heat map of Pearson Correlations');
        plt.show()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=101)
    # creating a regression model
    model = LogisticRegression()
    # fitting the model
    model.fit(X_train, y_train)
    print(model.coef_)
    # making predictions
    predictions = model.predict(X_test)
    # model evaluation
    print(
        'mean_squared_error : ', mean_squared_error(y_test, predictions))
    print(
        'mean_absolute_error : ', mean_absolute_error(y_test, predictions))
    print(classification_report(y_test, predictions))


def test(args=None, data=None, mlsql=False, test=True):
    if test:
        db_connect_str = str(data[0]._db_file)
        print("set sql connection to %s" % (db_connect_str))
        conn = sqlite3.connect(db_connect_str)
        print("The connection to the database %s is established" % (db_connect_str))
        TabsApp(conn=conn).run()
    else:
        print("hi", args)
        if data != None:
            db_connect_str = str(data[0]._db_file)
            print("set sql connection to %s" % (db_connect_str))
            conn = sqlite3.connect(db_connect_str)
            print("The connection to the database %s is established" % (db_connect_str))
            cur = conn.cursor()
            # Only for the Demo
            for row in cur.execute("SELECT * FROM results"):
                print(row)

            if args.cleaning and len(args.cleaning) != 0:
                print("cleaning \n")
                cur.execute(args.cleaning[0])
                print("cleaned \n")
            print(
                "--------------------------------------------------------------------------------------------------------\n")
            for row in cur.execute("SELECT * FROM results"):
                print(row)
            if args.features and args.target and args.model:
                df = pd.read_sql_query("SELECT * FROM results", conn)

                indep = 'RDiopsmeanOPS'
                dp = 'RDbwmaxMiB'
                if args.model[0] == "linear_regression":
                    print('---------------linear regression----------------\n')
                    linear_regression(args.target, args.features, df, show_dependency=True)
                elif args.model[0] == "logistic_regression":
                    print('----------------logistic regression----------------\n')
                    binary_logistic_regression(args.target, args.features, df, show_dependency=True)
                else:
                    print("not implemented yet!\n")

            if mlsql:
                from sqlite_ml.sqml import SQML
                # setup sqlite-ml extension
                sqml = SQML()
                sqml.setup_schema(conn)
                sqml.register_functions(conn)
                while True:
                    print()
                    print(conn.execute("""SELECT sqml_train(
                    'mytest',
                    'regression',
                    'linear_regression',
                    'resultsMY',
                    'RDbwmaxMiB'
                    ) AS training""").fetchone()[0])

                    print(conn.execute("""SELECT sqml_predict(
            'mytest',
            (
                SELECT json_array([WRbwmaxMiB], [WRbwminMiB], [WRbwmeanMiB], [WRiopsmaxOPS], [WRiopsminOPS], [WRiopsmeanOPS], [RDbwmaxMiB], [RDbwminMiB], [RDbwmeanMiB], [RDiopsmaxOPS],[RDiopsminOPS],[RDiopsmeanOPS])
                FROM resultsMY
                LIMIT 1
            )
        ) AS prediction;""").fetchone()[0])
                # execute sqlite-ml functions


class Type(Enum):
    classification = False
    regression = True


class TabsApp(App[None]):

    def __init__(self, conn) -> None:
        print(conn)
        self.conn = conn
        super().__init__()

    ml_config = {'name': "Demo", 'type': Type.regression, 'algo': "linear_regression", 'table': "resultsMY",
                 'target': "WRbwmaxMiB"}

    def generate_ml_sql(self, is_training=True):
        from sqlite_ml.sqml import SQML
        # setup sqlite-ml extension
        sqml = SQML()
        sqml.setup_schema(conn)
        sqml.register_functions(conn)
        # execute sqlite-ml functions
        print(conn.execute("SELECT sqml_python_version();").fetchone()[0])
        if is_training:
            smt = """
            SELECT sqml_train(
                '{name}',
                '{type}',
                '{algo}',
                '{table}',
                '{target}'
            )
            AS
            training;
            """.format(name=self.ml_config['name'], type=self.ml_config['type'].name, algo=self.ml_config['algo'],
                       table=self.ml_config['table'], target=self.ml_config['target'])
            print(smt)

    CSS = """
    Tabs {
        dock: top;
    }
    Screen {
        layout: grid;
        grid-size: 1;
        align: center top;
    }

    ScrollableContainer {
        layout: grid;
        grid-size: 2;
        align: left top;
    }
    Input {
        margin:1 1;
        height:100%;
        background: $panel;
        border: tall $primary;
        content-align: center top;
    }

    Select {
        margin:1 1;
        height: 100%;
        background: $panel;
        border: tall $primary;
        content-align: center top;
    }

    RadioSet{
        margin:1 1;
        height: 100%;
        width: 100%;
        background: $panel;
        border: tall $primary;
        content-align: center middle;
    }

    SelectionList {
     margin:1 1;
        background: $panel;
        border: tall $primary;
}

 #training-bt {
    margin: 1 1;
    width: 1fr;
    height: auto;

 }

    #two {
    column-span: 2;
    tint: magenta 40%;
}
    """

    def compose(self) -> ComposeResult:
        yield Tabs(Steps[0], Steps[1])
        yield Input(placeholder="ML Workflow name", type="text", value=self.ml_config['name'])
        yield ScrollableContainer(RadioSet(RadioButton("Regression", value=self.ml_config['type'].value),
                                           RadioButton("Classification", value=not self.ml_config['type'].value)),
                                  Select((line, line) for line in LINES))
        yield ScrollableContainer(SelectionList[int](id="test"), SelectionList[int](id="test2"))
        yield Button(id="training-bt", label="Training")
        yield Footer()

    def on_mount(self) -> None:
        c_names = self.conn.execute("select name from pragma_table_info('results');").fetchall()
        f_list = [(name[0], idx, False) for idx, name in enumerate(c_names)]
        t_list = [(name[0], idx, False) for idx, name in enumerate(c_names)]
        self.query_one(Tabs).focus()
        self.query_one(Input).border_title = "ML Task name:"
        self.query_one(RadioSet).border_title = "ML Task Type:"
        self.query_one(Select).border_title = "ML Task Algorithm:"
        self.query_one("#test", SelectionList).border_title = "feature selection"
        self.query_one("#test", SelectionList).add_options(f_list)
        self.query_one("#test", SelectionList).visible = False
        self.query_one("#test2", SelectionList).border_title = "target selection"
        self.query_one("#test2", SelectionList).add_options(t_list)
        self.query_one("#test2", SelectionList).visible = False

    @on(Input.Changed)
    def show_invalid_reasons(self, event: Input.Changed) -> None:
        self.ml_config['name'] = event.value

    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        self.ml_config['algo'] = event.value
        select_t = self.query_one("#test2", SelectionList)
        select_t.visible = True
        select_f = self.query_one("#test", SelectionList)
        select_f.visible = True

    @on(RadioSet.Changed)
    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        print("toggle_change:", self.ml_config['type'])
        if self.ml_config['type'] == Type.regression:
            self.ml_config['type'] = Type.classification
        else:
            self.ml_config['type'] = Type.regression
        print("toggle_change:", self.ml_config['type'])
        select = self.query_one(Select)
        if self.ml_config['type'].value:
            select.set_options((line, line) for line in LINES)
        else:
            select.set_options((line, line) for line in LINES2)

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        if event.tab is None:
            # When the tabs are cleared, event.tab will be None
            print("is None")
        else:
            print("is tab", event.tab)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.generate_ml_sql()

