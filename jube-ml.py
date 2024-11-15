from __future__ import (print_function,
                        unicode_literals,
                        division,
                        annotations)
from textual.app import App, ComposeResult
from textual.css.query import NoMatches
from textual.widgets import Footer, Input, Select, SelectionList, RadioButton, RadioSet, Static, \
    Pretty, TabbedContent, DataTable, Switch
from textual.validation import Length
from textual.binding import Binding
from textual import events
from textual.containers import ScrollableContainer
from textual import on
import sqlite3
from enum import Enum
import re
from PIL import Image

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Type(Enum):
    classification = False
    regression = True


def get_tables(conn):
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    return [table[0] for table in tables]


def get_table_columns(conn, table="results"):
    smt = "select name from pragma_table_info('{table}');".format(table=table)
    print("smt:", smt)
    columns = conn.execute(smt).fetchall()
    return [c_name for c_name in columns]


def get_avail_experiments(conn):
    try:
        s_smt = """
            SELECT experiment_id, name, created_at, updated_at, algorithm, dataset, target
            FROM (
                SELECT *
                FROM sqml_experiments se
                INNER JOIN sqml_runs sr
                ON se.id = sr.experiment_id
                WHERE se.updated_at = sr.updated_at
            );
            """
        return conn.execute(s_smt).fetchall()
    except Exception as e:
        print(e)



class TabsApp(App[None]):
    def __init__(self, conn) -> None:
        self.conn = conn
        super().__init__()

    BINDINGS = [
        Binding(key="q", action="quit", description="Quit the app"),
        Binding(
            key="question_mark",
            action="help",
            description="Show help",
            key_display="?",
        )

    ]

    CSS = """
             Tabs {
                 dock: top;
             }
             
                ImageViewer{
        min-width: 8;
        min-height: 8;
    }

             ScrollableContainer {
                 layout: grid;
                 grid-size: 6;
                 align: left top;
             }

             Input {
                column-span: 2;
                 margin:1 1;
                 height:100%;
                 border: tall $primary;
                 content-align: center top;
             }

             Select {
             column-span: 4;
                 margin:1 1;
                 height: 100%;
                 background: $panel;
                 border: tall $primary;
                 content-align: center top;
             }

             RadioSet{
                column-span: 2;
                 margin:1 1;
                 height: 100%;
                 width: 100%;
                 background: $panel;
                 border: tall $primary;
             }

             Pretty{
                 column-span: 6;
                 margin:1 1;
                 background: $panel;
                 border: tall $primary;
                 content-align: center middle;
             }

             SelectionList {
                column-span: 2;
                 height:100%;
                 margin:1 1;
                 background: $panel;
                 border: tall $primary;
         }

         DataTable
         {       column-span: 6;
                 margin:1 1;
                 background: $panel;
                 border: tall $primary;
         }
         
         #table_sec{
                column-span: 1;
                 margin:1 1;
                 background: $panel;
                 border: tall $primary;
         }
         
        #table_sec_list{
                column-span: 2;
                 margin:1 1;
                 background: $panel;
                 border: tall $primary;
         }
         
        #t_sel_table{
               column-span: 2;
               
        }
        #t_task_name{
            column-span: 2;
        }
         
         
        Switch {
                column-span: 1;
                 margin:1 1;
                 height: 100%;
                 width: 100%;
                 background: $panel;
                 border: tall $primary;
        }
        
            Input.-valid {
        border: tall $success 60%;
    }
    Input.-valid:focus {
        border: tall $success;
    }

             """

    def on_key(self, event: events.Key) -> None:
        if event.key == "ctrl+p":
            pt = self.query_one(PredictionTab)
            pt.generate_predict_sql()
        elif event.key == "ctrl+t":
            tt = self.query_one(TrainingTab)
            tt.generate_ml_sql()

    def action_training(self):
        tt = self.query_one(TrainingTab)
        tt.generate_ml_sql()

    def action_predict(self):
        pt = self.query_one(PredictionTab)
        pt.generate_predict_sql()

    def action_pre_processing(self):
        pt = self.query_one(PrePrecessingTab)
        pt.create_new_table()

    def action_analysis(self):
        pt = self.query_one(AnalysisTab)
        pt.apply_lamda()

    def action_apply_pandas_func(self):
        pt = self.query_one(AnalysisTab)
        pt.apply_pandas_func()

    def compose(self) -> ComposeResult:
        # yield Header()
        with TabbedContent("Training", "Prediction", "Preprocessing", "Analysis"):
            yield TrainingTab(self.conn)
            yield PredictionTab(self.conn)
            yield PrePrecessingTab(self.conn)
            yield AnalysisTab(self.conn)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(TabbedContent).focus()

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if event.tab is None:
            pass
        else:
            if event.pane.id == "tab-2":
                self.bind(keys="ctrl+p", action="predict", description="Do prediction", key_display="ctr + p",
                          show=True)
                self.bind(keys="ctrl+t", action="training", description="Do training", key_display="ctr + t",
                          show=False)
                self.bind(keys="ctrl+o", action="pre_processing", description="Do Preprocessing", key_display="ctr + o",
                          show=False)
                self.bind(keys="ctrl+a", action="analysis", description="analysis", key_display="ctr + a",
                          show=False)
                self.refresh_bindings()
                ex_sec = self.query_one("#ex_sec", Select)
                ex_sec.clear()
                if get_avail_experiments(self.conn):
                    ex_sec.set_options((line, line) for line in format_data(get_avail_experiments(self.conn)))
                    ex_sec_list = self.query_one("#ex_sec_list", SelectionList)
                    ex_sec_list.clear_options()
            elif event.pane.id == "tab-1":
                self.bind(keys="ctrl+p", action="predict", description="Do prediction", key_display="ctr + p",
                          show=False)
                self.bind(keys="ctrl+t", action="training", description="Do training", key_display="ctr + t", show=True)
                self.bind(keys="ctrl+o", action="pre_processing", description="Do Preprocessing", key_display="ctr + o",
                          show=False)
                self.bind(keys="ctrl+a", action="analysis", description="analysis", key_display="ctr + a",
                          show=False)
                self.refresh_bindings()
                t_task_name = self.query_one("#t_task_name", Input)
                t_task_name.value=""
                t_sel_table = self.query_one("#t_sel_table", Select)
                t_sel_table.clear()
                t_sel_table.set_options((line, line) for line in get_tables(self.conn))
                t_sel_list_target = self.query_one("#t_sel_list_target", SelectionList)
                t_sel_list_target.clear_options()

            elif event.pane.id == "tab-3":
                self.bind(keys="ctrl+p", action="predict", description="Do prediction", key_display="ctr + p",
                          show=False)
                self.bind(keys="ctrl+t", action="training", description="Do training", key_display="ctr + t",
                          show=False)
                self.bind(keys="ctrl+o", action="pre_processing", description="Create new table", key_display="ctr + o",
                          show=True)
                self.bind(keys="ctrl+a", action="analysis", description="analysis", key_display="ctr + a",
                          show=False)
                self.refresh_bindings()

            elif event.pane.id == "tab-4":
                self.bind(keys="ctrl+p", action="predict", description="Do prediction", key_display="ctr + p",
                          show=False)
                self.bind(keys="ctrl+t", action="training", description="Do training", key_display="ctr + t",
                          show=False)
                self.bind(keys="ctrl+o", action="pre_processing", description="Create new table", key_display="ctr + o",
                          show=False)
                self.bind(keys="ctrl+a", action="analysis", description="analysis", key_display="ctr + a",
                          show=True)
                self.bind(keys="ctrl+h", action="apply_pandas_func", description="apply pandas function", key_display="ctr + h",
                          show=True)
                self.refresh_bindings()


class TrainingTab(Static):
    def __init__(self, conn) -> None:
        self.conn = conn
        super().__init__()

    test = "jo"
    LR = ['linear_regression', 'sgd', 'ridge', 'ridge_cv', 'elastic_net', 'elastic_net_cv', 'lasso', 'lasso_cv',
          'decision_tree', 'ada_boost', 'bagging', 'random_forest', 'gradient_boosting', 'knn', 'mlp', 'svr']

    RG = ['logistic_regression', 'sgd', 'ridge', 'ridge_cv', 'decision_tree', 'ada_boost', 'bagging',
          'decision_tree', 'ada_boost', 'bagging', 'gradient_boosting', 'random_forest', 'knn', 'mlp', 'svc']
    ml_config = {'name': "", 'type': Type.regression, 'algo': "linear_regression", 'table': "results", 'target': []}

    def generate_ml_sql(self):
        from sqlite_ml.sqml import SQML
        # setup sqlite-ml extension
        sqml = SQML()
        sqml.setup_schema(self.conn)
        sqml.register_functions(self.conn)
        slist = self.query_one("#t_sel_list_target", SelectionList)
        if len(self.ml_config['target']) < 1:
            slist.styles.border_title_color = "red"
        else:
            slist.styles.border_title_color = "green"
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
                                  """.format(name=self.ml_config['name'], type=self.ml_config['type'].name,
                                             algo=self.ml_config['algo'],
                                             table=self.ml_config['table'], target=self.ml_config['target'][0])
            print(smt)
            results = self.conn.execute(smt).fetchone()[0]
            try:
                self.query_one("#sc_pre", ScrollableContainer)
            except NoMatches:
                self.mount(ScrollableContainer(id="sc_pre"))
            sc_pre = self.query_one("#sc_pre", ScrollableContainer)
            try:
                self.query_one("#pretty_result", Pretty)
            except NoMatches:
                sc_pre.mount(Pretty("", id="pretty_result"))
            pretty_result = self.query_one("#pretty_result", Pretty)
            pretty_result.update(results)
            pretty_result.border_title = "Training Output:"

    def compose(self) -> ComposeResult:
        yield ScrollableContainer(Select(id="t_sel_table", options=[]),SelectionList[int](id="t_sel_list_target"),
                                  RadioSet(RadioButton("Regression", value=self.ml_config['type'].value),
                                           RadioButton("Classification", value=not self.ml_config['type'].value))
                                  )

        yield ScrollableContainer(Input(id="t_task_name", placeholder="Give a name for the ML task...", type="text", value=self.ml_config['name']), Select(id="t_sel_algo", options=[]))

    def on_mount(self) -> None:
        self.query_one("#t_task_name", Input).border_title = "ML Task name:"
        self.query_one(RadioSet).border_title = "ML Task Type:"
        t_sel_algo = self.query_one("#t_sel_algo", Select)
        t_sel_algo.set_options((line, line) for line in self.LR)
        t_sel_algo.border_title = "ML Task Algorithm:"

        t_sel_table = self.query_one("#t_sel_table", Select)
        t_sel_table.set_options((line, line) for line in get_tables(self.conn))
        t_sel_table.border_title = "Select data source"
        self.query_one("#t_sel_list_target", SelectionList).border_title = "target selection"

    @on(Input.Changed)
    def show_invalid_reasons(self, event: Input.Changed) -> None:
        self.ml_config['name'] = event.value
        print("hi")
        select = self.query_one("#t_sel_algo", Select)
        print(select.value)
        print(self.ml_config['algo'] )

    @on(Select.Changed, "#t_sel_algo")
    def select_changed_algo(self, event: Select.Changed) -> None:
        self.ml_config['algo'] = event.value
        select_t = self.query_one("#t_sel_list_target", SelectionList)
        select_t.disabled = False

    @on(Select.Changed, "#t_sel_table")
    def select_changed(self, event: Select.Changed) -> None:
        self.ml_config['table'] = event.value
        tabs = get_table_columns(self.conn, self.ml_config['table'])
        self.query_one("#t_sel_list_target", SelectionList).clear_options()
        self.query_one("#t_sel_list_target", SelectionList).add_options([(name[0], idx, False) for idx, name in enumerate(tabs)])

    @on(RadioSet.Changed)
    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if self.ml_config['type'] == Type.regression:
            self.ml_config['type'] = Type.classification
        else:
            self.ml_config['type'] = Type.regression
        select = self.query_one("#t_sel_algo", Select)
        if self.ml_config['type'].value:
            select.set_options((line, line) for line in self.LR)
        else:
            select.set_options((line, line) for line in self.RG)

    @on(SelectionList.SelectedChanged)
    def update_selected_view(self) -> None:
        select_list = self.query_one("#t_sel_list_target", SelectionList)
        self.ml_config['target'] = [str(select_list.get_option_at_index(selected_item).prompt) for selected_item in select_list.selected]


def format_data(sql_results):
    formatted_list = []
    for entry in sql_results:
        experiment_id, name, c_date, u_date, algorithm, dataset, target = entry
        formatted_string = f"ID: {experiment_id} / NAME: {name} / Algorithm: {algorithm} / Dataset: {dataset} / Target: {target} - DATE: [{c_date}::{u_date}]"
        formatted_list.append(formatted_string)
    return formatted_list


def parse_experiment_to_dic(s):
    print("error_s: ", s)
    pattern = r"ID: (\d+) / NAME: ([\w\s\-_]+) / Algorithm: (\w+) / Dataset: ([\w\-_]+) / Target: ([\w\-_]+) - DATE: \[(.*?)::(.*?)\]"
    match = re.match(pattern, s)
    if match:
        return {
            "ex_id": match.group(1),
            "ex_name": match.group(2),
            "algorithm": match.group(3),
            "dataset": match.group(4),
            "target": match.group(5),
            "create_date": match.group(6),
            "update_date": match.group(7)
        }
    else:
        raise ValueError("String does not match the expected format")


class PredictionTab(Static):
    def __init__(self, conn) -> None:
        self.conn = conn
        super().__init__()

    predict_config = {}
    rows = []

    def generate_predict_sql(self):
        from sqlite_ml.sqml import SQML
        # setup sqlite-ml extension
        sqml = SQML()
        sqml.setup_schema(self.conn)
        sqml.register_functions(self.conn)
        sub_smt = ""
        if len(self.predict_config['features']) > 1:
            sub_smt = "json_object(" + ", ".join(
                re.escape("'") + f[0] + re.escape("'") + ', [' + f[0] + ']' for f in
                self.predict_config['features'][:-1]) + ',' + re.escape("'") + \
                      self.predict_config['features'][-1][0] + re.escape("'") + ', [' + self.predict_config['features'][-1][
                          0] + '])'
        else:
            sub_smt = "json_object(" + ", ".join( re.escape("'") + f[0] + re.escape("'") + ', [' + f[0] + '])' for f in self.predict_config['features'])
        smt_batched = """
        SELECT
          '{table}'.*,
          batch.value AS prediction
        FROM
          '{table}'
          JOIN json_each (
            (
              SELECT
                sqml_predict_batch(
                  '{name}',
                  json_group_array(
                  {features}
                  )
                )
              FROM
                '{table}'
            )
          ) batch ON (batch.rowid + 1) = '{table}'.rowid;
            """.format(name=self.predict_config['ex_name'], table=self.predict_config['dataset'], features=sub_smt,
                       target=self.predict_config['target'])
        print("sql: ", smt_batched)
        results = self.conn.execute(smt_batched).fetchall()
        get_table_columns(self.conn, self.predict_config['dataset'])
        header = [name[0] for name in get_table_columns(self.conn, self.predict_config['dataset'])] + ["prediction"]
        self.rows.append(tuple(header))
        for idx, row in enumerate(results):
            self.rows.append(tuple(row))
        try:
            self.query_one("#sc_dt", ScrollableContainer)
        except NoMatches:
            self.mount(ScrollableContainer(id="sc_dt"))
        sc_dt = self.query_one("#sc_dt", ScrollableContainer)
        try:
            self.query_one("#dt", DataTable)
        except NoMatches:
            sc_dt.mount(DataTable(id="dt"))
        table = self.query_one("#dt", DataTable)
        table.add_columns(*self.rows[0])
        table.add_rows(self.rows[1:])
        table.border_title = "Prediction Results:"

        df = pd.DataFrame(self.rows[1:], columns=self.rows[0])
        sns.pairplot(df)
        plt.show()

        plt.figure(figsize=(10, 8))
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        plt.show()
    """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot
        ax.scatter(df['feature1'], df['feature2'], df['target'], c='r', marker='o')

        # Labels
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Target')

        plt.title('3D Scatter Plot')
        plt.show()
    """
    def compose(self) -> ComposeResult:
        yield ScrollableContainer(Select([], id="ex_sec"), SelectionList(id="ex_sec_list"))

    def on_mount(self) -> None:
        select = self.query_one("#ex_sec", Select)
        select.border_title = "Model selection"
        sec_list = self.query_one('#ex_sec_list', SelectionList)
        sec_list.border_title = "Selected features "
        sec_list.add_options([])
        sec_list.disabled = True

    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        if event.value != Select.BLANK:
            try:
                self.query_one(DataTable)
            except NoMatches:
                pass
            else:
                self.query_one(DataTable).remove()
            s_string = event.value
            select_ex = parse_experiment_to_dic(s_string)
            self.predict_config = select_ex
            f_list = [(name[0], idx, True) for idx, name in enumerate(get_table_columns(self.conn, self.predict_config['dataset']))]
            self.predict_config['features'] = [f for f in f_list if str(f[0]) != select_ex['target']]
            sec_list = self.query_one('#ex_sec_list', SelectionList)
            sec_list.clear_options()
            sec_list.add_options(self.predict_config['features'])
            sec_list.disabled = True
            self.rows = []
            try:
                table = self.query_one(DataTable)
                table.remove()
            except NoMatches:
                pass

            sec_list.disabled = False

    @on(SelectionList.SelectedChanged)
    def update_selected_view(self) -> None:
        selected_items = self.query_one('#ex_sec_list', SelectionList).selected
        # self.predict_config = [f_list[idx][0] for idx in selected_items] if selected_items else []


class PrePrecessingTab(Static):
    selected_columns = []
    table = {'name': "", 'column_names': ['WRbwmaxMiB', 'WRbwminMiB'], "parent_table": "results"}
    df = pd.DataFrame()

    def __init__(self, conn) -> None:
        self.conn = conn
        super().__init__()

    def create_new_table(self, clean=True):
        smt = ""
        if clean:
            where_clause = ' AND '.join([f'{column} IS NOT NULL' for column in self.table['column_names']])
            smt = "CREATE TABLE '{t_name}' AS SELECT {t_columns} FROM {t_parent_table} WHERE {wc}".format(
                t_name=self.table['name'],
                t_columns=', '.join(
                    self.table['column_names']),
                t_parent_table=self.table[
                    'parent_table'],
                wc=where_clause)
        else:
            smt = "CREATE TABLE '{t_name}' AS SELECT {t_columns} FROM {t_parent_table}".format(t_name=self.table['name'],
                                                                                             t_columns=', '.join(
                                                                                                 self.table[
                                                                                                     'column_names']),
                                                                                             t_parent_table=self.table[
                                                                                                 'parent_table'])
        print("clean:", smt)
        self.conn.execute(smt)
        new_smt= "SELECT name FROM sqlite_master WHERE type='table' AND name='{t_name}';".format(t_name= self.table['name'])
        result = self.conn.execute(new_smt).fetchone()
        if result:
            self.table['parent_table'] = self.table['name']
            print(f"Table {self.table['name']} was created successfully.")
            select = self.query_one("#table_sec", Select)
            select.clear()
            select.set_options((line, line) for line in get_tables(self.conn))
            table_sec_list = self.query_one('#table_sec_list', SelectionList)
            table_sec_list.clear_options()
            dv_pretty = self.query_one('#dv_pretty', Pretty)
            dv_pretty.update("")
            return self.table['name']
        else:
            print(f"Failed to create table {self.table['name']}.")
            print("new-table:", smt)
            return None

    def compose(self) -> ComposeResult:
        yield ScrollableContainer(Input(id="new_tab_name", value=self.table['name'], placeholder="Give a name...",
                                        validators=Length(minimum=4, maximum=100)), Select([], id="table_sec"),
                                  SelectionList(id="table_sec_list"), Switch(value=True, id="sw_clean"))
        yield ScrollableContainer(Pretty(id="dv_pretty", object=None))

    def on_mount(self) -> None:
        input = self.query_one("#new_tab_name", Input)
        input.border_title = "Name for the new table:"
        select = self.query_one("#table_sec", Select)
        select.border_title = "Origin table:"
        select.set_options((line, line) for line in get_tables(self.conn))
        table_sec_list = self.query_one('#table_sec_list', SelectionList)
        table_sec_list.border_title = "Selected columns:"
        sw_clean = self.query_one("#sw_clean", Switch)
        sw_clean.border_title = "Do cleaning"
        dv_pretty = self.query_one("#dv_pretty", Pretty)
        dv_pretty.border_title = "Data overview:"


    @on(Input.Changed)
    def input_changed(self, event: Input.Changed) -> None:
        self.table['name'] = event.value

    @on(Select.Changed, "#table_sec")
    def select_changed(self, event: Select.Changed) -> None:
        if event.value and (event.value != Select.BLANK):
            self.table['parent_table'] = event.value
            col_names = get_table_columns(self.conn, self.table['parent_table'])
            sl = self.query_one("#table_sec_list", SelectionList)
            sl.clear_options()
            sl.add_options([(name[0], idx, True) for idx, name in enumerate(col_names)])
            print("pt_:",  self.table['parent_table'] )
            smt = "SELECT * from '{table}'".format(table=self.table['parent_table'])
            self.df = pd.read_sql_query(smt, self.conn)
            dv_pretty = self.query_one("#dv_pretty", Pretty)
            dv_pretty.update(self.df.describe())

    @on(SelectionList.SelectedChanged)
    def update_selected_view(self) -> None:
        select_list = self.query_one("#table_sec_list", SelectionList)
        self.selected_columns = [str(select_list.get_option_at_index(selected_item).prompt) for selected_item in
                                 select_list.selected]
        self.table['column_names'] = self.selected_columns
        clean_sw = self.query_one(Switch)
        if clean_sw.value:
            self.df = self.df.dropna(subset=self.selected_columns)
        dv_pretty = self.query_one("#dv_pretty", Pretty)
        print("selected : ", self.selected_columns)
        dv_pretty.update(self.df[self.selected_columns].describe())


class AnalysisTab(Static):
    selected_columns = []
    table = {'name': "", 'column_names': ['WRbwmaxMiB', 'WRbwminMiB'], "parent_table": "results", "lambda": "lambda x: x < 200"}
    df = pd.DataFrame()

    def __init__(self, conn) -> None:
        self.conn = conn
        super().__init__()

    def apply_lamda_to_data(self):
        input = self.query_one("#a_lamda", Input)
        print("a_lamda: ", input.value)
        lambda_func = eval(input.value)
        self.df[self.selected_columns] = self.df[self.selected_columns].apply(lambda_func)
        dv_pretty = self.query_one("#a_dv_pretty", Pretty)
        dv_pretty.update(self.df[self.selected_columns])

    def apply_lamda(self):
        input = self.query_one("#a_lamda", Input)
        print("a_lamda: ", input.value)
        lambda_func = eval(input.value)
        filtered_condition = self.df[self.selected_columns].apply(lambda_func)
        dv_pretty = self.query_one("#a_dv_pretty", Pretty)
        dv_pretty.update(self.df[filtered_condition])

    def apply_pandas_func(self):
        # input = self.query_one("#a_lamda", Input)
        exec_code = f"self.df[self.selected_columns]." + self.table['lambda']
        print("apply_pandas_func: ", exec_code)
        result = eval(exec_code)
        dv_pretty = self.query_one("#a_dv_pretty", Pretty)
        dv_pretty.update(self.df[result])

    def compose(self) -> ComposeResult:
        yield ScrollableContainer(Input(id="a_lamda", value=self.table['lambda'], placeholder="apply lamda function..."), Select([], id="a_table_sec"))
        yield ScrollableContainer(SelectionList(id="a_table_sec_list"), Pretty(id="a_dv_pretty", object=None))

    def on_mount(self) -> None:
        select = self.query_one("#a_table_sec", Select)
        select.border_title = "Origin table:"
        select.set_options((line, line) for line in get_tables(self.conn))
        table_sec_list = self.query_one('#a_table_sec_list', SelectionList)
        table_sec_list.border_title = "Selected columns:"
        dv_pretty = self.query_one("#a_dv_pretty", Pretty)
        dv_pretty.border_title = "Data overview:"
        dv_pretty.styles.height = "100%"
        dv_pretty.styles.overflow_y ="scroll"

    @on(Input.Changed, "#a_lamda")
    def input_changed(self, event: Input.Changed) -> None:
        self.table['lambda'] = event.value

    @on(Select.Changed, "#a_table_sec")
    def select_changed(self, event: Select.Changed) -> None:
        if event.value and (event.value != Select.BLANK):
            self.table['parent_table'] = event.value
            col_names = get_table_columns(self.conn, self.table['parent_table'])
            sl = self.query_one("#a_table_sec_list", SelectionList)
            sl.clear_options()
            sl.add_options([(name[0], idx, True) for idx, name in enumerate(col_names)])
            print("pt_:",  self.table['parent_table'] )
            smt = "SELECT * from '{table}'".format(table=self.table['parent_table'])
            self.df = pd.read_sql_query(smt, self.conn)
            dv_pretty = self.query_one("#a_dv_pretty", Pretty)
            dv_pretty.update(self.df.describe())

    @on(SelectionList.SelectedChanged)
    def update_selected_view(self) -> None:
        select_list = self.query_one("#a_table_sec_list", SelectionList)
        self.selected_columns = [str(select_list.get_option_at_index(selected_item).prompt) for selected_item in
                                 select_list.selected]
        self.table['column_names'] = self.selected_columns
        dv_pretty = self.query_one("#a_dv_pretty", Pretty)
        print("selected : ", self.selected_columns)
        dv_pretty.update(self.df[self.selected_columns].describe())


if __name__ == "__main__":
    #conn = sqlite3.connect("/Users/flash/Desktop/DBs/scale_3056_database.dat")
    conn = sqlite3.connect("/Users/flash/Desktop/DBs/corona-striping.dat")
    app = TabsApp(conn)
    app.run()
