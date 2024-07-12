from __future__ import (print_function,
                        unicode_literals,
                        division,
                        annotations)
from textual.app import App, ComposeResult
from textual.css.query import NoMatches
from textual.widgets import Footer, Input, Select, SelectionList, RadioButton, RadioSet, Static, \
    Pretty, TabbedContent, DataTable, Switch
from textual.validation import Function, Length, ValidationResult, Validator
from textual.binding import Binding
from textual import events
from textual.containers import ScrollableContainer
from textual import on
import sqlite3
from enum import Enum
import re

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


class TabsApp(App[None]):
    table_name = "results"

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

    def compose(self) -> ComposeResult:
        # yield Header()
        c_names = get_table_columns(self.conn, self.table_name)
        with TabbedContent("Training", "Prediction", "Preprocessing", "Data Overview"):
            yield TrainingTab(self.conn)
            yield PredictionTab(self.conn, c_names)
            yield PrePrecessingTab(self.conn)
            yield DataViewTab(self.conn)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(TabbedContent).focus()
        print("tabless:", get_tables(self.conn))

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
                self.refresh_bindings()
            elif event.pane.id == "tab-1":
                self.bind(keys="ctrl+p", action="predict", description="Do prediction", key_display="ctr + p",
                          show=False)
                self.bind(keys="ctrl+t", action="training", description="Do training", key_display="ctr + t", show=True)
                self.bind(keys="ctrl+o", action="pre_processing", description="Do Preprocessing", key_display="ctr + o",
                          show=False)
                self.refresh_bindings()
            elif event.pane.id == "tab-3":
                self.bind(keys="ctrl+p", action="predict", description="Do prediction", key_display="ctr + p",
                          show=False)
                self.bind(keys="ctrl+t", action="training", description="Do training", key_display="ctr + t",
                          show=False)
                self.bind(keys="ctrl+o", action="pre_processing", description="Do Preprocessing", key_display="ctr + o",
                          show=True)
                self.refresh_bindings()


class TrainingTab(Static):
    def __init__(self, conn) -> None:
        self.conn = conn
        super().__init__()

    LR = ['linear_regression', 'sgd', 'ridge', 'ridge_cv', 'elastic_net', 'elastic_net_cv', 'lasso', 'lasso_cv',
          'decision_tree', 'ada_boost', 'bagging', 'random_forest', 'gradient_boosting', 'knn', 'mlp', 'svr']

    RG = ['logistic_regression', 'sgd', 'ridge', 'ridge_cv', 'decision_tree', 'ada_boost', 'bagging',
          'decision_tree', 'ada_boost', 'bagging', 'gradient_boosting', 'random_forest', 'knn', 'mlp', 'svc']
    ml_config = {'name': "Demo", 'type': Type.regression, 'algo': "linear_regression", 'table': "results", 'target': []}

    def generate_ml_sql(self):
        from sqlite_ml.sqml import SQML
        # setup sqlite-ml extension
        sqml = SQML()
        sqml.setup_schema(self.conn)
        sqml.register_functions(self.conn)
        slist = self.query_one("#test2", SelectionList)
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
        yield ScrollableContainer(Select(id="t_sel_table", options=[]),SelectionList[int](id="test2"),
                                  RadioSet(RadioButton("Regression", value=self.ml_config['type'].value),
                                           RadioButton("Classification", value=not self.ml_config['type'].value))
                                  )

        yield ScrollableContainer(Input(id="t_task_name", placeholder="ML Workflow name", type="text", value=self.ml_config['name']), Select(id="t_sel_algo", options=[]))

    def on_mount(self) -> None:
        self.query_one("#t_task_name", Input).border_title = "ML Task name:"
        self.query_one(RadioSet).border_title = "ML Task Type:"
        t_sel_algo = self.query_one("#t_sel_algo", Select)
        t_sel_algo.set_options((line, line) for line in self.LR)
        t_sel_algo.border_title = "ML Task Algorithm:"

        t_sel_table = self.query_one("#t_sel_table", Select)
        t_sel_table.set_options((line, line) for line in get_tables(self.conn))
        t_sel_table.border_title = "Select data source"
        self.query_one("#test2", SelectionList).border_title = "target selection"

    @on(Input.Changed)
    def show_invalid_reasons(self, event: Input.Changed) -> None:
        self.ml_config['name'] = event.value

    @on(Select.Changed, "#t_sel_algo")
    def select_changed(self, event: Select.Changed) -> None:
        self.ml_config['algo'] = event.value
        select_t = self.query_one("#test2", SelectionList)
        select_t.disabled = False

    @on(Select.Changed, "#t_sel_table")
    def select_changed(self, event: Select.Changed) -> None:
        self.ml_config['table'] = event.value
        tabs = get_table_columns(self.conn, self.ml_config['table'])
        self.query_one("#test2", SelectionList).clear_options()
        self.query_one("#test2", SelectionList).add_options([(name[0], idx, False) for idx, name in enumerate(tabs)])

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
        select_list = self.query_one("#test2", SelectionList)
        self.ml_config['target'] = [str(select_list.get_option_at_index(selected_item).prompt) for selected_item in select_list.selected]


def format_data(sql_results):
    formatted_list = []
    for entry in sql_results:
        experiment_id, name, c_date, u_date, algorithm, dataset, target = entry
        formatted_string = f"ID: {experiment_id} / NAME: {name} / Algorithm: {algorithm} / Dataset: {dataset} / Target: {target} - DATE: [{c_date}::{u_date}]"
        formatted_list.append(formatted_string)
    return formatted_list


def parse_experiment_to_dic(s):
    pattern = r"ID: (\d+) / NAME: ([\w\s]+) / Algorithm: (\w+) / Dataset: (\w+) / Target: ([\w]+) - DATE: \[(.*?)::(.*?)\]"
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
    def __init__(self, conn, c_name) -> None:
        self.conn = conn
        super().__init__()

    predict_config = {}
    rows = []

    def get_avail_experiments(self):
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
        return self.conn.execute(s_smt).fetchall()

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
          {table}.*,
          batch.value AS prediction,
          {table}.{target} = batch.value AS match
        FROM
          {table}
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
                {table}
            )
          ) batch ON (batch.rowid + 1) = {table}.rowid
        WHERE match = True;
            """.format(name=self.predict_config['ex_name'], table=self.predict_config['dataset'], features=sub_smt,
                       target=self.predict_config['target'])
        results = self.conn.execute(smt_batched).fetchall()
        get_table_columns(self.conn, self.predict_config['dataset'])
        header = [name[0] for name in get_table_columns(self.conn, self.predict_config['dataset'])] + ["prediction", "match"]
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

    def compose(self) -> ComposeResult:
        yield ScrollableContainer(Select([], id="ex_sec"), SelectionList(id="ex_sec_list"))

    def on_mount(self) -> None:
        select = self.query_one("#ex_sec", Select)
        select.border_title = "Model selection"
        avail_ex = self.get_avail_experiments()
        select.border_title = "Model selection"
        select.set_options((line, line) for line in format_data(avail_ex))
        sec_list = self.query_one('#ex_sec_list', SelectionList)
        sec_list.border_title = "Selected features "
        sec_list.add_options([])
        sec_list.disabled = True

    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
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
    table = {'name': "demo", 'column_names': ['WRbwmaxMiB', 'WRbwminMiB'], "parent_table": "results"}

    def __init__(self, conn) -> None:
        self.conn = conn
        super().__init__()

    def create_new_table(self, clean=True):
        smt = ""
        if clean:
            where_clause = ' OR '.join([f'{column} IS NOT NULL' for column in self.table['column_names']])
            smt = "CREATE TABLE {t_name} AS SELECT {t_columns} FROM {t_parent_table} WHERE {wc}".format(
                t_name=self.table['name'],
                t_columns=', '.join(
                    self.table['column_names']),
                t_parent_table=self.table[
                    'parent_table'],
                wc=where_clause)
        else:
            smt = "CREATE TABLE {t_name} AS SELECT {t_columns} FROM {t_parent_table}".format(t_name=self.table['name'],
                                                                                             t_columns=', '.join(
                                                                                                 self.table[
                                                                                                     'column_names']),
                                                                                             t_parent_table=self.table[
                                                                                                 'parent_table'])
        self.conn.execute(smt)
        result = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (self.table['name'],)).fetchone()
        if result:
            print(f"Table {self.table['name']} was created successfully.")
            select = self.query_one("#table_sec", Select)
            select.set_options((line, line) for line in get_tables(self.conn))
            return self.table['name']
        else:
            print(f"Failed to create table {self.table['name']}.")
            print("new-table:", smt)
            return None

    def compose(self) -> ComposeResult:
        yield ScrollableContainer(Input(id="new_tab_name", value=self.table['name'], placeholder="Enter a number...",
                                        validators=Length(minimum=4, maximum=100)), Select([], id="table_sec"),
                                  SelectionList(id="table_sec_list"), Switch(value=True, id="sw_clean"))
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

    @on(Input.Changed)
    def input_changed(self, event: Input.Changed) -> None:
        self.table['name'] = event.value

    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        s_string = event.value
        self.table['parent_table'] = s_string
        col_names = get_table_columns(self.conn, s_string)
        sl = self.query_one("#table_sec_list", SelectionList)
        sl.clear_options()
        sl.add_options([(name[0], idx, True) for idx, name in enumerate(col_names)])
        self.update_selected_view()

    @on(SelectionList.SelectedChanged)
    def update_selected_view(self) -> None:
        select_list = self.query_one("#table_sec_list", SelectionList)
        self.selected_columns = [str(select_list.get_option_at_index(selected_item).prompt) for selected_item in
                                 select_list.selected]
        self.table['column_names'] = self.selected_columns


class DataViewTab(Static):
    def __init__(self, conn) -> None:
        self.conn = conn
        super().__init__()

    table = ""
    selected_columns = []
    df = pd.DataFrame()

    def compose(self) -> ComposeResult:
        yield ScrollableContainer(Select([], id="dv_table_sec"), SelectionList(id="dv_table_sec_list"))
        yield ScrollableContainer(Pretty(id="dv_pretty", object={}))

    def on_mount(self) -> None:
        select = self.query_one("#dv_table_sec", Select)
        select.border_title = "Origin table:"
        select.set_options((line, line) for line in get_tables(self.conn))
        table_sec_list = self.query_one('#dv_table_sec_list', SelectionList)
        table_sec_list.border_title = "Selected columns:"

    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        self.table = event.value
        col_names = get_table_columns(self.conn, self.table)
        sl = self.query_one("#dv_table_sec_list", SelectionList)
        sl.clear_options()
        sl.add_options([(name[0], idx, True) for idx, name in enumerate(col_names)])
        # self.update_selected_view()

        self.df = pd.read_sql_query("SELECT * from {table}".format(table=self.table), self.conn)
        dv_pretty = self.query_one("#dv_pretty", Pretty)
        dv_pretty.update(self.df.describe())
        """
        col_names = get_table_columns(self.conn, s_string)
        sl = self.query_one("#table_sec_list", SelectionList)
        sl.clear_options()
        sl.add_options([(name[0], idx, True) for idx, name in enumerate(col_names)])
        """

    @on(SelectionList.SelectedChanged)
    def update_selected_view(self) -> None:
        select_list = self.query_one("#dv_table_sec_list", SelectionList)
        self.selected_columns = [str(select_list.get_option_at_index(selected_item).prompt) for selected_item in
                                 select_list.selected]
        dv_pretty = self.query_one("#dv_pretty", Pretty)
        print("selected : ", self.selected_columns)
        dv_pretty.update(self.df[self.selected_columns].describe())


if __name__ == "__main__":
    conn = sqlite3.connect("/Users/flash/Desktop/real_result_database.dat")
    app = TabsApp(conn)
    app.run()
