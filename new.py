from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Footer, Label, Tabs, Input, Select, SelectionList, Button, RadioButton, RadioSet, Static, Pretty, Sparkline, TabbedContent, OptionList
from rich.table import Table
from textual.containers import ScrollableContainer
from textual import on
from textual import log
import sqlite3
from enum import Enum
import json
import re

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


class Type(Enum):
    classification = False
    regression = True


conn = sqlite3.connect("/Users/flash/Desktop/real_result_database.dat")
c_names = conn.execute("select name from pragma_table_info('results');").fetchall()
# f_list = [(name[0], idx, False) for idx, name in enumerate(c_names)]
t_list = [(name[0], idx, False) for idx, name in enumerate(c_names)]
f_list = [(name[0], idx, True) for idx, name in enumerate(c_names)]


class TabsApp(App):
    CSS = """
        Tabs {
            dock: top;
        }
        
        TrainingTab{
            layout: grid;
            grid-size: 1;
            align: center top;
        }
        
        ScrollableContainer {
            layout: grid;
            grid-size: 3;
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
    
OptionList {
}
    
    #ex_sec {
        column-span: 3;
        margin: 1 1;
        width: 1fr;
        height: auto;
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
        with TabbedContent("Training", "Prediction"):
            yield TrainingTab()
            yield PredictionTab()
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(TabbedContent).focus()

"""
    @on(TabbedContent.TabActivated)
    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if event.tab is None:
            # When the tabs are cleared, event.tab will be None
            print("is None")
        else:
            print("is tab", event)

"""


class TrainingTab(Static):
    ml_config = {'name': "Demo", 'type': Type.regression, 'algo': "linear_regression", 'table': "results", 'target': []}

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
                           table=self.ml_config['table'], target=self.ml_config['target'][0])
            print(smt)
            test = conn.execute(smt).fetchone()[0]
            data = json.loads(test)
            self.query_one(Pretty).update(test)
            l = [float(data["score"]) * 100, 22, 39]
            print(float(data["score"]))
            self.query_one(Sparkline).data = l

    def compose(self) -> ComposeResult:
        yield Input(placeholder="ML Workflow name", type="text", value=self.ml_config['name'])
        yield ScrollableContainer(RadioSet(RadioButton("Regression", value=self.ml_config['type'].value),
                                           RadioButton("Classification", value=not self.ml_config['type'].value)),
                                  Select((line, line) for line in LINES), SelectionList[int](id="test2"))
        yield ScrollableContainer(Button(id="training-bt", label="Training"), Pretty(""), Sparkline())
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(Input).border_title = "ML Task name:"
        self.query_one(RadioSet).border_title = "ML Task Type:"
        self.query_one(Select).border_title = "ML Task Algorithm:"
        """
        self.query_one("#test", SelectionList).border_title="feature selection"
        self.query_one("#test", SelectionList).add_options(f_list)
        self.query_one("#test", SelectionList).visible = False
        """
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
        """
        select_f = self.query_one("#test", SelectionList)
        select_f.visible = True
        """

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

    @on(SelectionList.SelectedChanged)
    def update_selected_view(self) -> None:
        selected_items = self.query_one(SelectionList).selected
        print(selected_items)
        selected = [t_list[idx][0] for idx in selected_items] if selected_items else []
        self.ml_config['target'] = selected
        print(self.ml_config['target'])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.generate_ml_sql()


def format_data(sql_results):
    formatted_list = []
    for entry in sql_results:
        print("entries: ", entry)

        experiment_id, name, c_date, u_date, algorithm, dataset, target = entry
        formatted_string = f"ID: {experiment_id} / NAME: {name} / Algorithm: {algorithm} / Dataset: {dataset} / Target: {target} - DATE: [{c_date}::{u_date}]"
        """
        id, c_date, u_date, name, prediction_type = entry
        formatted_string = f"ID: {id} / NAME: {name} / Type: {prediction_type} - DATE: [{c_date}::{u_date}]"
        """

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
    predict_config = {'name': "Demo877888", 'table': "results", 'feature_str': ""}

    def generate_predict_sql(self, is_training=True):
        from sqlite_ml.sqml import SQML
        # setup sqlite-ml extension
        sqml = SQML()
        sqml.setup_schema(conn)
        sqml.register_functions(conn)
        sub_smt = "json_object(" + ", ".join(re.escape("'")+f[0]+re.escape("'")+', [' + f[0] + ']' for f in f_list[:-1]) + ',' + re.escape("'")+f_list[-1][0] +re.escape("'") +', [' + f_list[-1][0] + '])'
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
              {features}
            )
          FROM
            {table}
        )
      ) batch ON (batch.rowid + 1) = {table}.rowid
    WHERE match = True;
        """.format(name=self.predict_config['name'], table=self.predict_config['table'], features=sub_smt, target="WRbwminMiB")
        print(smt_batched)
        test = conn.execute(smt_batched).fetchone()[0]
        """
        data = json.loads(test)
        self.query_one(Pretty).update(test)
        l = [float(data["score"]) * 100, 22, 39]
        print(float(data["score"]))
        self.query_one(Sparkline).data = l

        """

    def compose(self) -> ComposeResult:
        yield ScrollableContainer(Select([], id="ex_sec"), SelectionList(id="ex_sec_list"))
        yield ScrollableContainer(Button(id="predict-bt", label="Prediction"), Pretty(""), Sparkline())#

    def on_mount(self) -> None:
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
        results = conn.execute(s_smt).fetchall()
        """
        results = conn.execute("select * from ('sqml_experiments');").fetchall()
        """
        select = self.query_one("#ex_sec", Select)
        select.border_title = "Model selection"
        select.set_options((line, line) for line in format_data(results))
        sec_list = self.query_one('#ex_sec_list', SelectionList)
        sec_list.border_title = "Selected features "
        sec_list.add_options(f_list)
        sec_list.visible=False

    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        s_string = event.value
        print("new_string", s_string)
        select_ex = parse_experiment_to_dic(s_string)
        sec_list = self.query_one('#ex_sec_list', SelectionList)
        sec_list.visible = False
        """
        for i, f in enumerate(f_list):
            if str(f[0]) == select_ex['target']:  # Assuming target is the second element in the tuple
                f_l = list(f)
                f_l[2] = False  # Update the value
                f_list[i] = tuple(f_l)  # Assign the modified tuple back to the list
                print("found:", f_list[i])
        
        """
        sec_list.clear_options()
        b_f_list = [f for f in f_list if str(f[0]) != select_ex['target']]
        sec_list.add_options(b_f_list)
        """
        print("print-fs")
        for n in b_f_list:
            print(n)
        """
        sec_list.visible = True

        """
        get_target_smt = "select target from ('sqml_runs') where experiment_id = {ex_id} and updated_at = '{ex_u_date}';"
        .format(ex_id=select_ex['ex_id'], ex_u_date=select_ex['ex_u_date'])
        print("get_target_smt:", get_target_smt)
        """
        """
        select_f = self.query_one("#test", SelectionList)
        select_f.visible = True
        """
    @on(SelectionList.SelectedChanged)
    def update_selected_view(self) -> None:
        selected_items = self.query_one('#ex_sec_list', SelectionList).selected
        self.predict_config = [f_list[idx][0] for idx in selected_items] if selected_items else []

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.generate_predict_sql()





"""
    ml_ex = []
    d_rows: tuple[tuple[str, str, str, str], ...] = ()
    
    def on_mount(self) -> None:
        results = conn.execute("select created_at, updated_at, name, prediction_type from ('sqml_experiments');").fetchall()
        if not all(len(row) == 4 for row in results):
            raise ValueError("All rows must have exactly four columns")
        d_rows = tuple(results)
        option_list.add_options([self.data_table(*row) for row in d_rows])
    
    @staticmethod
    def data_table(c_date: str, u_date: str, name: str, type: str) -> Table:
        table = Table(title=f"Data for {name}", expand=True)
        table.add_column("Created Date")
        table.add_column("Updated Date")
        table.add_column("ML Type")
        table.add_row(c_date, u_date, type)
        return table

"""


if __name__ == "__main__":
    app = TabsApp()
    app.run()

