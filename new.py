from textual.app import App, ComposeResult
from textual.widgets import Footer, Label, Tabs, Input, Select, SelectionList, Button, RadioButton, RadioSet, Static
from textual import on
from textual import log
import sqlite3
from enum import Enum



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


class TargetAndFeatures(Static):
    def compose(self) -> ComposeResult:
        yield SelectionList[int](id="test")
        yield SelectionList[int](id="test2")



class TabsApp(App):
    conn = sqlite3.connect("/Users/flash/Desktop/real_result_database.dat")
    ml_config = {'name': "Demo", 'type': Type.regression, 'algo': "linear_regression", 'table': "resultsMY", 'target': "WRbwmaxMiB"}

    def generate_ml_sql(self, is_training = True):
        from sqlite_ml.sqml import SQML
        # setup sqlite-ml extension
        sqml = SQML()
        sqml.setup_schema(self.conn)
        sqml.register_functions(self.conn)
        # execute sqlite-ml functions
        print(self.conn.execute("SELECT sqml_python_version();").fetchone()[0])
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
            """.format(name=self.ml_config['name'], type=self.ml_config['type'].name, algo=self.ml_config['algo'], table= self.ml_config['table'], target=self.ml_config['target'])
            print(smt)

    CSS = """
    Tabs {
        dock: top;
    }
    Screen {
        layout: grid;
        grid-size: 2;
        align: center top;
    }

    Input {
        margin:1 1;
        height: 100%;
        background: $panel;
        border: tall $primary;
        content-align: center middle;
    }
        
    Select {
        margin:1 1;
        height: 100%;
        background: $panel;
        border: tall $primary;
        content-align: center middle;
    }
    
    SelectionList {
    padding: 1;
    border: solid $accent;
    width: 1fr;
}

    #two {
    column-span: 2;
    tint: magenta 40%;
}
    """

    def compose(self) -> ComposeResult:
        yield Tabs(Steps[0], Steps[1])
        yield Input(placeholder="ML Workflow name", type="text", value=self.ml_config['name'])
        with RadioSet():
            yield RadioButton("Regression", value=self.ml_config['type'].value)
            yield RadioButton("Classification", value=not self.ml_config['type'].value)
        yield Select((line, line) for line in LINES)
        yield SelectionList[int](id="test")
        yield SelectionList[int](id="test2")

        yield Button("Training")
        yield Footer()

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

    def on_mount(self) -> None:
        c_names = self.conn.execute("select name from pragma_table_info('resultsMY');").fetchall()
        f_list = [(name[0], idx, False) for idx, name in enumerate(c_names)]
        t_list = [(name[0], idx, False) for idx, name in enumerate(c_names)]
        self.query_one(Tabs).focus()
        self.query_one("#test", SelectionList).border_title="feature selection"
        self.query_one("#test", SelectionList).add_options(f_list)
        self.query_one("#test", SelectionList).visible = False
        self.query_one("#test2", SelectionList).border_title = "target selection"
        self.query_one("#test2", SelectionList).add_options(t_list)
        self.query_one("#test2", SelectionList).visible = False


    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        if event.tab is None:
            # When the tabs are cleared, event.tab will be None
            print("is None")
        else:
            print("is tab", event.tab)


    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.generate_ml_sql()


if __name__ == "__main__":
    app = TabsApp()
    app.run()

