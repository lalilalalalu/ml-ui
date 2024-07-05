from textual.app import App, ComposeResult
from textual.widgets import Footer, Label, Tabs, Input, Select, SelectionList, Button, RadioButton, RadioSet, Static
from textual.containers import ScrollableContainer
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


conn = sqlite3.connect("/Users/flash/Desktop/real_result_database.dat")


class TabsApp(App):

    ml_config = {'name': "Demo", 'type': Type.regression, 'algo': "linear_regression", 'table': "resultsMY", 'target': []}
    c_names = conn.execute("select name from pragma_table_info('resultsMY');").fetchall()
    # f_list = [(name[0], idx, False) for idx, name in enumerate(c_names)]
    t_list = [(name[0], idx, False) for idx, name in enumerate(c_names)]
    def generate_ml_sql(self, is_training = True):
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
            """.format(name=self.ml_config['name'], type=self.ml_config['type'].name, algo=self.ml_config['algo'], table= self.ml_config['table'], target=self.ml_config['target'][0])
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
                                  Select((line, line) for line in LINES),  SelectionList[int](id="test2"))
        yield Button(id="training-bt", label="Training")
        yield Footer()

    def on_mount(self) -> None:

        self.query_one(Tabs).focus()
        self.query_one(Input).border_title="ML Task name:"
        self.query_one(RadioSet).border_title = "ML Task Type:"
        self.query_one(Select).border_title = "ML Task Algorithm:"
        """
        self.query_one("#test", SelectionList).border_title="feature selection"
        self.query_one("#test", SelectionList).add_options(f_list)
        self.query_one("#test", SelectionList).visible = False
        """
        self.query_one("#test2", SelectionList).border_title = "target selection"
        self.query_one("#test2", SelectionList).add_options(self.t_list)
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
        selected = [self.t_list[idx][0] for idx in selected_items] if selected_items else []
        self.ml_config['target'] = selected
        print(self.ml_config['target'])

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

